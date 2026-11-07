#!/usr/bin/env python3

"""Numerically validate the bounded-memory D-TRAK projection implementation."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dtrak_extract_features import (
    build_gradient_chunks,
    create_projector,
    create_chunked_projectors,
    project_gradient_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def project(
    grads,
    chunks,
    projectors,
    device: torch.device,
) -> torch.Tensor:
    return project_gradient_chunks(
        grads=grads,
        chunks=chunks,
        projectors=projectors,
        proj_dim=512,
        model_id=0,
        device=device,
        kernel_batch_size=8,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(1234)

    # Deliberately force chunks to cross parameter boundaries and split a parameter.
    params = [
        torch.empty(5, device=device),
        torch.empty(13, device=device),
        torch.empty(4, 7, device=device),
        torch.empty(9, device=device),
    ]
    chunks = build_gradient_chunks(params, max_chunk_numel=11)
    projectors, chunk_seeds = create_chunked_projectors(
        chunks=chunks,
        proj_dim=512,
        seed=42,
        device=device,
        kernel_batch_size=8,
    )

    expected_coordinates = [
        (param_index, element_index)
        for param_index, parameter in enumerate(params)
        for element_index in range(parameter.numel())
    ]
    observed_coordinates = [
        (param_index, element_index)
        for chunk_slices, _ in chunks
        for param_index, flat_start, flat_end in chunk_slices
        for element_index in range(flat_start, flat_end)
    ]
    coverage_exact = observed_coordinates == expected_coordinates

    grads_a = tuple(torch.randn_like(parameter) for parameter in params)
    grads_b = tuple(torch.randn_like(parameter) for parameter in params)
    grads_sum = tuple(a + b for a, b in zip(grads_a, grads_b))

    projected_a = project(grads_a, chunks, projectors, device)
    projected_a_repeat = project(grads_a, chunks, projectors, device)
    projected_b = project(grads_b, chunks, projectors, device)
    projected_sum = project(grads_sum, chunks, projectors, device)

    determinism_max_abs = float((projected_a - projected_a_repeat).abs().max().item())
    linearity_relative_l2 = float(
        torch.linalg.vector_norm(projected_sum - projected_a - projected_b).item()
        / max(torch.linalg.vector_norm(projected_sum).item(), 1e-12)
    )

    # Calibrate the tolerance against the existing monolithic fast_jl kernel.
    # fast_jl's parallel float32 reduction is deterministic but not exactly linear
    # in floating-point arithmetic, even without chunking.
    flat_a = torch.cat([value.reshape(-1) for value in grads_a])
    flat_b = torch.cat([value.reshape(-1) for value in grads_b])
    full_input_a = torch.zeros((8, flat_a.numel()), dtype=torch.float32, device=device)
    full_input_b = torch.zeros_like(full_input_a)
    full_input_sum = torch.zeros_like(full_input_a)
    full_input_a[0].copy_(flat_a)
    full_input_b[0].copy_(flat_b)
    full_input_sum[0].copy_(flat_a + flat_b)
    legacy_projector = create_projector(
        grad_dim=flat_a.numel(),
        proj_dim=512,
        seed=42,
        device=device,
        max_batch_size=8,
    )
    legacy_a = legacy_projector.project(full_input_a, model_id=0)[0]
    legacy_b = legacy_projector.project(full_input_b, model_id=0)[0]
    legacy_sum = legacy_projector.project(full_input_sum, model_id=0)[0]
    legacy_linearity_relative_l2 = float(
        torch.linalg.vector_norm(legacy_sum - legacy_a - legacy_b).item()
        / max(torch.linalg.vector_norm(legacy_sum).item(), 1e-12)
    )

    # A Gaussian JL map should preserve the geometry of a small collection of
    # vectors. Cosines remove the project-wide scale used by fast_jl.
    source_rows = []
    projected_rows = []
    for _ in range(16):
        sample = tuple(torch.randn_like(parameter) for parameter in params)
        source_rows.append(torch.cat([value.reshape(-1) for value in sample]))
        projected_rows.append(project(sample, chunks, projectors, device))
    source = F.normalize(torch.stack(source_rows), dim=1)
    projected_values = F.normalize(torch.stack(projected_rows), dim=1)
    source_cosines = source @ source.T
    projected_cosines = projected_values @ projected_values.T
    off_diagonal = ~torch.eye(source.shape[0], dtype=torch.bool, device=device)
    cosine_errors = (source_cosines - projected_cosines).abs()[off_diagonal]
    cosine_mae = float(cosine_errors.mean().item())
    cosine_max_error = float(cosine_errors.max().item())

    passed = (
        coverage_exact
        and determinism_max_abs == 0.0
        and linearity_relative_l2 < max(1e-3, 2.0 * legacy_linearity_relative_l2)
        and cosine_mae < 0.08
        and cosine_max_error < 0.25
    )
    summary = {
        "status": "pass" if passed else "fail",
        "coverage_exact": coverage_exact,
        "num_chunks": len(chunks),
        "chunk_sizes": [size for _, size in chunks],
        "chunk_seeds": chunk_seeds,
        "determinism_max_abs": determinism_max_abs,
        "linearity_relative_l2": linearity_relative_l2,
        "legacy_linearity_relative_l2": legacy_linearity_relative_l2,
        "jl_cosine_mae": cosine_mae,
        "jl_cosine_max_error": cosine_max_error,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError("Chunked projection validation failed.")


if __name__ == "__main__":
    main()
