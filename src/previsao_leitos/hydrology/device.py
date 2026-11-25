"""Resolução preguiçosa (lazy) de acelerador GPU/CPU (ROCm, CUDA ou CPU-only).

Garante conformidade com C-07: importar este módulo não importa `torch` nem inicializa
contexto GPU até que `resolve_accelerator()` ou `release_gpu_memory()` seja chamado.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Literal

DeviceRequest = Literal["auto", "cpu", "rocm", "cuda"]
BackendKind = Literal["cpu", "rocm", "cuda"]


@dataclass(frozen=True)
class AcceleratorSpec:
    """Especificação determinística do acelerador resolvido em tempo de execução."""

    accelerator: Literal["cpu", "gpu"]
    device_str: Literal["cpu", "cuda"]
    backend: BackendKind
    device_name: str | None
    runtime_version: str | None
    precision: Literal["32-true", "bf16-mixed", "16-mixed"]
    deterministic_mode: Literal["strict", "warn"]


def resolve_accelerator(
    *,
    prefer_gpu: bool = True,
    requested_device: DeviceRequest = "auto",
    precision: Literal["32-true", "bf16-mixed", "16-mixed"] = "32-true",
) -> AcceleratorSpec:
    """Resolve acelerador de forma lazy (CPU no CI; ROCm/CUDA na workstation/Colab)."""
    if requested_device == "cpu" or not prefer_gpu:
        return AcceleratorSpec(
            accelerator="cpu",
            device_str="cpu",
            backend="cpu",
            device_name=None,
            runtime_version=None,
            precision="32-true",
            deterministic_mode="strict",
        )

    import torch

    if not torch.cuda.is_available():
        return AcceleratorSpec(
            accelerator="cpu",
            device_str="cpu",
            backend="cpu",
            device_name=None,
            runtime_version=None,
            precision="32-true",
            deterministic_mode="strict",
        )

    hip_ver = getattr(torch.version, "hip", None)
    cuda_ver = getattr(torch.version, "cuda", None)
    detected_backend: BackendKind = "rocm" if hip_ver is not None else "cuda"

    if requested_device == "rocm" and detected_backend != "rocm":
        return AcceleratorSpec(
            accelerator="cpu",
            device_str="cpu",
            backend="cpu",
            device_name=None,
            runtime_version=None,
            precision="32-true",
            deterministic_mode="strict",
        )
    if requested_device == "cuda" and detected_backend != "cuda":
        return AcceleratorSpec(
            accelerator="cpu",
            device_str="cpu",
            backend="cpu",
            device_name=None,
            runtime_version=None,
            precision="32-true",
            deterministic_mode="strict",
        )

    if detected_backend == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    elif detected_backend == "rocm":
        # Evita falha JIT do MIOpenDropoutHIP em nn.GRU com dropout quando headers rocrand
        # não estão instalados; usa kernels nativos HIP/rocBLAS diretamente na GPU AMD.
        torch.backends.cudnn.enabled = False

    torch.set_float32_matmul_precision("high")

    dev_name = str(torch.cuda.get_device_name(0))
    rt_ver = str(hip_ver) if hip_ver is not None else (str(cuda_ver) if cuda_ver else None)
    return AcceleratorSpec(
        accelerator="gpu",
        device_str="cuda",
        backend=detected_backend,
        device_name=dev_name,
        runtime_version=rt_ver,
        precision=precision,
        deterministic_mode="warn",
    )


def release_gpu_memory() -> None:
    """Libera referências Python e cache de VRAM entre folds/seeds."""
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
