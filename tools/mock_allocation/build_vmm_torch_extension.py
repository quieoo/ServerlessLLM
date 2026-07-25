#!/usr/bin/env python3
"""Build the non-owning PyTorch CUDA tensor wrapper for Tangram VMM."""

from pathlib import Path

from torch.utils.cpp_extension import load


root = Path(__file__).resolve().parent
build_dir = root / "build" / "torch_extensions" / "tangram_vmm_torch"
build_dir.mkdir(parents=True, exist_ok=True)
module = load(
    name="tangram_vmm_torch",
    sources=[str(root / "src" / "tangram_vmm_torch.cpp")],
    extra_cflags=["-O2"],
    build_directory=str(build_dir),
    verbose=True,
)
print(module.__file__)

