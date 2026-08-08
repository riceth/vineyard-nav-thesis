"""Deterministic cuDNN preload — import this BEFORE `import torch`.

Why (reproducibility, D049): on this WSL2 + Blackwell (sm_120) + cuDNN 9 stack the pip-wheel cuDNN
libraries (nvidia-cudnn-cu12: `libcudnn.so.9` + its `libcudnn_*.so.9` engines) live under
site-packages/nvidia/cudnn/lib, which is NOT on the dynamic-loader path. PyTorch preloads them itself
at `import torch`, but on a cold process that preload is intermittently missed and the first cuDNN
call aborts with
    Invalid handle. Cannot load symbol cudnnGetVersion
(a fresh terminal crashes; a warm one does not). CDLL(RTLD_GLOBAL)-loading the cuDNN family here makes
`cudnnGetVersion` resolve deterministically for torch + ultralytics regardless of load order.

Scope is deliberately the cuDNN directory ONLY. torch reliably preloads the other CUDA wheels
(cublas, cudart, …) itself — only cuDNN is flaky — and force-loading the whole `nvidia/` tree also
pulls in `libnvblas.so`, a BLAS-interception library that, once in the RTLD_GLOBAL namespace, hijacks
CPU BLAS calls and segfaults without an `nvblas.conf` (observed on `line_fit_infer` arm B). And it is
safe to scope this narrowly: `libcudnn.so.9`'s declared dependencies are all system libs (its engine
sub-libs are dlopen'd at runtime from $ORIGIN), so the cuDNN family loads standalone. Idempotent,
stdlib-only, and a silent no-op if the cuDNN wheel is absent (CPU install).
"""
from __future__ import annotations
import ctypes, glob, os


def preload() -> list[str]:
    """CDLL(RTLD_GLOBAL) the cuDNN wheel libraries (`libcudnn.so.9` + engines), and ONLY those.
    Returns the paths actually loaded (empty if nvidia-cudnn-cu12 isn't installed)."""
    try:
        import nvidia
    except ImportError:
        return []
    libdir = os.path.join(os.path.dirname(nvidia.__file__), "cudnn", "lib")
    # engine sub-libs first, the libcudnn.so.9 dispatcher last (it dlopens the engines at runtime)
    sos = sorted(glob.glob(os.path.join(libdir, "*.so*")),
                 key=lambda p: ("libcudnn.so" in os.path.basename(p), p))
    loaded: list[str] = []
    for _ in range(2):                                          # 2 passes: resolve inter-lib deps
        for so in sos:
            if so in loaded:
                continue
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                loaded.append(so)
            except OSError:
                pass
    return loaded


preload()
