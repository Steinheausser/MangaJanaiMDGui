"""
Small custom HIP kernels for ROCm, compiled at runtime with HIPRTC for the GPU that is present.

Uses the HIP runtime and HIPRTC DLLs that ship with ROCm PyTorch (rocm_sdk_core), so no SDK or
compiler is needed on the user's machine. Compiled code objects are cached on disk.

Kernels:
- depthwise 3x3 convolution (stride 1, padding 1) on channels-last data (B, H, W, C), with optional
  fused bias and GELU. MIOpen's fast depthwise path is NCHW-only, so token-layout models such as FDAT
  otherwise pay two full-tensor transposes per depthwise conv.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import threading
from pathlib import Path

import torch
from sanic.log import logger

_DWCONV_SOURCE = r"""
typedef SCALAR scalar_t;
typedef SCALAR vec_t __attribute__((ext_vector_type(VEC)));

extern "C" __global__ __launch_bounds__(256) void dwconv3x3_nhwc(
    const scalar_t* __restrict__ x,     // (B, H, W, C)
    const scalar_t* __restrict__ w9,    // (9, C): tap-major weights
    const scalar_t* __restrict__ bias,  // (C) or unused
    scalar_t* __restrict__ y,           // (B, H, W, C)
    int B, int H, int W, int C, int has_bias, int apply_gelu)
{
    const int cvecs = C / VEC;
    const long long total = (long long)B * H * W * cvecs;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const int cv = (int)(i % cvecs);
    const long long p = i / cvecs;
    const int px = (int)(p % W);
    const long long t = p / W;
    const int py = (int)(t % H);
    const int b = (int)(t / H);
    const int c0 = cv * VEC;

    float acc[VEC];
#pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = has_bias ? (float)bias[c0 + v] : 0.0f;

#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = py + dy;
        if (yy < 0 || yy >= H) continue;
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = px + dx;
            if (xx < 0 || xx >= W) continue;
            const vec_t xv = *(const vec_t*)(x + (((long long)b * H + yy) * W + xx) * C + c0);
            const vec_t wv = *(const vec_t*)(w9 + ((dy + 1) * 3 + (dx + 1)) * C + c0);
#pragma unroll
            for (int v = 0; v < VEC; ++v) acc[v] += (float)xv[v] * (float)wv[v];
        }
    }

    vec_t out;
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
        float a = acc[v];
        if (apply_gelu) a = 0.5f * a * (1.0f + erff(a * 0.70710678118654752f));
        out[v] = (scalar_t)a;
    }
    *(vec_t*)(y + p * C + c0) = out;
}
"""

_LAYERNORM_SOURCE = r"""
typedef SCALAR scalar_t;

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) v += __shfl_xor(v, offset, WARP);
    return v;
}

extern "C" __global__ __launch_bounds__(256) void layernorm_rows(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y, long long rows, int C, int has_bias, float eps)
{
    const int lane = threadIdx.x % WARP;
    const long long row = (long long)blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= rows) return;
    const scalar_t* xr = x + row * C;

    float v[MAXE];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXE; ++i) {
        const int c = lane + i * WARP;
        v[i] = c < C ? (float)xr[c] : 0.0f;
        sum += v[i];
    }
    const float mean = wave_sum(sum) / C;
    float sq = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXE; ++i) {
        const int c = lane + i * WARP;
        const float d = c < C ? v[i] - mean : 0.0f;
        sq += d * d;
    }
    const float rstd = rsqrtf(wave_sum(sq) / C + eps);
    scalar_t* yr = y + row * C;
#pragma unroll
    for (int i = 0; i < MAXE; ++i) {
        const int c = lane + i * WARP;
        if (c < C) {
            float o = (v[i] - mean) * rstd * (float)weight[c];
            if (has_bias) o += (float)bias[c];
            yr[c] = (scalar_t)o;
        }
    }
}
"""

_WINDOW_ATTENTION_SOURCE = r"""
typedef SCALAR scalar_t;

// One block per (window, head), one thread per query token of the window (WIN = WS * WS threads).
// q/k/v are read straight from the qkv projection in token layout (B, H*W, 3, NH, D) and the result is
// written in token layout (B, H*W, NH * D), so no window partition/reverse copies are needed.
extern "C" __global__ __launch_bounds__(WIN) void window_attention(
    const scalar_t* __restrict__ qkv, const scalar_t* __restrict__ bias,  // bias: (NH, WIN, WIN)
    scalar_t* __restrict__ out, int H, int W, float scale)
{
    __shared__ float ks[WIN][D];
    __shared__ float vs[WIN][D];

    const int head = blockIdx.x % NH;
    const int window = blockIdx.x / NH;
    const int windows_x = W / WS;
    const int windows_per_image = (H / WS) * windows_x;
    const int b = window / windows_per_image;
    const int wy = (window % windows_per_image) / windows_x;
    const int wx = window % windows_x;

    const int t = threadIdx.x;
    const int y = wy * WS + t / WS;
    const int x = wx * WS + t % WS;
    const long long token = (long long)b * H * W + (long long)y * W + x;
    const int C = NH * D;
    const scalar_t* row = qkv + token * 3 * C + head * D;

    float q[D];
#pragma unroll
    for (int k = 0; k < D; ++k) {
        q[k] = (float)row[k] * scale;
        ks[t][k] = (float)row[C + k];
        vs[t][k] = (float)row[2 * C + k];
    }
    __syncthreads();

    const scalar_t* bias_row = bias + ((long long)head * WIN + t) * WIN;
    float s[WIN];
    float m = -3.402823466e+38f;
#pragma unroll
    for (int j = 0; j < WIN; ++j) {
        float acc = (float)bias_row[j];
#pragma unroll
        for (int k = 0; k < D; ++k) acc += q[k] * ks[j][k];
        s[j] = acc;
        m = fmaxf(m, acc);
    }
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < WIN; ++j) {
        s[j] = __expf(s[j] - m);
        sum += s[j];
    }
    const float inv = 1.0f / sum;

    float o[D];
#pragma unroll
    for (int k = 0; k < D; ++k) o[k] = 0.0f;
#pragma unroll
    for (int j = 0; j < WIN; ++j) {
        const float p = s[j] * inv;
#pragma unroll
        for (int k = 0; k < D; ++k) o[k] += p * vs[j][k];
    }

    scalar_t* out_row = out + token * C + head * D;
#pragma unroll
    for (int k = 0; k < D; ++k) out_row[k] = (scalar_t)o[k];
}
"""

_SCALAR_NAMES = {torch.float16: "_Float16", torch.float32: "float"}

_HIP_SUCCESS = 0
_HIPRTC_SUCCESS = 0


class _Hip:
    """ctypes bindings to the HIP runtime and HIPRTC that torch already loaded."""

    def __init__(self) -> None:
        site_packages = Path(torch.__file__).parent.parent
        search_dirs = [site_packages / "_rocm_sdk_core" / "bin", Path(torch.__file__).parent / "lib"]

        # Bind to the HIP runtime torch already loaded: a second copy of the runtime would have its
        # own device context, and torch's device pointers would be invalid in it.
        self.hip = None
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        for directory in search_dirs:
            for dll in sorted(glob.glob(str(directory / "amdhip64*.dll"))):
                handle = kernel32.GetModuleHandleW(os.path.basename(dll))
                if handle:
                    self.hip = ctypes.CDLL(dll, handle=handle)
                    break
            if self.hip is not None:
                break
        if self.hip is None:
            raise RuntimeError("the HIP runtime loaded by torch was not found")

        rtc_dll = [p for d in search_dirs for p in sorted(glob.glob(str(d / "hiprtc*.dll"))) if "builtins" not in p]
        if not rtc_dll:
            raise RuntimeError("HIPRTC not found")
        self.rtc = ctypes.CDLL(rtc_dll[0])

        vp, sz, cp = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p
        self.rtc.hiprtcCreateProgram.argtypes = [ctypes.POINTER(vp), cp, cp, ctypes.c_int, ctypes.POINTER(cp), ctypes.POINTER(cp)]
        self.rtc.hiprtcCompileProgram.argtypes = [vp, ctypes.c_int, ctypes.POINTER(cp)]
        self.rtc.hiprtcGetProgramLogSize.argtypes = [vp, ctypes.POINTER(sz)]
        self.rtc.hiprtcGetProgramLog.argtypes = [vp, cp]
        self.rtc.hiprtcGetCodeSize.argtypes = [vp, ctypes.POINTER(sz)]
        self.rtc.hiprtcGetCode.argtypes = [vp, cp]
        self.rtc.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(vp)]
        self.hip.hipModuleLoadData.argtypes = [ctypes.POINTER(vp), vp]
        self.hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, cp]
        self.hip.hipModuleLaunchKernel.argtypes = [
            vp, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, vp, ctypes.POINTER(vp), ctypes.POINTER(vp),
        ]

    def compile(self, source: str, name: str, options: list[str]) -> bytes:
        prog = ctypes.c_void_p()
        _check_rtc(self.rtc.hiprtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None), "create")
        try:
            opts = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
            res = self.rtc.hiprtcCompileProgram(prog, len(options), opts)
            if res != _HIPRTC_SUCCESS:
                log_size = ctypes.c_size_t()
                self.rtc.hiprtcGetProgramLogSize(prog, ctypes.byref(log_size))
                log = ctypes.create_string_buffer(log_size.value + 1)
                self.rtc.hiprtcGetProgramLog(prog, log)
                raise RuntimeError(f"HIPRTC compile failed ({res}): {log.value.decode(errors='replace')}")
            code_size = ctypes.c_size_t()
            _check_rtc(self.rtc.hiprtcGetCodeSize(prog, ctypes.byref(code_size)), "code size")
            code = ctypes.create_string_buffer(code_size.value)
            _check_rtc(self.rtc.hiprtcGetCode(prog, code), "get code")
            return code.raw
        finally:
            self.rtc.hiprtcDestroyProgram(ctypes.byref(prog))

    def load_function(self, code: bytes, name: str) -> tuple[ctypes.c_void_p, ctypes.c_void_p, bytes]:
        module = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(code, len(code))
        _check_hip(self.hip.hipModuleLoadData(ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)), "module load")
        fn = ctypes.c_void_p()
        _check_hip(self.hip.hipModuleGetFunction(ctypes.byref(fn), module, name.encode()), "get function")
        return module, fn, buf


def _check_rtc(res: int, what: str) -> None:
    if res != _HIPRTC_SUCCESS:
        raise RuntimeError(f"HIPRTC {what} failed with error {res}")


def _check_hip(res: int, what: str) -> None:
    if res != _HIP_SUCCESS:
        raise RuntimeError(f"HIP {what} failed with error {res}")


def _cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    path = Path(base) / "MangaJaNaiConverterGui" / "hip_kernels"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _KernelCache:
    def __init__(self) -> None:
        self._hip: _Hip | None = None
        self._functions: dict[tuple, ctypes.c_void_p] = {}
        self._keepalive: list = []
        self._lock = threading.Lock()
        self.lock = threading.RLock()
        self.verified_devices: set[int | None] = set()
        self.disabled_reason: str | None = None

    def get(self, device: torch.device, name: str, source: str, defines: dict[str, object]) -> ctypes.c_void_p:
        """Compiles (or loads from the disk cache) kernel `name` from `source` with the given #defines."""
        key = (_device_index(device), name, tuple(sorted(defines.items())))
        fn = self._functions.get(key)
        if fn is not None:
            return fn
        with self._lock:
            fn = self._functions.get(key)
            if fn is None:
                fn = self._build(device, name, source, defines)
                self._functions[key] = fn
        return fn

    def _build(self, device: torch.device, name: str, source: str, defines: dict[str, object]) -> ctypes.c_void_p:
        if self._hip is None:
            self._hip = _Hip()
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        source = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + source
        options = [f"--offload-arch={arch}", "-O3"]
        digest = hashlib.sha1((source + "|".join(options) + str(torch.version.hip)).encode()).hexdigest()[:16]
        cache_file = _cache_dir() / f"{name}_{arch}_{digest}.co"

        if cache_file.is_file():
            try:
                with torch.cuda.device(device):
                    module, fn, buf = self._hip.load_function(cache_file.read_bytes(), name)
                self._keepalive.append((module, buf))
                return fn
            except Exception as e:
                # corrupt or stale cache entry: rebuild it
                logger.warning(f"Discarding cached HIP kernel {cache_file.name}: {e}")
                cache_file.unlink(missing_ok=True)

        logger.info(f"Compiling HIP kernel {name} for {arch} ({defines})")
        code = self._hip.compile(source, f"{name}.hip", options)
        with torch.cuda.device(device):
            module, fn, buf = self._hip.load_function(code, name)
        self._keepalive.append((module, buf))
        try:
            tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(code)
            os.replace(tmp, cache_file)
        except OSError as e:
            logger.warning(f"Could not cache HIP kernel {cache_file.name}: {e}")
        return fn

    def launch(self, fn: ctypes.c_void_p, grid: int, block: int, device: torch.device, args: list) -> None:
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        assert self._hip is not None
        if _device_index(device) != torch.cuda.current_device():
            # the module and stream belong to `device`; launch with it current
            with torch.cuda.device(device):
                res = self._hip.hip.hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, params, None)
        else:
            res = self._hip.hip.hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, params, None)
        _check_hip(res, "launch")


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


_kernels = _KernelCache()


def _pick_vec(c: int, dtype: torch.dtype) -> int:
    for vec in (8, 4, 2) if dtype == torch.float16 else (4, 2):
        if c % vec == 0:
            return vec
    return 0


def kernels_enabled(device: torch.device) -> bool:
    """Custom kernels are usable on this device (ROCm only; self-tested once, disabled on failure)."""
    if _kernels.disabled_reason is not None:
        return False
    if os.environ.get("MANGAJANAI_DISABLE_HIP_KERNELS") == "1":
        return False
    if device.type != "cuda" or getattr(torch.version, "hip", None) is None:
        return False
    index = _device_index(device)
    if index not in _kernels.verified_devices:
        with _kernels.lock:
            if index not in _kernels.verified_devices and _kernels.disabled_reason is None:
                with torch.cuda.device(index):
                    if warmup_or_disable(torch.device("cuda", index)):
                        _kernels.verified_devices.add(index)
        return _kernels.disabled_reason is None
    return True


def dwconv3x3_nhwc_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether dwconv3x3_nhwc can run this depthwise conv: x is (B, H, W, C), weight is (C, 1, 3, 3)."""
    if not kernels_enabled(x.device):
        return False
    if x.dtype not in _SCALAR_NAMES or weight.dtype != x.dtype:
        return False
    c = x.shape[-1]
    return tuple(weight.shape) == (c, 1, 3, 3) and _pick_vec(c, x.dtype) > 0


def dwconv3x3_nhwc(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, gelu: bool = False
) -> torch.Tensor:
    """
    Depthwise 3x3 conv, stride 1, zero padding 1, on channels-last data.
    x: (B, H, W, C), weight: (C, 1, 3, 3), bias: (C) or None. Returns a contiguous (B, H, W, C) tensor.
    Optionally applies the exact (erf) GELU to the result.
    """
    b, h, w, c = x.shape
    x = x.contiguous()
    vec = _pick_vec(c, x.dtype)
    if x.data_ptr() % (vec * x.element_size()) != 0:
        # vector loads need aligned rows; views with a storage offset may not be
        x = x.clone()
    w9 = weight.reshape(c, 9).t().contiguous()
    y = torch.empty_like(x)
    if x.numel() == 0:
        return y

    fn = _kernels.get(x.device, "dwconv3x3_nhwc", _DWCONV_SOURCE, {"SCALAR": _SCALAR_NAMES[x.dtype], "VEC": vec})
    total = b * h * w * (c // vec)
    block = 256
    _kernels.launch(fn, (total + block - 1) // block, block, x.device, [
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w9.data_ptr()),
        ctypes.c_void_p(bias.contiguous().data_ptr() if bias is not None else 0),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(b), ctypes.c_int(h), ctypes.c_int(w), ctypes.c_int(c),
        ctypes.c_int(1 if bias is not None else 0), ctypes.c_int(1 if gelu else 0),
    ])
    return y


_LAYERNORM_MAX_PER_LANE = 8


def layernorm_supported(x: torch.Tensor, weight: torch.Tensor | None) -> bool:
    """Whether layernorm_lastdim can normalize x over its last dimension."""
    if not kernels_enabled(x.device) or x.dtype not in _SCALAR_NAMES or weight is None or weight.dtype != x.dtype:
        return False
    warp = torch.cuda.get_device_properties(x.device).warp_size
    return x.shape[-1] <= warp * _LAYERNORM_MAX_PER_LANE


def layernorm_lastdim(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    """F.layer_norm over the last dimension: one wavefront per row, statistics in fp32."""
    c = x.shape[-1]
    x = x.contiguous()
    y = torch.empty_like(x)
    rows = x.numel() // c
    if rows == 0:
        return y
    warp = torch.cuda.get_device_properties(x.device).warp_size
    fn = _kernels.get(x.device, "layernorm_rows", _LAYERNORM_SOURCE, {
        "SCALAR": _SCALAR_NAMES[x.dtype], "WARP": warp, "MAXE": _LAYERNORM_MAX_PER_LANE,
    })
    block = 256
    rows_per_block = block // warp
    _kernels.launch(fn, (rows + rows_per_block - 1) // rows_per_block, block, x.device, [
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.contiguous().data_ptr()),
        ctypes.c_void_p(bias.contiguous().data_ptr() if bias is not None else 0),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_longlong(rows), ctypes.c_int(c), ctypes.c_int(1 if bias is not None else 0), ctypes.c_float(eps),
    ])
    return y


_WINDOW_ATTENTION_MAX_TOKENS = 64
_WINDOW_ATTENTION_MAX_HEAD_DIM = 64


def window_attention_supported(qkv: torch.Tensor, bias: torch.Tensor, h: int, w: int, window_size: int, num_heads: int) -> bool:
    """Whether window_attention_tokens can run this (B, H*W, 3*C) qkv tensor without padding."""
    if not kernels_enabled(qkv.device) or qkv.dtype not in _SCALAR_NAMES or bias.dtype != qkv.dtype:
        return False
    c3 = qkv.shape[-1]
    return (
        c3 % (3 * num_heads) == 0
        and c3 // (3 * num_heads) <= _WINDOW_ATTENTION_MAX_HEAD_DIM
        and window_size * window_size <= _WINDOW_ATTENTION_MAX_TOKENS
        and h % window_size == 0
        and w % window_size == 0
        and qkv.shape[-2] == h * w
        and tuple(bias.shape) == (num_heads, window_size * window_size, window_size * window_size)
    )


def window_attention_tokens(
    qkv: torch.Tensor, bias: torch.Tensor, h: int, w: int, window_size: int, num_heads: int, scale: float
) -> torch.Tensor:
    """
    Non-overlapping window self-attention with an additive (heads, N, N) bias, on token-layout data:
    qkv is the (B, H*W, 3*C) output of the qkv projection; returns the (B, H*W, C) attention output
    (before the output projection), with heads concatenated along C like the reference implementation.
    """
    b = qkv.shape[0]
    c = qkv.shape[-1] // 3
    qkv = qkv.contiguous()
    bias = bias.contiguous()
    out = torch.empty(b, h * w, c, device=qkv.device, dtype=qkv.dtype)
    if out.numel() == 0:
        return out
    fn = _kernels.get(qkv.device, "window_attention", _WINDOW_ATTENTION_SOURCE, {
        "SCALAR": _SCALAR_NAMES[qkv.dtype], "WS": window_size, "WIN": window_size * window_size,
        "NH": num_heads, "D": c // num_heads,
    })
    windows = b * (h // window_size) * (w // window_size)
    _kernels.launch(fn, windows * num_heads, window_size * window_size, qkv.device, [
        ctypes.c_void_p(qkv.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(h), ctypes.c_int(w), ctypes.c_float(scale),
    ])
    return out


def _window_attention_reference(qkv: torch.Tensor, bias: torch.Tensor, h: int, w: int, ws: int, nh: int, scale: float) -> torch.Tensor:
    b, _, c3 = qkv.shape
    c = c3 // 3
    x = qkv.view(b, h // ws, ws, w // ws, ws, c3).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, 3, nh, c // nh)
    q, k, v = x.permute(2, 0, 3, 1, 4).unbind(0)
    attn = torch.softmax((q * scale) @ k.transpose(-2, -1) + bias, dim=-1)
    o = (attn @ v).transpose(1, 2).reshape(b, h // ws, w // ws, ws, ws, c)
    return o.permute(0, 1, 3, 2, 4, 5).reshape(b, h * w, c)


def warmup_or_disable(device: torch.device) -> bool:
    """Compiles the kernels once and checks them against PyTorch; disables them on any failure."""
    try:
        for dtype in (torch.float16, torch.float32):
            x = torch.randn(1, 5, 7, 16, device=device, dtype=dtype)
            weight = torch.randn(16, 1, 3, 3, device=device, dtype=dtype)
            bias = torch.randn(16, device=device, dtype=dtype)
            # exact reference on the CPU: MIOpen's own (Winograd) convolutions are only ~1e-3 accurate
            ref = torch.nn.functional.gelu(
                torch.nn.functional.conv2d(x.permute(0, 3, 1, 2).cpu().double(), weight.cpu().double(), bias.cpu().double(), padding=1, groups=16)
            ).permute(0, 2, 3, 1)
            out = dwconv3x3_nhwc(x, weight, bias, gelu=True).cpu().double()
            err = (out - ref).abs().max().item()
            if err > (2e-2 if dtype == torch.float16 else 1e-3):
                raise RuntimeError(f"dwconv self-test mismatch ({dtype}, max err {err})")

            t = torch.randn(37, 120, device=device, dtype=dtype) * 3 + 1
            ln_w = torch.randn(120, device=device, dtype=dtype)
            ln_b = torch.randn(120, device=device, dtype=dtype)
            ref = torch.nn.functional.layer_norm(t.cpu().double(), (120,), ln_w.cpu().double(), ln_b.cpu().double(), 1e-5)
            err = (layernorm_lastdim(t, ln_w, ln_b, 1e-5).cpu().double() - ref).abs().max().item()
            if err > (3e-2 if dtype == torch.float16 else 1e-3):
                raise RuntimeError(f"layernorm self-test mismatch ({dtype}, max err {err})")

            qkv = torch.randn(2, 16 * 8, 3 * 12, device=device, dtype=dtype)
            attn_bias = torch.randn(2, 64, 64, device=device, dtype=dtype) * 0.1
            ref = _window_attention_reference(qkv.cpu().double(), attn_bias.cpu().double(), 16, 8, 8, 2, 6**-0.5)
            out = window_attention_tokens(qkv, attn_bias, 16, 8, 8, 2, 6**-0.5).cpu().double()
            err = (out - ref).abs().max().item()
            if err > (2e-2 if dtype == torch.float16 else 1e-3):
                raise RuntimeError(f"window attention self-test mismatch ({dtype}, max err {err})")
        return True
    except Exception as e:  # never let an optimization break upscaling
        _kernels.disabled_reason = str(e)
        logger.warning(f"Custom HIP kernels disabled: {e}")
        return False
