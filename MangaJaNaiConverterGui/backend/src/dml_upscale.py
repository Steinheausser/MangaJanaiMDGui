"""
DirectML (ONNX Runtime) inference for spandrel image models.

Used for GPUs that have no PyTorch backend (e.g. AMD GPUs without ROCm support, Intel/other DX12 GPUs).
The PyTorch model is exported to ONNX once (cached on disk), optionally converted to FP16, and run with
ONNX Runtime's DmlExecutionProvider.

Performance notes:
- DirectML is much faster with fully static shapes (the graph is compiled/fused once per shape), so every
  inference uses the same exact tile size: the image is split with `exact_split`, which pads small images
  and overlaps tiles so each tile has exactly the session's shape. Sessions are created per tile shape with
  free-dimension overrides on the (dynamic-axes) ONNX model.
- Tile sizes are multiples of 64. Some architectures (e.g. FDAT window attention + pixel unshuffle) contain
  shape-dependent padding branches that the legacy ONNX tracer bakes in for the traced shape; they are only
  correct for sizes divisible by window size * unshuffle factor, which multiples of 64 satisfy.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import os
import threading
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
import torch
from nodes.impl.upscale.auto_split_tiles import TileSize
from nodes.impl.upscale.basic_upscale import UpscaleInfo, basic_upscale
from nodes.impl.upscale.exact_split import exact_split
from nodes.utils.utils import get_h_w_c
from sanic.log import logger
from spandrel import ImageModelDescriptor, ModelTiling

from api import Progress

ONNX_OPSET = 17
# fixed DirectML tile size used for "Auto (Estimate)"; picked by benchmarking an RX 9070 XT
DEFAULT_TILE_SIZE = 512
TILE_MULTIPLE = 64
MIN_TILE_SIZE = 64
# number of DirectML sessions (model x device x tile shape) kept alive at once
MAX_CACHED_SESSIONS = 4

_INPUT_NAME = "input"
_OUTPUT_NAME = "output"

_lock = threading.RLock()
_weights_hash_cache: WeakKeyDictionary[torch.nn.Module, str] = WeakKeyDictionary()
_sessions: OrderedDict[tuple[str, int, int, int], object] = OrderedDict()
_logged_fp16_unavailable = False


def is_directml_available() -> bool:
    try:
        import onnxruntime as ort  # noqa: PLC0415

        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _weights_hash(model: ImageModelDescriptor) -> str:
    """Content hash of the model weights (independent of the dtype the model was loaded in)."""
    module = model.model
    cached = _weights_hash_cache.get(module)
    if cached is not None:
        return cached
    h = hashlib.sha1()
    h.update(f"{model.architecture.id}|{model.scale}|{model.input_channels}|{model.output_channels}".encode())
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        t = tensor.detach().cpu()
        if t.is_floating_point():
            t = t.float()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.contiguous().numpy().tobytes())
    digest = h.hexdigest()[:16]
    _weights_hash_cache[module] = digest
    return digest


def _fp16_converter():
    try:
        from onnxruntime.transformers.float16 import (  # noqa: PLC0415
            convert_float_to_float16,
        )

        return convert_float_to_float16
    except Exception:
        pass
    try:
        from onnxconverter_common.float16 import (  # noqa: PLC0415
            convert_float_to_float16,
        )

        return convert_float_to_float16
    except Exception:
        return None


def _export_onnx(model: ImageModelDescriptor, path: Path, fp16: bool) -> None:
    import onnx  # noqa: PLC0415

    start = time.time()
    module = copy.deepcopy(model.model).to("cpu", torch.float32).eval()
    # trace with a size that is a multiple of TILE_MULTIPLE so that no shape-dependent padding branch is taken
    dummy = torch.rand(1, model.input_channels, 128, 128, dtype=torch.float32)
    tmp_fp32 = path.with_name(path.stem + f".{os.getpid()}.fp32.tmp")
    tmp_out = path.with_name(path.stem + f".{os.getpid()}.tmp")
    try:
        with torch.no_grad(), warnings.catch_warnings():
            # the legacy exporter emits many TracerWarnings/DeprecationWarnings that are expected here
            warnings.simplefilter("ignore")
            torch.onnx.export(
                module,
                (dummy,),
                str(tmp_fp32),
                input_names=[_INPUT_NAME],
                output_names=[_OUTPUT_NAME],
                opset_version=ONNX_OPSET,
                do_constant_folding=True,
                dynamic_axes={
                    _INPUT_NAME: {0: "batch", 2: "height", 3: "width"},
                    _OUTPUT_NAME: {0: "batch", 2: "out_height", 3: "out_width"},
                },
                dynamo=False,
            )
        del module
        if fp16:
            converter = _fp16_converter()
            assert converter is not None
            onnx_model = converter(onnx.load(str(tmp_fp32)), keep_io_types=True)
            onnx.save(onnx_model, str(tmp_out))
            del onnx_model
            tmp_fp32.unlink(missing_ok=True)
        else:
            os.replace(tmp_fp32, tmp_out)
        os.replace(tmp_out, path)
    finally:
        tmp_fp32.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
        gc.collect()
    logger.info(f"Exported ONNX model {path.name} in {time.time() - start:.1f}s")


def get_onnx_model_path(
    model: ImageModelDescriptor, cache_dir: Path, use_fp16: bool
) -> tuple[Path, bool]:
    """Returns the cached ONNX model path (exporting it if needed) and whether it is FP16."""
    global _logged_fp16_unavailable
    fp16 = use_fp16 and model.supports_half
    if fp16 and _fp16_converter() is None:
        if not _logged_fp16_unavailable:
            logger.warning("No ONNX FP16 converter available, running DirectML in FP32.")
            _logged_fp16_unavailable = True
        fp16 = False

    name = (
        f"{model.architecture.id}_{model.scale}x_{_weights_hash(model)}"
        f"_{'fp16' if fp16 else 'fp32'}.onnx"
    )
    path = cache_dir / name
    with _lock:
        if not path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Converting model to ONNX for DirectML (one-time): {name}", flush=True)
            _export_onnx(model, path, fp16)
    return path, fp16


def _get_session(onnx_path: Path, device_id: int, tile_w: int, tile_h: int):
    import onnxruntime as ort  # noqa: PLC0415

    key = (str(onnx_path), device_id, tile_w, tile_h)
    with _lock:
        session = _sessions.get(key)
        if session is not None:
            _sessions.move_to_end(key)
            return session

        while len(_sessions) >= MAX_CACHED_SESSIONS:
            _sessions.popitem(last=False)
            gc.collect()

        start = time.time()
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # static shapes let DirectML compile and fuse the whole graph once
        options.add_free_dimension_override_by_name("batch", 1)
        options.add_free_dimension_override_by_name("height", tile_h)
        options.add_free_dimension_override_by_name("width", tile_w)
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=[
                ("DmlExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ],
        )
        if "DmlExecutionProvider" not in session.get_providers():
            logger.warning("DirectML execution provider could not be initialized, running on CPU.")
        logger.info(
            f"Created DirectML session for {onnx_path.name} on device {device_id}"
            f" ({tile_w}x{tile_h}) in {time.time() - start:.2f}s"
        )
        _sessions[key] = session
        return session


def clear_sessions() -> None:
    with _lock:
        _sessions.clear()
    gc.collect()


def _is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "out of memory" in msg
        or "allocate" in msg
        or "e_outofmemory" in msg
        or "8007000e" in msg
        or "887a0005" in msg  # DXGI_ERROR_DEVICE_REMOVED (e.g. TDR on huge tiles)
        or "887a0006" in msg  # DXGI_ERROR_DEVICE_HUNG
    )


def _round_tile_size(size: int) -> int:
    return max(MIN_TILE_SIZE, (size // TILE_MULTIPLE) * TILE_MULTIPLE)


def _ceil_to_multiple(size: int) -> int:
    return max(MIN_TILE_SIZE, -(-size // TILE_MULTIPLE) * TILE_MULTIPLE)


def _starting_tile_size(img: np.ndarray, tile_size: TileSize) -> int:
    h, w, _ = get_h_w_c(img)
    if tile_size == 0:  # ESTIMATE
        return DEFAULT_TILE_SIZE
    if tile_size < 0:  # NO_TILING / MAX_TILE_SIZE: the whole image in one tile
        return _ceil_to_multiple(max(w, h))
    return _round_tile_size(tile_size)


def _dml_auto_split(
    img: np.ndarray,
    model: ImageModelDescriptor,
    onnx_path: Path,
    device_id: int,
    tile_size: int,
    progress: Progress,
) -> np.ndarray:
    h, w, _ = get_h_w_c(img)

    while True:
        # tiles never need to be larger than the (rounded up) image
        tile_w = min(tile_size, _ceil_to_multiple(w))
        tile_h = min(tile_size, _ceil_to_multiple(h))
        session = _get_session(onnx_path, device_id, tile_w, tile_h)

        def upscale(tile: np.ndarray, _: object) -> np.ndarray:
            progress.check_aborted()
            if progress.paused:
                progress.suspend()

            _, _, c = get_h_w_c(tile)
            x = tile.astype(np.float32, copy=False)
            if x.ndim == 2:
                x = x[:, :, np.newaxis]
            # expand grayscale to match model input channels (same as pytorch_auto_split)
            if c == 1 and model.input_channels > 1:
                x = np.repeat(x, model.input_channels, axis=2)
            # (H, W, C) -> (1, C, H, W); channel order is passed through unchanged like pytorch_auto_split
            x = np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis, :])

            output: np.ndarray = session.run([_OUTPUT_NAME], {_INPUT_NAME: x})[0]  # type: ignore

            out = output[0].transpose(1, 2, 0)
            if c == 1:
                out = out[:, :, :1]
            return np.ascontiguousarray(out, dtype=np.float32)

        try:
            return exact_split(img, exact_size=(tile_w, tile_h), upscale=upscale)
        except Exception as e:
            if not _is_oom_error(e) or tile_size <= MIN_TILE_SIZE:
                raise
            new_tile_size = _round_tile_size(tile_size // 2)
            logger.warning(
                f"DirectML out of memory with tile size {tile_size}, retrying with {new_tile_size}: {e}"
            )
            print(
                f"DirectML out of memory with tile size {tile_size}, retrying with {new_tile_size}",
                flush=True,
            )
            tile_size = new_tile_size
            del session
            clear_sessions()


def dml_upscale_image(
    img: np.ndarray,
    model: ImageModelDescriptor,
    device_id: int,
    use_fp16: bool,
    tile_size: TileSize,
    cache_dir: Path,
    progress: Progress,
) -> np.ndarray:
    """
    Upscales an image (HWC float32 in [0, 1], same channel conventions as the PyTorch path) with the given
    model using ONNX Runtime DirectML on the DXGI adapter `device_id`.

    Mirrors `upscale_image_node` (no custom scale, no separate alpha) so the output matches the PyTorch path.
    """
    onnx_path, _ = get_onnx_model_path(model, cache_dir, use_fp16)

    if model.tiling == ModelTiling.INTERNAL:
        tile_size = TileSize(-1)
    starting_tile_size = _starting_tile_size(img, tile_size)

    info = UpscaleInfo(
        in_nc=model.input_channels, out_nc=model.output_channels, scale=model.scale
    )
    return basic_upscale(
        img,
        lambda i: _dml_auto_split(
            i, model, onnx_path, device_id, starting_tile_size, progress
        ),
        upscale_info=info,
        scale=model.scale,
        separate_alpha=False,
        clip=False,
    )
