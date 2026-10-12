from __future__ import annotations

import gc
import weakref

import numpy as np
import torch
import torch.nn.functional as F
from accelerator_detection import get_autocast_device_type, is_device_type_supported_for_autocast
from nodes.utils.utils import get_h_w_c
from spandrel import ImageModelDescriptor

from api import Progress
from sanic.log import logger

from ..upscale.auto_split import Split, Tiler, auto_split
from .utils import safe_accelerator_cache_empty


def _into_standard_image_form(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 2:
        # (H, W)
        return t
    elif len(t.shape) == 3:
        # (C, H, W) -> (H, W, C)
        return t.permute(1, 2, 0)
    elif len(t.shape) == 4:
        # (1, C, H, W) -> (H, W, C)
        return t.squeeze(0).permute(1, 2, 0)
    else:
        raise ValueError("Unsupported output tensor shape")


def _into_batched_form(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 2:
        # (H, W) -> (1, 1, H, W)
        return t.unsqueeze(0).unsqueeze(0)
    elif len(t.shape) == 3:
        # (H, W, C) -> (1, C, H, W)
        return t.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError("Unsupported input tensor shape")


def _rgb_to_bgr(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 3 and t.shape[2] == 3:
        # (H, W, C) RGB -> BGR
        return t.flip(2)
    elif len(t.shape) == 3 and t.shape[2] == 4:
        # (H, W, C) RGBA -> BGRA
        return torch.cat((t[:, :, 2:3], t[:, :, 1:2], t[:, :, 0:1], t[:, :, 3:4]), 2)
    else:
        return t


# On ROCm, MIOpen benchmarks convolution kernels for every new input shape (a few seconds per shape,
# cached on disk afterwards). For architectures that already pad their input internally, padding up to
# a multiple of this size lets pages/tiles of slightly different sizes share one shape with negligible
# effect on the output. Plain conv nets (ESRGAN etc.) are not padded: their border pixels would change.
ROCM_SHAPE_BUCKET = 128
ROCM_BUCKETED_ARCHITECTURES = {"FDAT"}
# auto tile size cap, in output pixels per tile (see rocm_max_tile_size)
ROCM_MAX_TILE_OUTPUT_PIXELS = 20_000_000

# models that hit a convolution MIOpen has no enabled solver for; they run with native convolutions
_NATIVE_CONV_MODELS: weakref.WeakSet[ImageModelDescriptor] = weakref.WeakSet()


def is_rocm_device(device: torch.device) -> bool:
    return device.type == "cuda" and getattr(torch.version, "hip", None) is not None


ROCM_PAD_MODE = "reflect"


def _pad_to_bucket(t: torch.Tensor, bucket: int) -> torch.Tensor:
    h, w = t.shape[-2:]
    pad_h = -h % bucket
    pad_w = -w % bucket
    if pad_h == 0 and pad_w == 0:
        return t
    if ROCM_PAD_MODE == "reflect" and (pad_h >= h or pad_w >= w):
        return F.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
    if ROCM_PAD_MODE == "zeros":
        return F.pad(t, (0, pad_w, 0, pad_h))
    return F.pad(t, (0, pad_w, 0, pad_h), mode=ROCM_PAD_MODE)


def _run_model(model: ImageModelDescriptor, t: torch.Tensor) -> torch.Tensor:
    if model in _NATIVE_CONV_MODELS:
        with torch.backends.cudnn.flags(enabled=False):
            return model(t)
    try:
        return model(t)
    except RuntimeError as e:
        message = str(e).lower()
        if "miopen" not in message or "alloc" in message or "out of memory" in message:
            raise
        logger.warning(f"MIOpen could not run this model ({e}); falling back to native convolutions.")
        _NATIVE_CONV_MODELS.add(model)
        with torch.backends.cudnn.flags(enabled=False):
            return model(t)


def prepare_model(model: ImageModelDescriptor, device: torch.device, use_fp16: bool) -> torch.dtype:
    """Moves the model to the device in its inference dtype (channels-last) and returns that dtype."""
    dtype = torch.float32
    if use_fp16:
        if model.supports_half:
            dtype = torch.float16
        elif torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
    if model.dtype != dtype or model.device != device:
        # (spandrel's descriptor .to() takes no memory_format; apply it to the module itself)
        model.to(device, dtype)
        model.model.to(memory_format=torch.channels_last)
    if is_rocm_device(device):
        _cap_allocator(device)
    return dtype


# device index -> allocator cap in bytes
_CAPPED_DEVICES: dict[int, int] = {}


def _cap_allocator(device: torch.device) -> None:
    """
    Windows pages GPU memory out to system RAM instead of failing when VRAM is oversubscribed, which
    makes inference many times slower without any error. Capping PyTorch's allocator at the VRAM that
    is free when upscaling starts makes it release cached blocks, and raise an out-of-memory error
    (handled by splitting into smaller tiles) rather than grow past physical memory.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _CAPPED_DEVICES:
        return
    try:
        free, total = torch.cuda.mem_get_info(device)
        usable = free + torch.cuda.memory_reserved(device) - 512 * 1024**2
        fraction = max(0.1, min(0.95, usable / total))
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        _CAPPED_DEVICES[index] = int(fraction * total)
        logger.debug(f"Capped PyTorch GPU memory at {fraction * total / 1024**3:.2f} GiB")
    except Exception as e:
        _CAPPED_DEVICES[index] = 0
        logger.warning(f"Could not cap GPU memory: {e}")


_BYTES_PER_PIXEL: weakref.WeakKeyDictionary[ImageModelDescriptor, dict[torch.dtype, float]] = weakref.WeakKeyDictionary()


@torch.inference_mode()
def measure_bytes_per_pixel(model: ImageModelDescriptor, device: torch.device, dtype: torch.dtype) -> float:
    """Peak GPU memory per input pixel of one forward pass, measured once per model on a probe tile."""
    per_dtype = _BYTES_PER_PIXEL.setdefault(model, {})
    if dtype in per_dtype:
        return per_dtype[dtype]

    probe = 256
    x = torch.rand(1, model.input_channels, probe, probe, device=device, dtype=dtype)
    x = x.to(memory_format=torch.channels_last)
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    # same autocast setup as pytorch_auto_split, so the measurement matches real runs
    with torch.autocast(
        device_type=get_autocast_device_type(device),
        dtype=dtype,
        enabled=is_device_type_supported_for_autocast(device) and dtype != torch.float32,
    ):
        _run_model(model, x)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - base
    per_dtype[dtype] = max(peak, 1) / (probe * probe)
    logger.debug(f"Measured {per_dtype[dtype] / 1024:.1f} KiB of GPU memory per input pixel")
    return per_dtype[dtype]


def rocm_max_tile_size(model: ImageModelDescriptor, device: torch.device, use_fp16: bool, budget_limit_gib: int) -> int:
    """Largest square tile (a multiple of ROCM_SHAPE_BUCKET) whose forward pass fits in free VRAM."""
    dtype = prepare_model(model, device, use_fp16)
    bytes_per_pixel = measure_bytes_per_pixel(model, device, dtype)
    free, _total = torch.cuda.mem_get_info(device)
    # memory cached by our allocator is reusable too, up to the allocator cap
    usable = free + torch.cuda.memory_reserved(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    cap = _CAPPED_DEVICES.get(index, 0)
    if cap > 0:
        usable = min(usable, cap)
    available = max(usable - torch.cuda.memory_allocated(device), 0)
    budget = available * 0.7
    if budget_limit_gib > 0:
        budget = min(budget, budget_limit_gib * 1024**3)
    # tiles get overlap and bucket padding added on top of this size
    side = int((budget / bytes_per_pixel) ** 0.5) - ROCM_SHAPE_BUCKET
    # Tiles whose output exceeds ~20 MP (e.g. above 1024px at 4x) run slower per pixel on RDNA
    # even when they fit in memory: their full-resolution activations are several GB each.
    side = min(side, int(ROCM_MAX_TILE_OUTPUT_PIXELS**0.5 / max(model.scale, 1)))
    side = side // ROCM_SHAPE_BUCKET * ROCM_SHAPE_BUCKET
    logger.debug(f"ROCm tile size estimate: {side}px ({available / 1024**3:.2f} GiB available)")
    return max(side, ROCM_SHAPE_BUCKET)


def _into_tensor(
    img: np.ndarray, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    img = np.ascontiguousarray(img)
    writeable = img.flags.writeable
    try:
        if not writeable and device == torch.device("cpu"):
            img = np.copy(img)
        else:
            # since we are going to copy the image to the GPU, we can skip the copy here
            try:
                img.flags.writeable = True
            except Exception:
                # Some arrays cannot be made writeable, and we need to copy them
                img = np.copy(img)
        if device == torch.device("cpu"):
            input_tensor = (
                torch.from_numpy(img).to(device, dtype, non_blocking=True)
            )
        else:
            input_tensor = (
                torch.from_numpy(img).pin_memory().to(device, dtype, non_blocking=True)
            )
        return input_tensor
    finally:
        img.flags.writeable = writeable


@torch.inference_mode()
def pytorch_auto_split(
    img: np.ndarray,
    model: ImageModelDescriptor[torch.nn.Module],
    device: torch.device,
    use_fp16: bool,
    tiler: Tiler,
    progress: Progress,
) -> np.ndarray:
    dtype = prepare_model(model, device, use_fp16)

    def upscale(img: np.ndarray, _: object):
        progress.check_aborted()
        if progress.paused:
            # clear resources before pausing
            gc.collect()
            safe_accelerator_cache_empty(device)
            progress.suspend()

        input_tensor = None
        try:
            _, _, input_channels = get_h_w_c(img)
            # convert to tensor
            input_tensor = _into_tensor(img, device, dtype)
            # expand grayscale tensor to match model input channels
            if input_channels == 1 and model.input_channels > 1:
                input_tensor = input_tensor.repeat(1, 1, model.input_channels)
            input_tensor = _into_batched_form(input_tensor)
            input_tensor = input_tensor.to(
                memory_format=torch.channels_last
            )  # TODO refactor
            # inference with accelerator-aware autocast
            autocast_device_type = get_autocast_device_type(device)
            autocast_enabled = is_device_type_supported_for_autocast(device) and use_fp16
            
            in_h, in_w = input_tensor.shape[-2:]
            if is_rocm_device(device) and model.architecture.id in ROCM_BUCKETED_ARCHITECTURES:
                input_tensor = _pad_to_bucket(input_tensor, ROCM_SHAPE_BUCKET)

            with torch.autocast(device_type=autocast_device_type, dtype=dtype, enabled=autocast_enabled):
                output_tensor = _run_model(model, input_tensor)

            output_tensor = output_tensor[..., : in_h * model.scale, : in_w * model.scale]

            # convert back to numpy
            output_tensor = _into_standard_image_form(output_tensor)
            if input_channels == 1:
                output_tensor = output_tensor[:, :, 0].unsqueeze(-1)
            # print("out dtype", output_tensor.dtype, flush=True)
            # result = output_tensor.detach().cpu().detach().float().numpy()
            result = output_tensor.detach().cpu().detach()
            if result.dtype == torch.bfloat16:
                result = result.float()
            result = result.numpy()

            return result
        except RuntimeError as e:
            # Check to see if its actually an out of memory error
            message = str(e)
            lowered = message.lower()
            if (
                "allocate" in message
                or "CUDA" in message
                or "out of memory" in lowered
                or "outofmemory" in lowered  # hipErrorOutOfMemory
                or "allocfailed" in lowered  # miopenStatusAllocFailed
            ):
                # Collect garbage (clear memory)
                if input_tensor is not None:
                    try:
                        input_tensor.detach().cpu()
                    except Exception:
                        pass
                    del input_tensor
                gc.collect()
                safe_accelerator_cache_empty(device)
                return Split()
            else:
                # Re-raise the exception if not an OOM error
                raise

    return auto_split(img, upscale, tiler)
