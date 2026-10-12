# MangaJaNaiConverterGui on AMD (ROCm) — results

Test machine: AMD Radeon RX 9070 XT (gfx1201, 16 GB), Windows 11, torch 2.9.1+rocm7.13 (TheRock nightly).
Test set: `testdata/images` (4 grayscale + 4 color pages, ~1200x1700) and `testdata/test.cbz`, default GUI workflow.

## End to end (bench_e2e.py, whole job including startup and encoding)

| Run | 2x, 8 pages | 4x, 8 pages | CBZ 2x |
|---|---|---|---|
| Stock app (CUDA build, falls back to CPU) | ~24 min (est., ~3 min/page) | — | — |
| ROCm torch, stock code paths (opt1) | 121 s cold / 82 s warm | — | — |
| Final (opt5), first run / repeat run | 11.3 s / **9.75 s** | 32.4 s / **25.3 s** | **10.85 s** |

"First run" includes MIOpen kernel lookups for new shapes, which are cached on disk afterwards.

## Per model (bench_models.py, one 1200x1700 page, fp16, median)

| Model | torch 2.9.1 | torch 2.10.0 |
|---|---|---|
| 2x MangaJaNai ESRGAN | 0.43 s | 0.69 s |
| 4x MangaJaNai ESRGAN | 2.09 s | 5.21 s |
| 2x IllustrationJaNai FDAT M | 0.47 s | 0.39 s |
| 4x IllustrationJaNai FDAT M | 1.89 s | 17.4 s |

torch 2.10.0 is much slower on the 4x models, so the installer pins torch 2.9.1 / torchvision 0.24.0.
Original FDAT 2x model time on ROCm before the custom kernels: 2.5–3.8 s per page.

## What changed

- **Install**: the GUI detects the GPU (WMI, PCI device id). NVIDIA keeps the prebuilt CUDA backend; RDNA3/RDNA4/Strix Halo
  get a pip install of AMD's ROCm torch wheels; other AMD/Intel GPUs get the CPU torch + ONNX Runtime DirectML.
  `backend/python/Flavor.txt` records the installed flavor (e.g. `rocm-gfx120X-all|torch=2.9.1`).
- **Device handling**: ROCm GPUs no longer listed twice; GUI device index off-by-one fixed; DirectML devices listed
  as a fallback; auto-detection and model chains work as on NVIDIA.
- **MIOpen**: solver set restricted (Winograd / GEMM) with hybrid find mode — new conv shapes take ~2–4 s instead of
  70–330 s, with no loss in steady-state speed. FDAT inputs are padded to multiples of 128 so pages share shapes.
- **FDAT**: custom HIP kernels (depthwise 3x3 + GELU, LayerNorm, fused 8x8 window attention), compiled at runtime
  with HIPRTC, self-tested against a CPU reference on first use and disabled automatically if anything is off
  (`MANGAJANAI_DISABLE_HIP_KERNELS=1` forces them off).
- **Memory**: tile size from measured memory per pixel, allocator capped at free VRAM (Windows otherwise silently pages
  VRAM to system RAM, ~10x slower), 4x tiles capped at 1024 px.
- **Pipeline**: models load on the CPU in the preprocessing process; image encoding runs on several threads; saving
  processes no longer initialise the GPU; save errors fail the job instead of hanging it.

Raw data: benchmark scripts and raw data were kept outside this repository.
