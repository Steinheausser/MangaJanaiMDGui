"""
Enumerates the DXGI adapters that ONNX Runtime's DirectML execution provider can run on.

ONNX Runtime's DmlExecutionProvider interprets its `device_id` option as the index passed to
IDXGIFactory1::EnumAdapters1, so the index reported here is the raw DXGI adapter index.

DXGI also lists adapters that cannot (or should not) be used for compute:
- software adapters (Microsoft Basic Render Driver / WARP)
- indirect display adapters (e.g. Parsec / virtual display drivers). These are reported with the
  description of the GPU that renders for them, so they would show up as a duplicate of a real GPU.
  They are detected with D3DKMTQueryAdapterInfo(KMTQAITYPE_ADAPTERTYPE) and skipped.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

from sanic.log import logger

DXGI_ADAPTER_FLAG_SOFTWARE = 2
DXGI_ERROR_NOT_FOUND = 0x887A0002

# vtable slots
_RELEASE = 2
_IDXGIFACTORY1_ENUMADAPTERS1 = 12
_IDXGIADAPTER1_GETDESC1 = 10

# D3DKMT_ADAPTERTYPE bits
_KMTQAITYPE_ADAPTERTYPE = 15
_ADAPTERTYPE_RENDER_SUPPORTED = 1 << 0
_ADAPTERTYPE_SOFTWARE_DEVICE = 1 << 2
_ADAPTERTYPE_INDIRECT_DISPLAY_DEVICE = 1 << 6

MICROSOFT_VENDOR_ID = 0x1414


@dataclass(frozen=True)
class DxgiAdapter:
    index: int  # raw DXGI adapter index (== ONNX Runtime DirectML device_id)
    name: str
    vendor_id: int
    device_id: int
    dedicated_video_memory: int


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):  # noqa: N801
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", wintypes.UINT),
    ]


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _D3DKMT_OPENADAPTERFROMLUID(ctypes.Structure):  # noqa: N801
    _fields_ = [("AdapterLuid", _LUID), ("hAdapter", wintypes.UINT)]


class _D3DKMT_QUERYADAPTERINFO(ctypes.Structure):  # noqa: N801
    _fields_ = [
        ("hAdapter", wintypes.UINT),
        ("Type", ctypes.c_int),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", wintypes.UINT),
    ]


class _D3DKMT_CLOSEADAPTER(ctypes.Structure):  # noqa: N801
    _fields_ = [("hAdapter", wintypes.UINT)]


# {770aae78-f26f-4dba-a829-253c83d1b387}
_IID_IDXGIFactory1 = _GUID(  # noqa: N816
    0x770AAE78,
    0xF26F,
    0x4DBA,
    (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)


def _com_method(obj: ctypes.c_void_p, slot: int, restype, *argtypes):  # noqa: ANN001
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[slot])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _com_method(obj, _RELEASE, ctypes.c_ulong)(obj)


def _kmt_adapter_type(luid: _LUID) -> int | None:
    """Returns the D3DKMT_ADAPTERTYPE bits of the adapter, or None if they cannot be queried."""
    try:
        gdi32 = ctypes.WinDLL("gdi32")
        open_args = _D3DKMT_OPENADAPTERFROMLUID(luid, 0)
        if gdi32.D3DKMTOpenAdapterFromLuid(ctypes.byref(open_args)) != 0:
            return None
        try:
            adapter_type = wintypes.UINT(0)
            query = _D3DKMT_QUERYADAPTERINFO(
                open_args.hAdapter,
                _KMTQAITYPE_ADAPTERTYPE,
                ctypes.cast(ctypes.byref(adapter_type), ctypes.c_void_p),
                ctypes.sizeof(adapter_type),
            )
            if gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(query)) != 0:
                return None
            return adapter_type.value
        finally:
            gdi32.D3DKMTCloseAdapter(ctypes.byref(_D3DKMT_CLOSEADAPTER(open_args.hAdapter)))
    except Exception as e:
        logger.debug(f"D3DKMT adapter type query failed: {e}")
        return None


def enumerate_dml_adapters() -> list[DxgiAdapter]:
    """List hardware DXGI adapters that can run DirectML compute, with their raw DXGI index."""
    if sys.platform != "win32":
        return []

    adapters: list[DxgiAdapter] = []
    dxgi = ctypes.WinDLL("dxgi")
    factory = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory))
    if hr != 0:
        logger.debug(f"CreateDXGIFactory1 failed: {hr & 0xFFFFFFFF:#x}")
        return adapters

    try:
        enum_adapters1 = _com_method(
            factory,
            _IDXGIFACTORY1_ENUMADAPTERS1,
            ctypes.c_long,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
        )
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = enum_adapters1(factory, index, ctypes.byref(adapter))
            if hr != 0:
                if (hr & 0xFFFFFFFF) != DXGI_ERROR_NOT_FOUND:
                    logger.debug(f"EnumAdapters1({index}) failed: {hr & 0xFFFFFFFF:#x}")
                break
            try:
                desc = _DXGI_ADAPTER_DESC1()
                get_desc1 = _com_method(
                    adapter,
                    _IDXGIADAPTER1_GETDESC1,
                    ctypes.c_long,
                    ctypes.POINTER(_DXGI_ADAPTER_DESC1),
                )
                if get_desc1(adapter, ctypes.byref(desc)) != 0:
                    continue
                name = desc.Description.strip()

                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE or (
                    desc.VendorId == MICROSOFT_VENDOR_ID and "Basic Render" in name
                ):
                    logger.debug(f"Skipping software DXGI adapter {index}: {name}")
                    continue

                kmt_type = _kmt_adapter_type(desc.AdapterLuid)
                if kmt_type is not None and (
                    not kmt_type & _ADAPTERTYPE_RENDER_SUPPORTED
                    or kmt_type & _ADAPTERTYPE_SOFTWARE_DEVICE
                    or kmt_type & _ADAPTERTYPE_INDIRECT_DISPLAY_DEVICE
                ):
                    logger.debug(
                        f"Skipping non-render/indirect DXGI adapter {index}: {name} (type {kmt_type:#x})"
                    )
                    continue

                adapters.append(
                    DxgiAdapter(
                        index=index,
                        name=name,
                        vendor_id=desc.VendorId,
                        device_id=desc.DeviceId,
                        dedicated_video_memory=desc.DedicatedVideoMemory,
                    )
                )
            finally:
                _release(adapter)
                index += 1
    finally:
        _release(factory)

    return adapters


if __name__ == "__main__":
    for a in enumerate_dml_adapters():
        print(a)
