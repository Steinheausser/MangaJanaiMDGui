import json
import os
import sys

sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))))
from accelerator_detection import AcceleratorDevice, AcceleratorType, get_accelerator_detector

# Suffixes that let the GUI tell apart the same GPU exposed through different backends,
# e.g. "AMD Radeon RX 9070 XT (ROCm)" vs "AMD Radeon RX 9070 XT (DirectML)"
TYPE_SUFFIXES = {
    AcceleratorType.CUDA: "CUDA",
    AcceleratorType.ROCM: "ROCm",
    AcceleratorType.MPS: "MPS",
    AcceleratorType.XPU: "XPU",
    AcceleratorType.DIRECTML: "DirectML",
}


def display_name(device: AcceleratorDevice) -> str:
    suffix = TYPE_SUFFIXES.get(device.type)
    if suffix is None or device.name.endswith(f"({suffix})"):
        return device.name
    return f"{device.name} ({suffix})"


# Get all available accelerator devices
detector = get_accelerator_detector()
all_devices = detector.available_devices
best_device = detector.get_best_device()

device_list = []
for device in all_devices:
    device_info = {
        "type": device.type.value,
        "index": device.index,
        "name": display_name(device),
        "device_string": device.device_string,
        "supports_fp16": device.supports_fp16,
        "supports_bf16": device.supports_bf16,
    }
    if device.memory_total:
        device_info["memory_total"] = device.memory_total
    if device.memory_free:
        device_info["memory_free"] = device.memory_free

    device_list.append(device_info)

print(json.dumps({"all_devices": device_list, "best_device": all_devices.index(best_device)}))