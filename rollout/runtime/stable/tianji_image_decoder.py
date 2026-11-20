"""Decode HAL observation camera payloads into RGB HWC arrays.

The recorder package is not required by the XR1 robot client. Keeping this
small decoder beside the adapter also prevents the model client from importing
an older recorder/ROS environment when the host HAL has been upgraded.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


# Stable values from the HAL observation wire protocol.
RAW_RGB8 = 1
JPEG = 2


class ImageDecodeError(RuntimeError):
    """A HAL camera payload is malformed or uses an unsupported encoding."""


def _decode_raw(payload) -> np.ndarray:
    shape = (int(payload.height), int(payload.width), int(payload.channels))
    logical_size = int(np.prod(shape))
    if len(payload.data) != logical_size:
        raise ImageDecodeError(
            f"{payload.name}: raw payload size {len(payload.data)} != {logical_size}"
        )
    return np.ascontiguousarray(
        np.frombuffer(payload.data, dtype=np.uint8).reshape(shape)
    )


def _decode_jpeg(payload) -> np.ndarray:
    import cv2

    encoded = np.frombuffer(payload.data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        shape = None if bgr is None else tuple(bgr.shape)
        raise ImageDecodeError(f"{payload.name}: invalid JPEG frame, shape={shape}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def decode_camera_payload(payload) -> np.ndarray:
    encoding = int(payload.encoding)
    if encoding == RAW_RGB8:
        image = _decode_raw(payload)
    elif encoding == JPEG:
        image = _decode_jpeg(payload)
    else:
        raise ImageDecodeError(f"{payload.name}: unknown encoding {encoding}")
    image.setflags(write=False)
    return image


def decode_observation_frames(
    camera_frames: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Decode one aligned observation atomically into RGB HWC arrays."""
    return {name: decode_camera_payload(payload) for name, payload in camera_frames.items()}
