"""Length-prefixed, copy-light ndarray protocol for model/robot processes."""
from io import BytesIO
import json
import socket
import struct

import numpy as np

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
WIRE_FORMAT = "raw_v2"
_RAW_MAGIC = b"XRW2"
_PREFIX = struct.Struct(">I")
_RAW_HEADER_LENGTH = struct.Struct(">I")


def _read_exact(sock, size):
    value = bytearray(size)
    view = memoryview(value)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:])
        if not count:
            raise ConnectionError("connection closed during message")
        received += count
    return bytes(value)


def _send_buffers(sock, buffers):
    """Send an iovec without first joining multi-megabyte camera arrays."""
    pending = []
    for value in buffers:
        view = memoryview(value).cast("B")
        if view.nbytes:
            pending.append(view)
    while pending:
        if hasattr(sock, "sendmsg"):
            sent = sock.sendmsg(pending)
        else:  # pragma: no cover - Linux robot runtime always has sendmsg.
            sent = sock.send(pending[0])
        if sent <= 0:
            raise ConnectionError("connection closed during message")
        while pending and sent >= len(pending[0]):
            sent -= len(pending.pop(0))
        if sent:
            pending[0] = pending[0][sent:]


def send_arrays(sock, arrays, metadata=None):
    values = []
    descriptors = []
    for key, value in arrays.items():
        if not isinstance(key, str) or not key or key == "__metadata__":
            raise ValueError(f"invalid array key {key!r}")
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ValueError(f"object array is not allowed: {key}")
        values.append(array)
        descriptors.append({"name": key, "dtype": array.dtype.str,
                            "shape": list(array.shape), "nbytes": array.nbytes})
    header = json.dumps(
        {"format": WIRE_FORMAT, "metadata": metadata or {}, "arrays": descriptors},
        ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")
    payload_size = len(_RAW_MAGIC) + _RAW_HEADER_LENGTH.size + len(header)
    payload_size += sum(array.nbytes for array in values)
    if not 0 < payload_size <= MAX_MESSAGE_BYTES:
        raise ValueError(f"message is too large: {payload_size} bytes")
    framing = (_PREFIX.pack(payload_size), _RAW_MAGIC,
               _RAW_HEADER_LENGTH.pack(len(header)), header)
    _send_buffers(sock, (*framing, *values))
    return payload_size


def _decode_raw(sock, remaining):
    if remaining < _RAW_HEADER_LENGTH.size:
        raise ValueError("raw message header length is missing")
    header_size = _RAW_HEADER_LENGTH.unpack(_read_exact(sock, 4))[0]
    remaining -= 4
    if not 0 < header_size <= remaining:
        raise ValueError(f"invalid raw header length {header_size}")
    header = json.loads(_read_exact(sock, header_size).decode("utf-8"))
    remaining -= header_size
    if header.get("format") != WIRE_FORMAT:
        raise ValueError(f"unsupported wire format {header.get('format')!r}")
    arrays = {}
    for descriptor in header.get("arrays", []):
        name = descriptor.get("name")
        shape = descriptor.get("shape")
        nbytes = descriptor.get("nbytes")
        if (not isinstance(name, str) or not name or name in arrays or
                not isinstance(shape, list) or
                not all(isinstance(dim, int) and dim >= 0 for dim in shape) or
                not isinstance(nbytes, int) or nbytes < 0):
            raise ValueError("invalid raw array descriptor")
        dtype = np.dtype(descriptor.get("dtype"))
        if dtype.hasobject:
            raise ValueError(f"object array is not allowed: {name}")
        expected = int(np.prod(shape, dtype=np.int64))*dtype.itemsize
        if expected != nbytes or nbytes > remaining:
            raise ValueError(f"invalid byte count for {name}: {nbytes} != {expected}")
        # The immutable bytes object remains owned by ndarray.base, so no
        # second full image copy is needed after receiving each camera.
        arrays[name] = np.frombuffer(_read_exact(sock, nbytes), dtype=dtype).reshape(shape)
        remaining -= nbytes
    if remaining:
        raise ValueError(f"raw message has {remaining} trailing bytes")
    metadata = header.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("message metadata must be an object")
    return arrays, metadata


def recv_arrays(sock):
    length = _PREFIX.unpack(_read_exact(sock, 4))[0]
    if not 0 < length <= MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid message length {length}")
    magic = _read_exact(sock, len(_RAW_MAGIC))
    if magic == _RAW_MAGIC:
        return _decode_raw(sock, length-len(_RAW_MAGIC))
    # Read-only compatibility lets the async client produce a clear handshake
    # mismatch when an old/synchronous server still owns the configured port.
    payload = magic + _read_exact(sock, length-len(magic))
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if "__metadata__" not in archive.files:
            raise ValueError("message metadata is missing")
        metadata = json.loads(archive["__metadata__"].astype(np.uint8).tobytes().decode("utf-8"))
        arrays = {key: archive[key].copy() for key in archive.files if key != "__metadata__"}
    return arrays, metadata


def connect(host, port, timeout):
    sock = socket.create_connection((host, port), timeout)
    sock.settimeout(timeout)
    return sock
