"""Read-only HAL RPC latency measurement; never acquires control or sends motion."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(os.environ.get('XR1_TUOZHUAI_ROOT', '/opt/deeptouch')) / 'deeptouch_robot_hal/src'))
from robot_hal.ipc.client.rpc.connection import RpcConnection
from robot_hal.protocol.proto import service_pb2 as proto


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=45)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    conn = RpcConnection('127.0.0.1', 7860)
    conn.connect()
    conn.socket.settimeout(2)
    rows = []
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            before = time.monotonic()
            result = conn.call(proto.RequestType.REQUEST_TYPE_PING,
                               ping_request=proto.PingRequest())
            if not result.ping_response.ok:
                raise RuntimeError('HAL ping not acknowledged')
            rows.append({'wall': time.time(), 'elapsed': time.monotonic() - before})
            time.sleep(.05)
    finally:
        conn.close()
    values = sorted(row['elapsed'] for row in rows)
    report = {'n': len(values), 'median_ms': statistics.median(values) * 1000,
              'p99_ms': values[int(.99 * (len(values) - 1))] * 1000,
              'max_ms': max(values) * 1000, 'commands_sent': 0,
              'samples': rows}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}))


if __name__ == '__main__':
    main()
