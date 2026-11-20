"""Wait for a local inference listener without issuing model or robot commands."""
import argparse
import socket
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('port', type=int)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            with socket.create_connection(('127.0.0.1', args.port), timeout=2):
                break
        except OSError:
            if time.monotonic() >= deadline:
                raise SystemExit(f'Port {args.port} did not become ready')
            time.sleep(1)
