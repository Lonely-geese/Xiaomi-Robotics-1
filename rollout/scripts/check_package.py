"""Verify the pinned RECAP package without connecting to robot hardware."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'control'))
from common import verify_dependencies, verify_package

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--code-only', action='store_true')
    args = parser.parse_args()
    if args.full and args.code_only:
        parser.error('--full requires model weights')
    verify_dependencies()
    identity = verify_package(full=args.full, require_weights=not args.code_only)
    print(f"Verified RECAP epoch{identity['completed_epochs']} step{identity['checkpoint_step']}")
