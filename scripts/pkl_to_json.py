"""
pkl_to_json.py
Converts every .pkl file in a directory to a sibling .json file.

Intended for releasing experiment outputs in a more portable format than
Python pickles. Handles the common types produced by this project:
plain Python primitives, lists, dicts, tuples, numpy arrays/scalars, and
torch tensors. Anything else is converted via repr() with a warning.

Usage:
    python pkl_to_json.py <directory> [--overwrite] [--delete-pkl]

By default, existing .json files are skipped and .pkl files are kept.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path


def to_jsonable(obj):
    """Recursively convert ``obj`` into something ``json.dump`` can handle."""
    # Primitives that json handles natively
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Containers
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]

    # numpy
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist())
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass

    # torch tensors
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return to_jsonable(obj.detach().cpu().tolist())
    except ImportError:
        pass

    # Fallback: stringify and warn
    print(f"  warning: converting {type(obj).__name__} via repr()", file=sys.stderr)
    return repr(obj)


def convert_file(pkl_path: Path, overwrite: bool) -> bool:
    """Convert one .pkl to .json. Returns True if a new file was written."""
    json_path = pkl_path.with_suffix(".json")
    if json_path.exists() and not overwrite:
        print(f"  skip (json exists): {json_path.name}")
        return False

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    converted = to_jsonable(data)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"  wrote: {json_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing .pkl files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .json files")
    parser.add_argument("--delete-pkl", action="store_true", help="Delete each .pkl after a successful conversion")
    args = parser.parse_args()

    if not args.directory.is_dir():
        sys.exit(f"error: not a directory: {args.directory}")

    pkl_files = sorted(args.directory.glob("*.pkl"))
    if not pkl_files:
        print(f"no .pkl files in {args.directory}")
        return

    print(f"converting {len(pkl_files)} file(s) in {args.directory}")
    converted = 0
    for pkl_path in pkl_files:
        print(pkl_path.name)
        if convert_file(pkl_path, overwrite=args.overwrite):
            converted += 1
            if args.delete_pkl:
                pkl_path.unlink()
                print(f"  deleted: {pkl_path.name}")

    print(f"done: {converted}/{len(pkl_files)} converted")


if __name__ == "__main__":
    main()
