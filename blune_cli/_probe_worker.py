"""Internal: runs a single probe in its own process, invoked as
`python -m blune_cli._probe_worker <repo> --library <lib>`.

`blune sweep` runs hundreds to thousands of these one at a time. Isolating
each probe in its own process means a single model that segfaults or gets
OOM-killed by the OS only takes down that one subprocess -- the sweep loop
sees a nonzero exit code, records it as FAILED, and moves on to the next
model instead of losing the entire run.
"""
import argparse
import json
import sys

from . import config_cache, probe_mlx, probe_vllm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--library", default="mlx")
    args = parser.parse_args()

    config = config_cache.get_config(args.repo, offline=True)

    if args.library == "mlx":
        result = probe_mlx.probe(args.repo, config)
    elif args.library == "vllm":
        result = probe_vllm.probe(args.repo, config)
    else:
        raise ValueError(f"unknown library: {args.library}")

    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
