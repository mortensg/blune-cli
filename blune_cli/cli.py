#!/usr/bin/env python3
"""
blune-cli: find out which model runs best, how fast, in which library,
on the machine you're actually sitting at.

Interactive by default (run with no arguments -- starts broad, narrows down
as you choose). Every choice is also available as a direct flag for
scripting: `blune --search --library mlx --limit 20 --rank`.
"""
import argparse
import sys

from rich.prompt import IntPrompt, Prompt

from . import config_cache, measurements, probe_llamacpp, probe_mlx, probe_vllm, search, ui
from .hardware import detect_machine


def run_probe(repo_id: str, library: str, machine, offline: bool) -> dict:
    """Run one probe, preferring a real recorded measurement over an
    estimate if we have one for this exact repo+library+machine."""
    real = measurements.find_real_measurement(
        repo_id, library, machine=_machine_key(machine)
    )
    if real:
        return {**real, "source": "measured"}

    config = config_cache.get_config(repo_id, offline=offline)

    if library == "mlx":
        result = probe_mlx.probe(repo_id, config)
    elif library == "vllm":
        result = probe_vllm.probe(repo_id, config)
    elif library == "llama.cpp":
        raise NotImplementedError(
            "llama.cpp probing needs a GGUF URL, not a repo ID -- use "
            "`blune gguf <url>` directly for now."
        )
    else:
        raise ValueError(f"unknown library: {library}")

    result["source"] = "estimated"
    return result


def _machine_key(machine) -> str:
    if machine.chip_spec:
        return f"{machine.chip_spec.id}_{int(machine.total_ram_gb)}gb"
    return "unknown"


def cmd_search_and_rank(args, machine):
    ui.info(f"searching HF Hub (filter={args.search_filter}, limit={args.limit})...")
    repos = search.search_models(query_filter=args.search_filter, limit=args.limit)
    ui.info(f"found {len(repos)} candidate(s), probing each (library={args.library})...\n")

    results = []
    for repo in repos:
        try:
            r = run_probe(repo, args.library, machine, args.offline)
            r["repo_id"] = repo
            results.append(r)
        except Exception as e:
            results.append({"repo_id": repo, "error": str(e)})
    ui.show_ranking(results)


def cmd_test_one(args, machine):
    try:
        r = run_probe(args.repo, args.library, machine, args.offline)
        r["repo_id"] = args.repo
        ui.show_single_result(r)
    except Exception as e:
        ui.error(str(e))


def cmd_compare_libraries(args, machine):
    results = []
    for lib in ("mlx", "vllm"):
        try:
            r = run_probe(args.repo, lib, machine, args.offline)
            r["repo_id"] = args.repo
            results.append(r)
        except Exception as e:
            results.append({"repo_id": args.repo, "library": lib, "error": str(e)})
    ui.show_ranking(results)


def cmd_gguf(args, machine):
    if not machine.bandwidth_gbs:
        ui.error(
            "Unknown machine bandwidth -- the GGUF probe needs this to "
            "apply the tok/s = bandwidth / bytes_per_token law. Run "
            "`blune hw` to check what was detected."
        )
        return
    try:
        r = probe_llamacpp.probe(args.url, machine.bandwidth_gbs)
        ui.show_single_result(r)
    except Exception as e:
        ui.error(str(e))


def interactive_wizard(machine):
    ui.show_banner(f"{machine.os_name} -- {machine.chip_name}")
    ui.show_machine_panel(machine)

    console_choice = IntPrompt.ask(
        "\n[bold]What do you want to do?[/]\n"
        "  1. Find the fastest model for my machine (search + rank)\n"
        "  2. Test a specific model\n"
        "  3. Compare one model across libraries\n"
        "  4. Just show my hardware info (done above)\n",
        choices=["1", "2", "3", "4"],
        default="1",
    )

    if console_choice == 4:
        return

    if console_choice == 1:
        lib = Prompt.ask(
            "Which library?", choices=["mlx", "vllm"], default="mlx"
        )
        limit = IntPrompt.ask("How many candidates to check?", default=10)
        args = argparse.Namespace(
            search_filter="mixture-of-experts,mlx",
            limit=limit,
            library=lib,
            offline=False,
        )
        cmd_search_and_rank(args, machine)

    elif console_choice == 2:
        repo = Prompt.ask("Hugging Face repo ID")
        lib = Prompt.ask("Which library?", choices=["mlx", "vllm"], default="mlx")
        args = argparse.Namespace(repo=repo, library=lib, offline=False)
        cmd_test_one(args, machine)

    elif console_choice == 3:
        repo = Prompt.ask("Hugging Face repo ID")
        args = argparse.Namespace(repo=repo, offline=False)
        cmd_compare_libraries(args, machine)


def main():
    parser = argparse.ArgumentParser(prog="blune", description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use only cached configs")
    sub = parser.add_subparsers(dest="command")

    p_hw = sub.add_parser("hw", help="show detected hardware and exit")

    p_search = sub.add_parser("search", help="find + rank candidate models")
    p_search.add_argument("--search-filter", default="mixture-of-experts,mlx")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--library", choices=["mlx", "vllm"], default="mlx")

    p_test = sub.add_parser("test", help="probe one specific model")
    p_test.add_argument("repo")
    p_test.add_argument("--library", choices=["mlx", "vllm"], default="mlx")

    p_compare = sub.add_parser("compare", help="compare one model across libraries")
    p_compare.add_argument("repo")

    p_gguf = sub.add_parser("gguf", help="probe a GGUF file by URL (llama.cpp)")
    p_gguf.add_argument("url")

    args = parser.parse_args()
    machine = detect_machine()

    if args.command == "hw":
        ui.show_banner(f"{machine.os_name} -- {machine.chip_name}")
        ui.show_machine_panel(machine)
    elif args.command == "search":
        cmd_search_and_rank(args, machine)
    elif args.command == "test":
        cmd_test_one(args, machine)
    elif args.command == "compare":
        cmd_compare_libraries(args, machine)
    elif args.command == "gguf":
        cmd_gguf(args, machine)
    else:
        interactive_wizard(machine)


if __name__ == "__main__":
    sys.exit(main())
