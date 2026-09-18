#!/usr/bin/env python3
"""
blune-cli: find out which model runs best, how fast, in which library,
on the machine you're actually sitting at.

Interactive by default (run with no arguments -- starts broad, narrows down
as you choose). Every choice is also available as a direct flag for
scripting: `blune --search --library mlx --limit 20 --rank`.
"""
import argparse
import json
import subprocess
import sys

from rich.prompt import IntPrompt, Prompt

from . import (
    config_cache,
    measurements,
    probe_formula,
    probe_llamacpp,
    probe_mlx,
    probe_vllm,
    search,
    size_estimate,
    ui,
)
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
    for i, repo in enumerate(repos, 1):
        ui.progress_step(i, len(repos), repo)
        try:
            r = run_probe(repo, args.library, machine, args.offline)
            r["repo_id"] = repo
            results.append(r)
        except Exception as e:
            results.append({"repo_id": repo, "error": str(e)})
    ui.console.print()
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


def cmd_sync_configs(args, machine):
    """Bulk-populate the curated config cache from every text-generation
    model an org has published -- config.json only, no probing (probing
    thousands of repos would take hours; caching their configs takes
    minutes and is the actual "don't spam Hugging Face" deliverable)."""
    already = set(config_cache.list_curated())
    ui.info(f"listing {args.author}'s models on the Hub...")
    repos = search.list_org_models(author=args.author, max_results=args.limit)
    todo = [r for r in repos if r not in already]
    ui.info(
        f"found {len(repos)} model(s), {len(already & set(repos))} already cached, "
        f"fetching {len(todo)}...\n"
    )

    fetched, failed = 0, []
    for i, repo in enumerate(todo, 1):
        ui.progress_step(i, len(todo), repo, action="fetching")
        try:
            config_cache.get_config(repo, offline=False, save_curated=True)
            fetched += 1
        except Exception as e:
            failed.append((repo, str(e)))
    ui.console.print()
    ui.info(f"done: {fetched} new config(s) cached, {len(failed)} failed.")
    if failed:
        for repo, err in failed[:20]:
            ui.console.print(f"  [red]FAILED[/] {repo}: {err}")
        if len(failed) > 20:
            ui.console.print(f"  ...and {len(failed) - 20} more")


def _run_probe_isolated(repo_id: str, library: str, timeout: int) -> dict:
    """Run one probe in a fresh subprocess (see _probe_worker.py) so a
    crash on one model can't take the whole sweep down, and enforce a
    wall-clock timeout so one stuck model can't stall it forever."""
    proc = subprocess.run(
        [sys.executable, "-m", "blune_cli._probe_worker", repo_id, "--library", library],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        try:
            err = json.loads(proc.stderr.strip().splitlines()[-1])["error"]
        except Exception:
            err = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise RuntimeError(err)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def cmd_sweep(args, machine):
    """Probe every model already in the curated config cache, one at a
    time, printing a permanent result line for each as it finishes --
    unlike cmd_search_and_rank (which only shows a final table), this is
    built for a run over hundreds/thousands of models where you want to
    see progress accumulate, not wonder if it's still alive."""
    if (args.formula or args.compare) and not machine.bandwidth_gbs:
        ui.error(
            "Unknown machine bandwidth -- the formula estimate needs this. "
            "Run `blune hw` to check what was detected."
        )
        return

    repos = sorted(config_cache.list_curated())
    if args.limit:
        repos = repos[: args.limit]

    if args.compare:
        mode = "compare (real probe + formula side by side)"
    elif args.formula:
        mode = "formula (instant, ~9.6% avg error)"
    else:
        mode = f"real probe, {args.timeout}s/model timeout"
    ui.info(f"sweeping {len(repos)} cached model(s) (library={args.library}, mode={mode})...\n")

    machine_key = _machine_key(machine)
    ok, failed, skipped = 0, 0, 0

    for i, repo in enumerate(repos, 1):
        real = measurements.find_real_measurement(repo, args.library, machine=machine_key)
        if real:
            ui.stream_result(i, len(repos), repo, f"{real['real_decode_tps']} tok/s (measured)", "green")
            ok += 1
            continue

        try:
            config = config_cache.get_config(repo, offline=True)
        except Exception as e:
            ui.stream_result(i, len(repos), repo, f"FAILED: {e}", "red")
            failed += 1
            continue

        if args.compare:
            try:
                formula_tps = probe_formula.probe(
                    repo, config, machine.bandwidth_gbs, context_length=args.context
                )["estimated_real_tps"]
            except Exception as e:
                formula_tps = None
                formula_err = str(e)

            if machine.total_ram_gb and not size_estimate.fits_in_ram(config, machine.total_ram_gb):
                text = "SKIPPED: too large for RAM (real)"
                if formula_tps is not None:
                    text += f" | {formula_tps} tok/s (formula)"
                ui.stream_result(i, len(repos), repo, text, "yellow")
                skipped += 1
                continue

            ui.stream_in_progress(i, len(repos), repo)
            try:
                real_tps = _run_probe_isolated(repo, args.library, args.timeout)["estimated_real_tps"]
                if formula_tps is not None:
                    delta = (formula_tps - real_tps) / real_tps * 100
                    text = f"{real_tps} tok/s (real) vs {formula_tps} tok/s (formula), Δ{delta:+.1f}%"
                else:
                    text = f"{real_tps} tok/s (real) | formula FAILED: {formula_err}"
                ui.stream_result(i, len(repos), repo, text, "cyan")
                ok += 1
            except subprocess.TimeoutExpired:
                text = f"real probe timed out after {args.timeout}s"
                if formula_tps is not None:
                    text += f" | {formula_tps} tok/s (formula)"
                ui.stream_result(i, len(repos), repo, f"FAILED: {text}", "red")
                failed += 1
            except Exception as e:
                text = f"real probe FAILED: {e}"
                if formula_tps is not None:
                    text += f" | {formula_tps} tok/s (formula)"
                ui.stream_result(i, len(repos), repo, text, "red")
                failed += 1
            continue

        if args.formula:
            try:
                r = probe_formula.probe(repo, config, machine.bandwidth_gbs, context_length=args.context)
                tps = r["estimated_real_tps"]
                ui.stream_result(i, len(repos), repo, f"{tps} tok/s (formula)", "magenta")
                ok += 1
            except Exception as e:
                ui.stream_result(i, len(repos), repo, f"FAILED: {e}", "red")
                failed += 1
            continue

        if machine.total_ram_gb and not size_estimate.fits_in_ram(config, machine.total_ram_gb):
            ui.stream_result(i, len(repos), repo, "SKIPPED: too large for this machine's RAM", "yellow")
            skipped += 1
            continue

        ui.stream_in_progress(i, len(repos), repo)
        try:
            r = _run_probe_isolated(repo, args.library, args.timeout)
            tps = r.get("estimated_real_tps")
            ui.stream_result(i, len(repos), repo, f"{tps} tok/s (estimated)", "cyan")
            ok += 1
        except subprocess.TimeoutExpired:
            ui.stream_result(i, len(repos), repo, f"FAILED: timed out after {args.timeout}s", "red")
            failed += 1
        except Exception as e:
            ui.stream_result(i, len(repos), repo, f"FAILED: {e}", "red")
            failed += 1

    ui.console.print()
    ui.info(f"done: {ok} ok, {failed} failed, {skipped} skipped (too large).")


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

    p_sync = sub.add_parser(
        "sync-configs", help="bulk-populate the curated config cache from an org's models"
    )
    p_sync.add_argument("--author", default="mlx-community")
    p_sync.add_argument("--limit", type=int, default=None, help="cap on how many to fetch")

    p_sweep = sub.add_parser(
        "sweep", help="probe every cached model one at a time, streaming results as they finish"
    )
    p_sweep.add_argument("--library", choices=["mlx", "vllm"], default="mlx")
    p_sweep.add_argument("--limit", type=int, default=None, help="cap on how many to probe")
    p_sweep.add_argument("--timeout", type=int, default=120, help="per-model timeout in seconds")
    p_sweep.add_argument(
        "--formula",
        action="store_true",
        help="instant config-only math estimate instead of actually running each model "
        "(much faster, ~9.6%% avg error -- see probe_formula.py)",
    )
    p_sweep.add_argument(
        "--compare",
        action="store_true",
        help="run both the real probe and the formula estimate for each model, "
        "showing both plus the delta between them",
    )
    p_sweep.add_argument(
        "--context",
        type=int,
        default=115,
        help="context length (tokens already cached) to assume for the formula's "
        "KV-cache term -- longer contexts show slower speed for models without "
        "sliding-window/hybrid-SSM layers (default 115, matching this project's "
        "own MLX calibration measurements)",
    )

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
    elif args.command == "sync-configs":
        cmd_sync_configs(args, machine)
    elif args.command == "sweep":
        cmd_sweep(args, machine)
    else:
        interactive_wizard(machine)


if __name__ == "__main__":
    sys.exit(main())
