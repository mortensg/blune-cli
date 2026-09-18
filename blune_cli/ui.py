"""Rich-based terminal UI helpers -- the visual layer, kept separate from
the probing logic so the library modules stay usable headless/scriptable."""
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

BANNER = r"""
[bold cyan] _     _                    [/][bold magenta]  ____ _     ___ [/]
[bold cyan]| |__ | |_   _ _ __   ___   [/][bold magenta] / ___| |   |_ _|[/]
[bold cyan]| '_ \| | | | | '_ \ / _ \  [/][bold magenta]| |   | |    | | [/]
[bold cyan]| |_) | | |_| | | | |  __/  [/][bold magenta]| |___| |___ | | [/]
[bold cyan]|_.__/|_|\__,_|_| |_|\___|  [/][bold magenta] \____|_____|___|[/]
"""


def show_banner(machine_line: str):
    console.print(BANNER)
    console.print(
        Panel(
            Text(machine_line, style="dim"),
            border_style="cyan",
            box=ROUNDED,
            padding=(0, 1),
        )
    )
    console.print()


def show_machine_panel(machine):
    bw = f"{machine.bandwidth_gbs:.0f} GB/s" if machine.bandwidth_gbs else "unknown"
    lines = (
        f"[bold]{machine.chip_name}[/]\n"
        f"RAM: {machine.total_ram_gb:.0f} GB   |   Bandwidth: {bw}"
    )
    console.print(Panel(lines, title="Your machine", border_style="green", box=ROUNDED))


def show_ranking(results: list[dict]):
    table = Table(title="Ranking (fastest first)", box=ROUNDED, header_style="bold cyan")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Library")
    table.add_column("Est. tok/s", justify="right")
    table.add_column("Source")

    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r.get("estimated_real_tps", r.get("real_decode_tps", 0)), reverse=True)

    for i, r in enumerate(ok, 1):
        tps = r.get("estimated_real_tps", r.get("real_decode_tps"))
        source = "[green]measured[/]" if r.get("source") == "measured" else "[yellow]estimated[/]"
        table.add_row(
            str(i),
            r.get("repo_id", "?"),
            r.get("library", "?"),
            f"{tps:.1f}" if tps else "-",
            source,
        )
    console.print(table)

    failed = [r for r in results if "error" in r]
    if failed:
        console.print()
        for r in failed:
            console.print(f"  [red]FAILED[/] {r['repo_id']}: {r['error']}")


def show_single_result(r: dict):
    lines = []
    for k, v in r.items():
        if k in ("repo_id",):
            continue
        lines.append(f"[dim]{k}:[/] {v}")
    console.print(
        Panel(
            "\n".join(lines),
            title=r.get("repo_id", "result"),
            border_style="cyan",
            box=ROUNDED,
        )
    )


def error(msg: str):
    console.print(f"[bold red]Error:[/] {msg}")


def info(msg: str):
    console.print(f"[dim]{msg}[/]")


def progress_step(i: int, total: int, repo_id: str):
    """Overwrite the same line with '[i/total] probing <repo>' so a long
    search+rank run (dozens to hundreds of candidates, each needing a
    network fetch and a real probe) shows visible progress instead of
    looking hung."""
    console.print(f"[dim][{i}/{total}][/] probing {repo_id}...", end="\r")
