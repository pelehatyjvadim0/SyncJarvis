"""Форматирование стрима агента в консоли (rich, префиксы [THOUGHT] и т.д.)."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Единый консольный вывод для стрима и панелей финального отчёта.
console = Console()


async def print_stream_message(message: str) -> None:
    # Печать стрима с лёгкой rich-стилизацией (не меняя исходный смысл сообщений).
    text = message or ""
    if text.startswith("[THOUGHT]") or "\n[THOUGHT]\n" in text:
        console.print(f"[bold cyan]{text}[/bold cyan]")
        return
    if text.startswith("[PLANNER]"):
        console.print(f"[bold magenta]{text}[/bold magenta]")
        return
    if text.startswith("[SELF-CHECK]"):
        console.print(f"[yellow]{text}[/yellow]")
        return
    if text.startswith("[COST]"):
        console.print(f"[green]{text}[/green]")
        return
    if text.startswith("[FINAL-REPORT]"):
        console.print(f"[bold bright_blue]{text}[/bold bright_blue]")
        return
    console.print(text)


def print_final_report_panel(markdown_body: str) -> None:
    """Панель с Markdown для итогового отчёта (как в точке входа приложения)."""
    console.print(
        Panel(
            Markdown(markdown_body),
            title="[bold green]FINAL REPORT",
            border_style="bright_blue",
        )
    )
