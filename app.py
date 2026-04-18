from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.config.settings import load_app_settings
from agent.models.state import AgentState
from agent.perception.accessibility import collect_interactive_elements
from agent.runtime.orchestrator import TaskOrchestrator
from agent.tools.browser_executor import BrowserToolExecutor

console = Console()


async def _print_stream(message: str) -> None:
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


async def main() -> None:
    # Загрузка .env, настроек и запуск сессионного цикла: несколько задач подряд в одном браузере.
    load_dotenv()
    settings = load_app_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("Укажи OPENROUTER_API_KEY в переменных окружения или в .env")

    executor = BrowserToolExecutor(
        headless=settings.browser_headless,
        cdp_url=settings.browser_cdp_url,
    )
    await executor.start()

    try:
        console.print("[bold green]Браузер запущен. Можно выполнять задачи подряд в одной сессии.[/bold green]")
        console.print("[dim]Чтобы выйти, отправьте пустую строку в поле задачи или Ctrl+C.[/dim]")
        orchestrator = TaskOrchestrator(executor=executor, settings=settings)
        while True:
            user_goal = input("\nВведите задачу для агента: ").strip()
            if not user_goal:
                console.print("[yellow]Пустая задача — завершаю агент-сессию. Браузер не закрываю.[/yellow]")
                return

            state = await orchestrator.run(user_goal=user_goal, stream_callback=_print_stream)

            while state == AgentState.AWAITING_USER_CONFIRMATION:
                console.print("[bold yellow]Агент ожидает подтверждения пользователя для опасного действия.[/bold yellow]")
                if not executor.has_pending_dangerous_action():
                    console.print("[yellow]Нет отложенного действия, остановка.[/yellow]")
                    break
                ans = input("Выполнить опасное действие в браузере? (y/n): ").strip().lower()
                if ans != "y":
                    executor.clear_pending_dangerous_action()
                    console.print("[red]Действие отменено.[/red]")
                    state = AgentState.ERROR
                    break
                page = executor.page
                if not page:
                    console.print("[red]Нет страницы браузера.[/red]")
                    state = AgentState.ERROR
                    break
                observation = await collect_interactive_elements(page)
                confirmed = await executor.execute_pending_dangerous_confirmation(observation)
                if not confirmed.success:
                    console.print(f"[red]Не удалось выполнить после подтверждения:[/red] {confirmed.message}")
                    state = AgentState.ERROR
                    break
                state = await orchestrator.resume_after_dangerous_confirmation(
                    stream_callback=_print_stream,
                    confirmed_result=confirmed,
                )

            if state == AgentState.FINISHED:
                console.print("[bold green]Задача успешно завершена.[/bold green]")
                if orchestrator.final_report_markdown:
                    console.print(
                        Panel(
                            Markdown(orchestrator.final_report_markdown),
                            title="[bold green]FINAL REPORT",
                            border_style="bright_blue",
                        )
                    )
            elif state == AgentState.PARTIAL:
                console.print("[yellow]Задача завершена частично (были пропуски подзадач по лимиту или см. лог).[/yellow]")
            elif state == AgentState.SUBTASK_STEP_LIMIT:
                console.print("[yellow]Исчерпан лимит шагов подзадачи.[/yellow]")
            elif state == AgentState.BLOCKED_CAPTCHA:
                console.print("[yellow]Остановка: капча не решена в отведённое число попыток ожидания.[/yellow]")
            elif state == AgentState.AWAITING_USER_CONFIRMATION:
                console.print("[yellow]Ожидание подтверждения прервано.[/yellow]")
            elif state == AgentState.ERROR:
                console.print("[bold red]Агент завершил работу с ошибкой или проверка цели не прошла.[/bold red]")
            else:
                console.print(f"[yellow]Итоговое состояние:[/yellow] {state.value}")
    finally:
        # По запросу пользователя браузер не закрываем автоматически.
        pass


if __name__ == "__main__":
    asyncio.run(main())
