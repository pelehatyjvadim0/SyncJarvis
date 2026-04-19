"""Интерактивная консольная сессия: браузер, оркестратор, подтверждение опасных действий."""

from __future__ import annotations

from agent.config.settings import AppSettings
from agent.models.state import AgentState
from agent.perception.accessibility import collect_interactive_elements
from agent.runtime.orchestrator import TaskOrchestrator
from agent.tools.browser_executor import BrowserToolExecutor

from agent.cli.streaming import console, print_final_report_panel, print_stream_message


async def run_console_session(settings: AppSettings) -> None:
    executor = BrowserToolExecutor(
        headless=settings.browser_headless,
        cdp_url=settings.browser_cdp_url,
        viewport_width=settings.browser_viewport_width,
        viewport_height=settings.browser_viewport_height,
        navigate_wait_until=settings.browser_navigate_wait_until,
        navigate_timeout_ms=settings.browser_navigate_timeout_ms,
        navigate_networkidle_timeout_ms=settings.browser_navigate_networkidle_timeout_ms,
        navigate_post_settle_seconds=settings.browser_navigate_post_settle_seconds,
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

            state = await orchestrator.run(user_goal=user_goal, stream_callback=print_stream_message)

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
                    stream_callback=print_stream_message,
                    confirmed_result=confirmed,
                )

            if state == AgentState.FINISHED:
                console.print("[bold green]Задача успешно завершена.[/bold green]")
                if orchestrator.final_report_markdown:
                    print_final_report_panel(orchestrator.final_report_markdown)
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
