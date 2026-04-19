from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from agent.cli.session import run_console_session
from agent.config.settings import load_app_settings


async def main() -> None:
    # Загрузка .env, настроек и запуск сессионного цикла: несколько задач подряд в одном браузере.
    load_dotenv()
    settings = load_app_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("Укажи OPENROUTER_API_KEY в переменных окружения или в .env")
    await run_console_session(settings)


if __name__ == "__main__":
    asyncio.run(main())
