"""Детекция и обработка капчи в цикле подзадачи."""

from .detect import is_captcha_present
from .solver import handle_captcha_iteration

__all__ = ["handle_captcha_iteration", "is_captcha_present"]
