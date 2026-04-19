from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InteractiveElement(BaseModel):
    ax_id: str = Field(..., description="Идентификатор узла из пути в дереве доступности")
    role: str
    name: str | None = None
    dom_id: str | None = None
    disabled: bool | None = None
    focused: bool | None = None
    checked: bool | Literal["mixed"] | None = None
    pressed: bool | Literal["mixed"] | None = None
    expanded: bool | None = None
    selected: bool | None = None
    value: str | float | int | None = None
    index_within_role_name: int = 0
    # Ось layout viewport в CSS px документа (как у CDP DOM.getBoxModel content quad): левый верх content box.
    bbox_doc_x: float | None = None
    bbox_doc_y: float | None = None
    bbox_doc_w: float | None = None
    bbox_doc_h: float | None = None
    # Ближайший «контейнер» вверх по AX (section/article/listitem/heading с текстом) — для промпта и дизambigвации карточек.
    parent_anchor: str | None = None

