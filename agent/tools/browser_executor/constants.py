from __future__ import annotations

TEXT_INPUT_ROLES = {"textbox", "searchbox", "combobox"}

# Порог «маленькой» кнопки в карточках (кол-во, +/-): у них часто одинаковые role+name, nth() бьёт не в ту карточку - геометрия по bbox_doc привязана к конкретному a11y-узлу.
_SMALL_CONTROL_BBOX_MAX_PX = 56
