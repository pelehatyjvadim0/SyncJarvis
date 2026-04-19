"""Фазы одного шага `run_subtask_pipeline` (см. `engine/pipeline.py`).

Граф вызовов за итерацию (сверху вниз):
  ``pipeline.run_subtask_pipeline`` →
  ``observation_phase`` → ``self_check_phase`` → ``grounding_phase`` →
  ``vision_recovery_phase`` → ``llm_decision_phase`` → ``execute_phase``.

Вспомогательный модуль ``persist_metrics`` (не шаг пайплайна): вызывается из
``grounding_phase`` и ``vision_recovery_phase`` там же, где раньше вызывался
``update_performance_metrics`` в монолитном цикле. Путь LLM+исполнение —
``execute_phase`` → ``action_executor.run_guarded_action_with_fingerprint_and_metrics``.
"""
