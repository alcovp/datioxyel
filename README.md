# datioxyel

Репозиторий для проверки возможностей `codex exec parallel` на подопытном проекте `cobaia`.

## Подтверждённые результаты (2026-02-22)

- Run `20260222_165647` (`parallel_review_baseline`, 30 воркеров, без stagger): `success=15`, `failed=15`; в failed-логах общий финал — `429 Too Many Requests`.
- Run `20260222_172816` (`parallel_review_baseline`, 30 воркеров, со stagger запуска 1-10 сек): `success=30`, `failed=0`.
- Вывод: ключевой bottleneck при `30` — не общий объём работы, а стартовый burst запросов к API.
- Практика для стресс-прогонов на `30`: использовать запуск со stagger (`scripts/run_parallel_review_baseline_stagger.sh`) или эквивалентный jitter старта в новых сценариях.
