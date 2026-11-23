# Unit Testing para Ciencia de Datos — Tarea 3

Proyecto mínimo con dos clases de utilidades y sus esqueletos de pruebas.

## Estructura

- `src/metrics_utils.py` — `MetricsUtils`: `rmse`, `mae`, `accuracy`, `confusion_matrix` (NumPy).
- `src/feature_encoder.py` — `FeatureEncoder`: `one_hot`, `label_encode`, `bin_numeric` (pandas).
- `src/mutants/metrics_utils_m1.py` — copia de `MetricsUtils` con **un bug plantado** (no editar).
- `tests/test_metrics_utils.py` — 2 tests de ejemplo + 7 a completar.
- `tests/test_feature_encoder.py` — 2 tests de ejemplo + 7 a completar.

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar las pruebas (siempre desde la raíz del repositorio)

```bash
python -m unittest discover -v          # todos los tests
python -m unittest tests.test_metrics_utils -v   # un archivo
```

Los tests pendientes fallan con `TODO: implementar este test` hasta que los complete.

## Prueba de mutación

```bash
# Windows (PowerShell):  $env:MCD_MUTANT="1"; python -m unittest tests.test_metrics_utils
# macOS/Linux:
MCD_MUTANT=1 python -m unittest tests.test_metrics_utils
```

Con la variable `MCD_MUTANT=1` la suite corre contra la versión con bug.
Una suite correcta hace que **al menos un test falle** en este modo.
