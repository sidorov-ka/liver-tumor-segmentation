# Data

Рекомендуемый датасет: **LITS17** [Kaggle: javariatahir/litstrain-val](https://www.kaggle.com/datasets/javariatahir/litstrain-val/data) (~53 GB). Структура: `train_CT/volume-*.nii`, `train_mask/segmentation-*.nii`.

## Скачивание через скрипт

1. Установи зависимости: `pip install -r requirements.txt` (или в venv).
2. Токен Kaggle: скопируй `.env.example` в `.env`, подставь свой `KAGGLE_API_TOKEN` (или задай `export KAGGLE_API_TOKEN=...` в терминале; или положи `~/.kaggle/kaggle.json`). Файл `.env` в репо не коммитится.
3. Из корня проекта: `python scripts/download_dataset.py`. Путь сохранится в `downloaded_path.txt`.

## Вручную

Скачай датасет с Kaggle, распакуй. Запиши **полный путь** к корню (где лежат `train_CT/` и `train_mask/`) в `downloaded_path.txt` (одна строка). Дальше: `python scripts/prepare_dataset.py` — данные попадут в `nnUNet_raw/Dataset001_LiverTumor/`.
