# liver-tumor-segmentation

Сегментация опухолей печени с nnU-Net (датасет LiTS-style, Kaggle). ВКР.

## Структура

```
liver-tumor-segmentation/
├── README.md
├── .gitignore
├── requirements.txt
├── LICENSE
├── data/
│   ├── README.md
│   └── downloaded_path.txt
├── nnUNet_raw/                # сырые датасеты
├── nnUNet_preprocessed/       # препроцесс nnU-Net
├── nnUNet_results/            # чекпоинты и результаты
└── scripts/
    ├── download_dataset.py    # загрузка LITS17 с Kaggle
    ├── prepare_dataset.py     # конвертация в nnUNet_raw/Dataset001_LiverTumor
    ├── train.sh               # план + препроцесс + обучение nnU-Net v2
    └── inference.sh           # предсказание
```

## Зависимости

- **Скачивание:** `pip install -r requirements.txt`
- **Обучение:** nnUNet v2 (в requirements.txt или `pip install nnunetv2`)

## Шаги

1. **Данные** — скачай LITS17 ([Kaggle](https://www.kaggle.com/datasets/javariatahir/litstrain-val/data)) или запусти `python scripts/download_dataset.py` (нужен `.env` с `KAGGLE_API_TOKEN`). Запиши путь в `data/downloaded_path.txt` и выполни `python scripts/prepare_dataset.py`.

2. **Обучение** — из корня проекта (с активированным venv):
   ```bash
   bash scripts/train.sh
   ```
   Скрипт сам задаёт переменные nnUNet_raw, nnUNet_preprocessed, nnUNet_results.

3. **Инференс** — `bash scripts/inference.sh /path/to/test/images /path/to/predictions`

## Упрощённое обучение

Для снижения нагрузки измени в `train.sh`:
- `nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -np 1` (уже есть -np 1)
- `nnUNetv2_train "$DATASET_ID" 2d 0 -tr nnUNetTrainer_50epochs` вместо 3d_fullres
