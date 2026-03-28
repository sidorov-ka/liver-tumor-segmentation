# Liver tumor segmentation (nnU-Net v2)

Мультиклассовая сегментация печени и опухоли на КТ с [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) на датасете в формате nnU-Net (`Dataset001_LiverTumor`). В репозитории также есть **вторая стадия** — небольшой 2D U-Net для уточнения маски опухоли по грубой маске nnU-Net.

## Требования

- Python 3.10+
- GPU с CUDA (рекомендуется для обучения и инференса)
- Зависимости из `requirements.txt` (включая nnU-Net v2)

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Переменные окружения nnU-Net (если не заданы, подставляются каталоги **в корне репозитория**):

| Переменная | Назначение |
|------------|------------|
| `nnUNet_raw` | сырые данные (`<repo>/nnUNet_raw`) |
| `nnUNet_preprocessed` | препроцесс (`<repo>/nnUNet_preprocessed`) |
| `nnUNet_results` | чекпойнты nnU-Net (`<repo>/nnUNet_results`) |

## Данные

Структура [raw dataset](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md):

```
nnUNet_raw/Dataset001_LiverTumor/
├── dataset.json
├── imagesTr/
└── labelsTr/
```

При изменении числа кейсов обновите `numTraining` в `dataset.json`.

## Пайплайн (кратко)

1. **nnU-Net** — планирование, препроцесс, обучение fold 0 (`scripts/train.sh`).
2. **Export** — слайсы с предсказанием stage-1 и GT для обучения refiner (`scripts/export.py`).
3. **Refinement** — обучение второй стадии (`scripts/train_refiner.py` → `refinement_results/...`).
4. **Infer** — полный объём: только nnU-Net или nnU-Net + refiner (`scripts/infer.py`).

**Согласованность stage-1:** по умолчанию и `export.py`, и `infer.py` используют **tile step 0.75** (как общий «лёгкий» бейзлайн). Для одного и того же шага явно задайте одинаковый `--tile-step-size` в обоих скриптах.

## 1. Обучение nnU-Net

Из корня репозитория (с активированным `.venv`):

```bash
bash scripts/train.sh
```

Скрипт выставляет `nnUNet_*` при необходимости, запускает `nnUNetv2_plan_and_preprocess` для конфигурации `2d` с `--clean`, затем `nnUNetv2_train` для fold `0` и тренера `nnUNetTrainer_100epochs`.

Повторный запуск **без** пересчёта препроцесса (если кэш уже валиден):

```bash
bash scripts/train.sh --skip-preprocess
# или: SKIP_NNUNET_PREPROCESS=1 bash scripts/train.sh
```

Продолжить обучение nnU-Net с последнего чекпойнта:

```bash
export nnUNet_raw=$PWD/nnUNet_raw nnUNet_preprocessed=$PWD/nnUNet_preprocessed nnUNet_results=$PWD/nnUNet_results
nnUNetv2_train 1 2d 0 -tr nnUNetTrainer_100epochs --c
```

### Параметры по умолчанию (nnU-Net)

| Параметр | Значение |
|----------|----------|
| Dataset ID | `1` |
| Конфигурация | `2d` |
| Fold | `0` |
| Trainer | `nnUNetTrainer_100epochs` |
| Препроцесс | `-np 1 -npfp 1` (мало RAM) |

## 2. Export слайсов для refinement

После обучения nnU-Net (нужен `fold_0` и `splits_final.json` в препроцессе для train/val):

```bash
python scripts/export.py --output-dir refinement_export/slices
```

Опции: `--model-dir`, `--fold`, `--checkpoint`, `--tile-step-size` (по умолчанию `0.75`), `--all-slices`, `--device`.

## 3. Обучение refinement (стадия 2)

```bash
python scripts/train_refiner.py --export-dir refinement_export/slices
```

По умолчанию логи и чекпойнты пишутся под `refinement_results/Dataset001_LiverTumor/fold_0/run_<timestamp>/`. См. `--out-dir`, `--epochs`, `--use-coarse-prob`, `--crop-size`.

После обучения в каталоге рана будут `checkpoint_best.pth` и `meta.json` — их указывает `infer.py` для двухстадийного инференса.

## 4. Инференс на полных объёмах

- **Только nnU-Net (бейзлайн):** тот же стек предиктора, что и во второй стадии, с шагом **0.75** по умолчанию:

  ```bash
  python scripts/infer.py --stage1-only -i <папка_с_КТ_как_в_nnUNet_raw> -o <куда_писать_сегментации>
  ```

- **Две стадии (nnU-Net + refiner):**

  ```bash
  python scripts/infer.py -i <вход> -o <выход> --refinement-dir <папка_рана_с_meta.json_и_checkpoint_best.pth>
  ```

Общие флаги: `--tile-step-size`, `--no-mirroring`, `--fold`, `--model-dir`, `--export-stage1-to` (опционально сохранить stage-1). Справка: `python scripts/infer.py -h`.

## Структура репозитория

```
liver-tumor-segmentation/
├── README.md
├── requirements.txt
├── LICENSE
├── src/refinement/       # модель, датасет, обучение refinement
├── data/
├── nnUNet_raw/
├── nnUNet_preprocessed/
├── nnUNet_results/
├── refinement_results/   # логи и чекпойнты второй стадии (не в git)
└── scripts/
    ├── train.sh          # nnU-Net: план / препроцесс / train
    ├── export.py         # слайсы + stage-1 для refinement
    ├── train_refiner.py  # обучение второй стадии
    └── infer.py          # полный объём: stage 1 или 1+2
```

## Лицензия

См. `LICENSE`.
