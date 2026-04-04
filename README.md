# Liver tumor segmentation (nnU-Net v2)

Мультиклассовая сегментация печени и опухоли на КТ с [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) на датасете в формате nnU-Net (`Dataset001_LiverTumor`). **coarse_to_fine** — отдельный 2D U-Net по грубой маске nnU-Net. **multiview** — **независимая** надстройка: своя сеть `MultiviewUNet2d` (три фиксированных HU-окна + канал вероятности опухоли), в духе multi-view fusion для узлов на КТ (см. *Deep Multi-View Fusion Network for Lung Nodule Segmentation* и аналоги); чекпоинт обучается отдельно, не из весов coarse_to_fine. Код: `src/multiview`, обучение `scripts/train_multiview.py`, инференс `scripts/infer_multiview.py`. **uncertainty** — ещё одна независимая вторая стадия: `UncertaintyUNet2d` (три HU-окна + вероятность + энтропия Bernoulli), `src/uncertainty`, `scripts/train_uncertainty.py`, `scripts/infer_uncertainty.py`.

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
2. **Export** — слайсы с предсказанием stage-1 и GT для вторых стадий (`scripts/export.py`; общий экспорт для coarse_to_fine, multiview и uncertainty).
3. **coarse_to_fine** — вторая стадия (`scripts/train_coarse_to_fine.sh` или `train_coarse_to_fine.py` → `results_coarse_to_fine/...`).
4. **multiview** — вторая стадия (`scripts/train_multiview.sh` или `train_multiview.py` → `results_multiview/.../multiview/run_*`).
5. **uncertainty** — вторая стадия (`scripts/train_uncertainty.sh` или `train_uncertainty.py` → `results_uncertainty/.../uncertainty/run_*`).
6. **Инференс coarse_to_fine** — полный объём: только nnU-Net или nnU-Net + coarse_to_fine (`scripts/infer_coarse_to_fine.py`).
7. **Инференс multiview** (опционально) — ROI + multi-window (`scripts/infer_multiview.py --multiview-dir ...`; e.g. `inference_comparison/multiview/`).
8. **Инференс uncertainty** (опционально) — `scripts/infer_uncertainty.py --uncertainty-dir ...` (по умолчанию `-o inference_comparison/uncertainty`). Не смешивать каталоги чекпоинтов разных стадий.

**Согласованность stage-1:** по умолчанию и `export.py`, и `infer_coarse_to_fine.py` / `infer_multiview.py` / `infer_uncertainty.py` используют **tile step 0.75** (как общий «лёгкий» бейзлайн). Для одного и того же шага явно задайте одинаковый `--tile-step-size` в нужных скриптах.

Эксперименты по полному инференсу обычно складывают в `inference_comparison/` с подпапками **`baseline`** (только nnU-Net), **`coarse_to_fine`**, **`multiview`**, **`uncertainty`**.

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

## 2. Export слайсов (coarse_to_fine, multiview, uncertainty)

Экспорт **только из обученной базовой nnU-Net** (ничего из вторых стадий сюда не подмешивается). Те же `.npz` используются для `train_coarse_to_fine`, `train_multiview` и `train_uncertainty`. Уже сделанный экспорт для этой же базовой модели можно не переделывать — переэкспорт нужен, если поменялись nnU-Net, данные или параметры экспорта.

После обучения nnU-Net (нужен `fold_0` и `splits_final.json` в препроцессе для train/val):

```bash
python3 scripts/export.py --output-dir refinement_export/fold0
```

Каталог с `train/*.npz` и `val/*.npz` может быть любым; в этом репозитории экспорт лежит в **`refinement_export/fold0/`** (метаданные: `refinement_export/fold0/export_meta.json`). Обёртки `train_* .sh` по умолчанию берут **`--export-dir` оттуда же**.

Опции: `--model-dir`, `--fold`, `--checkpoint`, `--tile-step-size` (по умолчанию `0.75`), `--all-slices`, `--device`.

## 3. Обучение coarse_to_fine (стадия 2)

```bash
bash scripts/train_coarse_to_fine.sh
# эквивалентно: python3 scripts/train_coarse_to_fine.py --export-dir refinement_export/fold0
# другой экспорт: EXPORT_DIR=/path/to/slices bash scripts/train_coarse_to_fine.sh
```

По умолчанию логи и чекпойнты пишутся под  
`results_coarse_to_fine/<dataset>/fold_<n>/coarse_to_fine/run_<timestamp>/`  
(те же `--dataset-folder` и `--fold`, что у nnU-Net / `export.py`; для Liver по умолчанию `Dataset001_LiverTumor`, `fold_0`).

**По умолчанию (в духе *Universal Topology Refinement*, Li et al., arXiv:2409.09796):** второй канал — **вероятность опухоли nnU-Net** (`coarse_tumor_prob`), не бинарная маска; обучение на **ROI**: кроп вокруг (GT ∪ coarse) с паддингом, затем resize в `--crop-size` — **как при инференсе** в `infer_coarse_to_fine.py` (bbox по грубой опухоли). Полиномиальный синтез из статьи не реализован. Отключения: `--no-coarse-prob`, `--no-roi-align`. См. также `--out-dir`, `--epochs`, `--crop-size`, `--roi-pad`, `--min-roi`.

После обучения в каталоге рана будут `checkpoint_best.pth` и `meta.json`; для инференса укажите этот каталог в `infer_coarse_to_fine.py`. По умолчанию раны пишутся в **`results_coarse_to_fine/`**.

Для multiview — **отдельный** каталог ранов: **`results_multiview/`** (не смешивать с `results_coarse_to_fine`). Для uncertainty — **`results_uncertainty/`**. См. разделы 4–5 (обучение и инференс multiview и uncertainty).

## 4. Обучение multiview (MultiviewUNet2d)

Тот же `--export-dir`, что и для coarse_to_fine (раздел 2). Четыре входных канала: три HU-окна (`multiview.config`) + вероятность опухоли nnU-Net. По умолчанию **`--roi-mode infer`**: кроп на срезе строится по **той же «подозрительной» полосе** `[prob_lo, prob_hi]`, что и на инференсе, с паддингами `roi_pad` / `min_roi_side` (ось Y/X); слайсы **без** таких пикселей из датасета отбрасываются. Для старого варианта (ROI = GT∪coarse): `--roi-mode legacy`. `infer_multiview` читает `hu_windows`, `crop_size`, `base` и пр. из `meta.json`.

```bash
bash scripts/train_multiview.sh
# эквивалентно: python3 scripts/train_multiview.py --export-dir refinement_export/fold0
# другой экспорт: EXPORT_DIR=/path/to/slices bash scripts/train_multiview.sh
```

По умолчанию логи и чекпоинты: `results_multiview/<dataset>/fold_<n>/multiview/run_<timestamp>/`  
(зеркально coarse_to_fine: папка задачи `multiview` внутри fold). Дальше: `python scripts/infer_multiview.py ... --multiview-dir <этот_ран>`.

Старые раны multiview без подпапки `multiview/` (напрямую `.../fold_0/run_*`) остаются валидными: укажите путь к каталогу с `checkpoint_best.pth` как и раньше.

### 4.1. Обучение uncertainty (UncertaintyUNet2d)

Тот же `--export-dir`, что в разделах 2–4 (по умолчанию **`refinement_export/fold0`**). Пять входных каналов: три HU-окна, вероятность опухоли nnU-Net, нормированная энтропия. Чекпоинты: `results_uncertainty/<dataset>/fold_<n>/uncertainty/run_<timestamp>/` (отдельно от `results_coarse_to_fine` и `results_multiview`).

```bash
bash scripts/train_uncertainty.sh
# эквивалентно: python3 scripts/train_uncertainty.py --export-dir refinement_export/fold0
# другой экспорт: EXPORT_DIR=/path/to/slices bash scripts/train_uncertainty.sh
```

Дальше: `python3 scripts/infer_uncertainty.py ... --uncertainty-dir <каталог_с_checkpoint_best.pth_и_meta.json>` (по умолчанию `-o inference_comparison/uncertainty`). Справка: `python3 scripts/infer_uncertainty.py -h`.

## 5. Инференс на полных объёмах

### 5.1. nnU-Net и/или coarse_to_fine (`infer_coarse_to_fine.py`)

- **Только nnU-Net (baseline):** тот же стек предиктора, что и во второй стадии, с шагом **0.75** по умолчанию:

  ```bash
  python scripts/infer_coarse_to_fine.py --stage1-only -i <папка_с_КТ_как_в_nnUNet_raw> -o <куда_писать_сегментации>
  ```

- **Две стадии (nnU-Net + coarse_to_fine):**

  ```bash
  python scripts/infer_coarse_to_fine.py -i <вход> -o <выход> --coarse-to-fine-dir <папка_рана_с_meta.json_и_checkpoint_best.pth>
  ```

Общие флаги: `--tile-step-size`, `--no-mirroring`, `--fold`, `--model-dir`, `--export-stage1-to` (опционально сохранить stage-1), **`--save-probabilities`** (полный softmax nnU-Net: `case_id.npz` + `case_id.pkl` + stage-1 `case_id.nii.gz`; для двух стадий по умолчанию каталог `<out_dir>/nnunet_stage1_softmax`, чтобы не перезаписать финальную маску). Для метрик на **валидации** того же fold: `--split val` (список кейсов из `nnUNet_preprocessed/.../splits_final.json`). После OOM или обрыва: тот же `-o` и **`--skip-existing`**, чтобы не пересчитывать готовые кейсы. Справка: `python scripts/infer_coarse_to_fine.py -h`.

### 5.2. Multiview (`infer_multiview.py`)

Отдельный скрипт: nnU-Net + **MultiviewUNet2d** по ROI (не путать с coarse_to_fine). Пример:

```bash
python scripts/infer_multiview.py -i <imagesTr_или_аналог> -o <выход> --multiview-dir <results_multiview/.../multiview/run_<timestamp>>
```

Справка: `python scripts/infer_multiview.py -h`.

### 5.3. Uncertainty (`infer_uncertainty.py`)

Отдельный скрипт: nnU-Net + **UncertaintyUNet2d** по ROI (не путать с coarse_to_fine / multiview). Пример:

```bash
python3 scripts/infer_uncertainty.py -i <imagesTr_или_аналог> --uncertainty-dir <results_uncertainty/.../uncertainty/run_<timestamp>> -o inference_comparison/uncertainty
```

Справка: `python3 scripts/infer_uncertainty.py -h`.

## 6. Метрики на полных объёмах (Dice / IoU опухоли)

После инференса — отдельно, по сравнению с разметкой (те же имена кейсов, что в `labelsTr`):

```bash
python scripts/evaluate_segmentations.py \
  --pred-dir <папка_с_предсказаниями> \
  --gt-dir nnUNet_raw/Dataset001_LiverTumor/labelsTr \
  --output-json metrics.json
```

Класс опухоли по умолчанию читается из `dataset.json`; при необходимости: `--tumor-label 2`.

## Структура репозитория

```
liver-tumor-segmentation/
├── README.md
├── requirements.txt
├── LICENSE
├── src/coarse_to_fine/   # стадия coarse_to_fine: модель, датасет, обучение
├── src/multiview/        # MultiviewUNet2d, ROI, окна HU (независимо от coarse_to_fine)
├── src/uncertainty/      # UncertaintyUNet2d, энтропия, ROI (независимо от других вторых стадий)
├── data/
├── nnUNet_raw/
├── nnUNet_preprocessed/
├── nnUNet_results/
├── refinement_export/        # экспорт слайсов (e.g. fold0/train, fold0/val, export_meta.json)
├── results_coarse_to_fine/   # только coarse_to_fine (чекпойнты и логи)
├── results_multiview/        # только multiview / MultiviewUNet2d (не в git)
├── results_uncertainty/      # только uncertainty / UncertaintyUNet2d (не в git)
└── scripts/
    ├── train.sh                  # стадия 1: nnU-Net план / препроцесс / train
    ├── train_coarse_to_fine.sh   # обёртка: PYTHONPATH, EXPORT_DIR → train_coarse_to_fine.py
    ├── train_multiview.sh        # то же для train_multiview.py
    ├── train_uncertainty.sh      # то же для train_uncertainty.py
    ├── export.py                 # слайсы + stage-1 (общий экспорт для вторых стадий)
    ├── train_coarse_to_fine.py   # обучение coarse_to_fine
    ├── train_multiview.py        # обучение MultiviewUNet2d
    ├── train_uncertainty.py      # обучение UncertaintyUNet2d
    ├── infer_coarse_to_fine.py   # полный объём: stage 1 или 1+coarse_to_fine
    ├── infer_multiview.py        # nnU-Net + MultiviewUNet2d
    ├── infer_uncertainty.py      # nnU-Net + UncertaintyUNet2d
    └── evaluate_segmentations.py # Dice/IoU опухоли vs labelsTr
```

## Лицензия

См. `LICENSE`.
