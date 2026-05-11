# Liver tumor segmentation (nnU-Net v2)

Мультиклассовая сегментация печени и опухоли на КТ с [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) на датасете в формате nnU-Net (`Dataset001_LiverTumor`). **coarse_to_fine** — отдельный 2D U-Net по грубой маске nnU-Net. **multiview** — **независимая** надстройка: своя сеть `MultiviewUNet2d` (три фиксированных HU-окна + канал вероятности опухоли), в духе multi-view fusion для узлов на КТ (см. *Deep Multi-View Fusion Network for Lung Nodule Segmentation* и аналоги); чекпоинт обучается отдельно, не из весов coarse_to_fine. Код: `src/2d/multiview`, обучение `scripts/train_multiview.py`, инференс `scripts/infer_multiview.py`. **uncertainty** — ещё одна независимая вторая стадия: `UncertaintyUNet2d` (три HU-окна + вероятность + энтропия Bernoulli), `src/2d/uncertainty`, `scripts/train_uncertainty.py`, `scripts/infer_uncertainty.py`. **boundary_aware_coarse_to_fine** — отдельная вторая стадия: компактный 2D U-Net `BoundaryAwareTinyUNet2d` (по умолчанию **пять** входов: три HU-окна + вероятность опухоли stage-1 + нормированная энтропия; старые раны могли быть с тремя каналами — см. `meta.json`); на инференсе уточнение применяется **только в морфологическом кольце** вокруг грубой маски опухоли, с опционально адаптивной шириной кольца и порогом (`src/2d/boundary_aware_coarse_to_fine`, `scripts/train_boundary_aware_coarse_to_fine.py`, `scripts/infer_boundary_aware_coarse_to_fine.py`).

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
2. **Export** — слайсы с предсказанием stage-1 и GT для вторых стадий (`scripts/export.py`; общий экспорт для coarse_to_fine, multiview, uncertainty и boundary_aware_coarse_to_fine).
3. **coarse_to_fine** — вторая стадия (`scripts/train_coarse_to_fine.sh` или `train_coarse_to_fine.py` → `results_coarse_to_fine/...`).
4. **multiview** — вторая стадия (`scripts/train_multiview.sh` или `train_multiview.py` → `results_multiview/.../multiview/run_*`).
5. **uncertainty** — вторая стадия (`scripts/train_uncertainty.sh` или `train_uncertainty.py` → `results_uncertainty/.../uncertainty/run_*`).
6. **boundary_aware_coarse_to_fine** — вторая стадия (`scripts/train_boundary_aware_coarse_to_fine.sh` → `results_boundary_aware_coarse_to_fine/.../boundary_aware_coarse_to_fine/run_*`).
7. **Инференс coarse_to_fine** — полный объём: только nnU-Net или nnU-Net + coarse_to_fine (`scripts/infer_coarse_to_fine.py`).
8. **Инференс multiview** (опционально) — ROI + multi-window (`scripts/infer_multiview.py --multiview-dir ...`; e.g. `inference_comparison/multiview/`).
9. **Инференс uncertainty** (опционально) — `scripts/infer_uncertainty.py --uncertainty-dir ...` (по умолчанию `-o inference_comparison/uncertainty`).
10. **Инференс boundary_aware_coarse_to_fine** (опционально) — `scripts/infer_boundary_aware_coarse_to_fine.py --boundary-aware-dir ...` (по умолчанию вывод под `inference_comparison/boundary_aware_coarse_to_fine/`). Не смешивать каталоги чекпоинтов разных стадий.

**Согласованность stage-1:** по умолчанию и `export.py`, и `infer_coarse_to_fine.py` / `infer_multiview.py` / `infer_uncertainty.py` / `infer_boundary_aware_coarse_to_fine.py` используют **tile step 0.75** (как общий «лёгкий» бейзлайн). Для одного и того же шага явно задайте одинаковый `--tile-step-size` в нужных скриптах.

Эксперименты по полному инференсу обычно складывают в `inference_comparison/` с подпапками **`baseline`** (только nnU-Net), **`coarse_to_fine`**, **`multiview`**, **`uncertainty`**, **`boundary_aware_coarse_to_fine`**.

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

### 1.1. 3D nnU-Net и fine-tuning

Для 3D fullres экспериментов используется `3d_fullres` с планом
`nnUNetPlans_3d_midres125` и target spacing `1.25 1.0 1.0`. Базовый 3D
тренер:

```bash
bash scripts/train_3d.sh
# повтор без препроцесса:
bash scripts/train_3d.sh --skip-preprocess
```

Контрольный fine-tune 3D baseline с обычным nnU-Net Dice+CE loss:

```bash
bash scripts/train_3d_default_finetune.sh --skip-preprocess
```

Полнообъёмная nnU-Net-валидация пишется в `fold_*/validation/`. Оба скрипта
`train_3d_default_finetune.sh` и `train_3d_boundary_shape.sh` по умолчанию
передают **`--val_best`**, то есть сегментации считаются с **`checkpoint_best.pth`**
(а не с весами последней эпохи). Старое поведение (final):

`NNUNET_VALIDATION_WITH_BEST=0 bash scripts/train_3d_default_finetune.sh --skip-preprocess`.
Уже завершённый ран пересчитать только валидацию: `nnUNetv2_train` с тем же
dataset/config/fold/`-tr`/`-p` и `nnUNet_results` на каталог эксперимента,
флаги **`--val --val_best`** (или без `--val_best` для final). Каталог
`validation/` один и тот же — результаты перезаписываются.

Boundary/shape-aware fine-tune стартует из 3D baseline checkpoint. Референсные
раны лежат под `results_3d_boundary_shape_runs/` (локально, не в git):
`20260504_083549_saved_good_boundary/`, `20260509_131406_boundary_adaptive_large_tumor/`,
`20260509_160927_boundary_size_gated/`.

```bash
bash scripts/train_3d_boundary_shape.sh --skip-preprocess
```

По умолчанию он добавляет boundary-ring loss и hard-negative FP penalties
вне печени и внутри печени. Новый size-gated режим стартует от saved-good
Tversky-настроек и плавно отключает все custom loss terms, когда опухоль
занимает большую долю foreground (`tumor / (tumor + liver)`). Поэтому малые
кейсы остаются под BoundaryOverseg, а большие ближе к default Dice+CE fine-tune.
Tversky guard с `beta > alpha` остаётся включённым, но тоже проходит через этот
size gate.
Основные ручки:

```bash
NNUNET_BOUNDARY_OVERSEG_LR=1e-3
NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_FP_WEIGHT=4.0
NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_FP_WEIGHT=0.5
NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_VOLUME_GUARD_THRESHOLD=0.02
NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_VOLUME_GUARD_MIN_SCALE=0.10
NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_WEIGHT=0.05
NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_ALPHA=0.30
NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_BETA=0.70
NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_LARGE_TUMOR_THRESHOLD=0.02
NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_LARGE_TUMOR_MAX_THRESHOLD=0.10
NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_MIN_SCALE=1.0
NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_IGNORE_EXTRA_RADIUS=0
NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_WEIGHT=0.0
NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_THRESHOLD=0.05
NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_FRACTION=0.85
NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_THRESHOLD=0.04
NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_TEMPERATURE=0.015
NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_MIN_SCALE=0.0
```

Текущий size-gated preset сохранён в
`src/3d/boundary_shape/presets/size_gated_boundary_2026_05_09.env`.
Предыдущие наборы гиперпараметров сохранены в
`src/3d/boundary_shape/presets/adaptive_large_tumor_2026_05_09.env` и
`src/3d/boundary_shape/presets/tversky_guard_2026_05_04.env`; их можно
`source`-нуть перед запуском для воспроизведения.

Массовая перевалидация всех BoundaryOverseg ранов (одинаково `--val` и по умолчанию
`--val_best`):

```bash
bash scripts/revalidate_3d_boundary_shape_runs.sh
# только раны, в имени которых есть подстрока:
# RUN_FILTER=20260509 bash scripts/revalidate_3d_boundary_shape_runs.sh
# веса последней эпохи вместо best:
# VALIDATE_WITH_BEST=0 bash scripts/revalidate_3d_boundary_shape_runs.sh
```

Локальные 3D trainer-классы лежат в `src/3d/`; запуск через
`scripts/run_nnunet_with_local_3d_trainers.py` делает их видимыми для
nnU-Net и загружает fine-tune weights полностью, включая segmentation head.
Подробные параметры BoundaryOverseg описаны в `src/3d/boundary_shape/README.md`.

## 2. Export слайсов (coarse_to_fine, multiview, uncertainty)

Экспорт идёт из **одной** обученной nnU-Net (2D baseline по умолчанию или любая другая
папка с `fold_*` / `dataset.json` / `plans.json`, которую примет `nnUNetPredictor`).
Те же `.npz` используются для `train_coarse_to_fine`, `train_multiview`, `train_uncertainty`
и `train_boundary_aware_coarse_to_fine`. Переэкспорт нужен, если поменялись модель,
данные или параметры экспорта.

После обучения nnU-Net (нужен `fold_0` и `splits_final.json` в препроцессе для train/val):

```bash
python3 scripts/export.py --output-dir refinement_export/fold0
```

**3D fine-tune (BoundaryOverseg, `checkpoint_best`):** готовые каталоги под multiview:

```bash
bash scripts/export_3d_finetune_slices.sh size_gated      # → refinement_export/fold0_3d_size_gated/
bash scripts/export_3d_finetune_slices.sh adaptive_large  # → refinement_export/fold0_3d_adaptive_large/
bash scripts/export_3d_finetune_slices.sh all             # оба подряд
```

Дальше: `EXPORT_DIR=.../fold0_3d_size_gated bash scripts/train_multiview.sh` (или
`adaptive_large`). Вручную: `python3 scripts/export.py --output-dir ... --model-dir <путь_к_...__3d_fullres>`.

Каталог с `train/*.npz` и `val/*.npz` может быть любым; классический экспорт лежит в
**`refinement_export/fold0/`** (метаданные: `export_meta.json`). Обёртки `train_*.sh`
по умолчанию берут **`--export-dir` оттуда же**.

Опции `export.py`: `--model-dir`, `--fold`, `--checkpoint`, `--tile-step-size` (по умолчанию `0.75`), `--all-slices`, `--device`.

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

Для multiview — **отдельный** каталог ранов: **`results_multiview/`** (не смешивать с `results_coarse_to_fine`). Для uncertainty — **`results_uncertainty/`**. Для boundary_aware_coarse_to_fine — **`results_boundary_aware_coarse_to_fine/`**. См. разделы 4–5 (обучение и инференс multiview, uncertainty и boundary_aware).

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

### 4.2. Обучение boundary_aware_coarse_to_fine (BoundaryAwareTinyUNet2d)

Тот же `--export-dir`, что в разделе 2. Обучение на ROI **GT ∪ coarse** (как coarse_to_fine), resize в `--crop-size`. **Пять входов** (по умолчанию): три HU-окна (`boundary_aware_coarse_to_fine.config`, по смыслу согласованы с multiview/uncertainty) + `coarse_tumor_prob` + нормированная энтропия Bernoulli по этой вероятности. Опции: `--hu-windows` (шесть чисел W/L × 3), `--lambda-boundary` (вес BCE+Dice на **кольце границы** вокруг грубой маски), `--focal-gamma` (фокальный BCE), `--bce-weight`, `--boundary-dilate-iters` / `--boundary-erode-iters`, см. `python3 scripts/train_boundary_aware_coarse_to_fine.py -h`.

```bash
bash scripts/train_boundary_aware_coarse_to_fine.sh
# эквивалентно: python3 scripts/train_boundary_aware_coarse_to_fine.py --export-dir refinement_export/fold0
```

Чекпоинты: `results_boundary_aware_coarse_to_fine/<dataset>/fold_<n>/boundary_aware_coarse_to_fine/run_<timestamp>/` (`checkpoint_best.pth`, после каждой эпохи — `checkpoint_last.pth`, `meta.json` с `hu_windows`, `in_channels`, `focal_gamma`, параметрами кольца и адаптивного инференса). **Доучивание:** `--resume` на каталог рана или на `.pth` (предпочтительно `checkpoint_last.pth`), `--epochs` — сколько **дополнительных** эпох; если `--out-dir` не задан, берётся каталог рана из чекпоинта. Старые раны с `in_channels: 3` в `meta.json` совместимы с инференсом (один нормализованный CT + prob + entropy), но **не** с новой сетью на 5 каналов без переобучения.

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

### 5.4. boundary_aware_coarse_to_fine (`infer_boundary_aware_coarse_to_fine.py`)

Тот же стек stage-1, что и у `infer_coarse_to_fine.py`. Две стадии: `--boundary-aware-dir` — каталог рана с `checkpoint_best.pth` и `meta.json` (HU-окна и `in_channels` читаются оттуда). Baseline только nnU-Net: `--stage1-only`. Пример двух стадий:

```bash
python3 scripts/infer_boundary_aware_coarse_to_fine.py \
  -i <imagesTr_или_аналог> \
  --boundary-aware-dir <results_boundary_aware_coarse_to_fine/.../boundary_aware_coarse_to_fine/run_<timestamp>>
```

Справка: `python3 scripts/infer_boundary_aware_coarse_to_fine.py -h`.

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
├── src/2d/               # существующие 2D/second-stage обучения
├── src/2d/coarse_to_fine/
├── src/2d/multiview/
├── src/2d/uncertainty/
├── src/2d/boundary_aware_coarse_to_fine/
├── src/3d/               # локальные 3D nnU-Net trainers и fine-tuning losses
├── data/
├── nnUNet_raw/
├── nnUNet_preprocessed/
├── nnUNet_results/
├── results_3d_default_finetune/  # 3D fine-tune контроль с default loss (не в git)
├── results_3d_boundary_shape_runs/  # 3D BoundaryOverseg: референсные раны saved_good / adaptive_large / size_gated (не в git)
├── refinement_export/        # экспорт слайсов (e.g. fold0/train, fold0/val, export_meta.json)
├── results_coarse_to_fine/   # только coarse_to_fine (чекпойнты и логи)
├── results_multiview/        # только multiview / MultiviewUNet2d (не в git)
├── results_uncertainty/      # только uncertainty / UncertaintyUNet2d (не в git)
├── results_boundary_aware_coarse_to_fine/  # boundary_aware вторая стадия (не в git)
└── scripts/
    ├── train.sh                  # стадия 1: nnU-Net план / препроцесс / train
    ├── train_3d.sh               # 3D fullres baseline
    ├── train_3d_default_finetune.sh  # 3D fine-tune с default Dice+CE
    ├── train_3d_boundary_shape.sh    # 3D fine-tune с BoundaryOverseg loss
    ├── revalidate_3d_boundary_shape_runs.sh  # --val [--val_best] для всех ранов в results_3d_boundary_shape_runs/
    ├── run_nnunet_with_local_3d_trainers.py  # запуск локальных 3D trainer-классов
    ├── train_coarse_to_fine.sh   # обёртка: PYTHONPATH, EXPORT_DIR → train_coarse_to_fine.py
    ├── train_multiview.sh        # то же для train_multiview.py
    ├── train_uncertainty.sh      # то же для train_uncertainty.py
    ├── train_boundary_aware_coarse_to_fine.sh  # → train_boundary_aware_coarse_to_fine.py
    ├── export.py                 # слайсы + stage-1 (общий экспорт для вторых стадий)
    ├── train_coarse_to_fine.py   # обучение coarse_to_fine
    ├── train_multiview.py        # обучение MultiviewUNet2d
    ├── train_uncertainty.py      # обучение UncertaintyUNet2d
    ├── train_boundary_aware_coarse_to_fine.py  # обучение BoundaryAwareTinyUNet2d
    ├── infer_coarse_to_fine.py   # полный объём: stage 1 или 1+coarse_to_fine
    ├── infer_multiview.py        # nnU-Net + MultiviewUNet2d
    ├── infer_uncertainty.py      # nnU-Net + UncertaintyUNet2d
    ├── infer_boundary_aware_coarse_to_fine.py  # nnU-Net + уточнение в кольце границы
    └── evaluate_segmentations.py # Dice/IoU опухоли vs labelsTr
```

## Лицензия

См. `LICENSE`.
