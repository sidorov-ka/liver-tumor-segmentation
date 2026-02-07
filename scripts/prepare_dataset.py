import json
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADED_PATH_FILE = PROJECT_ROOT / "data" / "downloaded_path.txt"

OUT = PROJECT_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor"
IMAGES_TR = OUT / "imagesTr"
LABELS_TR = OUT / "labelsTr"


def read_src_path() -> Path:
    p = DOWNLOADED_PATH_FILE.read_text(encoding="utf-8").strip()
    src = Path(p)
    if not src.exists():
        raise FileNotFoundError(f"Downloaded dataset path not found: {src}")
    return src


def resolve_data_root(src: Path) -> Path:
    """Return the directory that contains train_CT/ and train_mask/ (may be inside LiTS(train_test)/)."""
    if (src / "train_CT").exists() and (src / "train_mask").exists():
        return src
    sub = src / "LiTS(train_test)"
    if sub.exists() and (sub / "train_CT").exists() and (sub / "train_mask").exists():
        return sub
    return src


def extract_idx(path: Path) -> int:
    m = re.search(r"(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse index from filename: {path.name}")
    return int(m.group(1))


def collect_pairs(src: Path):
    volumes = list(src.glob("train_CT/volume-*.nii")) + list(src.glob("train_CT/volume-*.nii.gz"))
    segs = list(src.glob("train_mask/segmentation-*.nii")) + list(src.glob("train_mask/segmentation-*.nii.gz"))
    if not volumes or not segs:
        volumes = list(src.glob("volume_pt*/volume-*.nii")) + list(src.glob("volume_pt*/volume-*.nii.gz"))
        segs = list(src.glob("segmentations/segmentation-*.nii")) + list(src.glob("segmentations/segmentation-*.nii.gz"))

    vol_map = {extract_idx(v): v for v in volumes}
    seg_map = {extract_idx(s): s for s in segs}

    common = sorted(set(vol_map) & set(seg_map))
    pairs = [(vol_map[i], seg_map[i], i) for i in common]

    return pairs, len(vol_map), len(seg_map), len(common)


def prepare_dirs():
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs):
    for new_id, (vol, seg, orig_id) in enumerate(pairs):
        case_id = f"case_{new_id:04d}"
        img_dst = IMAGES_TR / f"{case_id}_0000.nii.gz"
        lbl_dst = LABELS_TR / f"{case_id}.nii.gz"
        shutil.copy(vol, img_dst)
        shutil.copy(seg, lbl_dst)


def write_dataset_json(num_cases: int):
    dataset = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1, "tumor": 2},
        "numTraining": num_cases,
        "file_ending": ".nii.gz"
    }
    with open(OUT / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


def main():
    src = read_src_path()
    src = resolve_data_root(src)
    pairs, n_vol, n_seg, n_common = collect_pairs(src)

    print(f"SRC: {src}")
    print(f"Found volumes: {n_vol}, segmentations: {n_seg}, paired: {n_common}")

    if n_common == 0:
        raise RuntimeError("No paired volume/segmentation files found.")

    prepare_dirs()
    copy_pairs(pairs)
    write_dataset_json(n_common)

    print(f"Done. nnU-Net dataset at: {OUT}")


if __name__ == "__main__":
    main()
