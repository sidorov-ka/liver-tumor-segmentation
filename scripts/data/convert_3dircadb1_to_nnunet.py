#!/usr/bin/env python3
"""Convert official 3D-IRCADb-01 zip into nnU-Net test layout for Dataset001.

Writes:
  nnUNet_raw/Dataset001_LiverTumor/imagesTs/ircad_XX_0000.nii.gz
  nnUNet_raw/Dataset001_LiverTumor/labelsTs/ircad_XX.nii.gz

Labels match LiTS / Dataset001: background=0, liver=1, tumor=2.
Tumor = union of MASKS_DICOM/livertumor* (liver cysts are not tumor).
"""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = Path("/mnt/c/Users/kasid/Downloads/3Dircadb1.zip")
DEFAULT_OUT = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor"


def _case_dirs(outer: zipfile.ZipFile) -> list[tuple[int, str]]:
    cases: dict[int, str] = {}
    for name in outer.namelist():
        m = re.match(r"^(3Dircadb1/3Dircadb1\.(\d+)/)", name)
        if not m:
            continue
        cases[int(m.group(2))] = m.group(1)
    return sorted(cases.items())


def _extract_nested(outer: zipfile.ZipFile, member: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / Path(member).name
    zip_path.write_bytes(outer.read(member))
    extract_dir = dest / (zip_path.stem + "_dir")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    # Prefer the conventional inner folder if present.
    for child in extract_dir.iterdir():
        if child.is_dir():
            return child
    return extract_dir


def _read_dicom_series(folder: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise RuntimeError(f"No DICOM series in {folder}")
    files = reader.GetGDCMSeriesFileNames(str(folder), series_ids[0])
    reader.SetFileNames(files)
    return reader.Execute()


def _tumor_mask_dirs(masks_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for path in sorted(masks_root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if name == "livertumor" or name.startswith("livertumor"):
            dirs.append(path)
    return dirs


def _to_label_image(
    patient: sitk.Image,
    liver: sitk.Image | None,
    tumor_images: list[sitk.Image],
) -> sitk.Image:
    patient_arr = sitk.GetArrayFromImage(patient)
    labels = np.zeros(patient_arr.shape, dtype=np.uint8)
    if liver is not None:
        liver_arr = sitk.GetArrayFromImage(liver)
        if liver_arr.shape != labels.shape:
            raise RuntimeError(
                f"Liver mask shape {liver_arr.shape} != CT {labels.shape}"
            )
        labels[liver_arr > 0] = 1
    for tumor in tumor_images:
        tumor_arr = sitk.GetArrayFromImage(tumor)
        if tumor_arr.shape != labels.shape:
            raise RuntimeError(
                f"Tumor mask shape {tumor_arr.shape} != CT {labels.shape}"
            )
        labels[tumor_arr > 0] = 2
    out = sitk.GetImageFromArray(labels)
    out.CopyInformation(patient)
    return out


def convert_case(
    outer: zipfile.ZipFile,
    case_id: int,
    case_prefix: str,
    images_ts: Path,
    labels_ts: Path,
) -> dict[str, int | str]:
    case_name = f"ircad_{case_id:02d}"
    with tempfile.TemporaryDirectory(prefix=f"{case_name}_") as tmp:
        tmp_path = Path(tmp)
        patient_dir = _extract_nested(
            outer,
            f"{case_prefix}PATIENT_DICOM.zip",
            tmp_path / "patient",
        )
        masks_dir = _extract_nested(
            outer,
            f"{case_prefix}MASKS_DICOM.zip",
            tmp_path / "masks",
        )

        patient = _read_dicom_series(patient_dir)
        liver_path = masks_dir / "liver"
        liver = _read_dicom_series(liver_path) if liver_path.is_dir() else None
        tumor_dirs = _tumor_mask_dirs(masks_dir)
        tumors = [_read_dicom_series(path) for path in tumor_dirs]
        label_img = _to_label_image(patient, liver, tumors)

        img_out = images_ts / f"{case_name}_0000.nii.gz"
        lab_out = labels_ts / f"{case_name}.nii.gz"
        sitk.WriteImage(patient, str(img_out), useCompression=True)
        sitk.WriteImage(label_img, str(lab_out), useCompression=True)

        lab = sitk.GetArrayFromImage(label_img)
        return {
            "case": case_name,
            "shape": "x".join(str(x) for x in lab.shape),
            "liver_voxels": int((lab == 1).sum()),
            "tumor_voxels": int((lab == 2).sum()),
            "n_tumor_masks": len(tumor_dirs),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=DEFAULT_ZIP,
        help="Path to 3Dircadb1.zip",
    )
    parser.add_argument(
        "--out-dataset",
        type=Path,
        default=DEFAULT_OUT,
        help="Dataset001_LiverTumor root",
    )
    args = parser.parse_args()

    if not args.zip.is_file():
        raise SystemExit(f"Zip not found: {args.zip}")

    images_ts = args.out_dataset / "imagesTs"
    labels_ts = args.out_dataset / "labelsTs"
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as outer:
        cases = _case_dirs(outer)
        if len(cases) != 20:
            print(f"Warning: expected 20 cases, found {len(cases)}")
        for case_id, prefix in cases:
            print(f"convert {case_id:02d} ...", flush=True)
            info = convert_case(outer, case_id, prefix, images_ts, labels_ts)
            print(
                f"  {info['case']}: shape={info['shape']} "
                f"liver={info['liver_voxels']} tumor={info['tumor_voxels']} "
                f"tumor_masks={info['n_tumor_masks']}",
                flush=True,
            )

    n_img = len(list(images_ts.glob("ircad_*_0000.nii.gz")))
    n_lab = len(list(labels_ts.glob("ircad_*.nii.gz")))
    print(f"DONE imagesTs={n_img} labelsTs={n_lab} -> {args.out_dataset}")


if __name__ == "__main__":
    main()
