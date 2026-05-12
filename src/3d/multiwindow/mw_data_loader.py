from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from multiwindow.windows import lim_three_windows_from_norm, load_fingerprint_stats


class MultiWindowRefinementDataLoader(nnUNetDataLoader):
    """Build 4-channel patches: [tumor_prob, Lim window x3] + same spatial crop as nnU-Net."""

    def __init__(
        self,
        data: nnUNetBaseDataset,
        batch_size: int,
        patch_size,
        final_patch_size,
        label_manager: LabelManager,
        oversample_foreground_percent: float = 0.0,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling: bool = False,
        transforms=None,
        *,
        prob_dir: str,
        fingerprint_json: str,
    ):
        self.prob_dir = str(prob_dir)
        if not self.prob_dir or not os.path.isdir(self.prob_dir):
            raise FileNotFoundError(
                f"MultiWindowRefinementDataLoader: prob_dir missing or not a directory: {self.prob_dir!r}. "
                "Run scripts/cache_tumor_prob_for_multiwindow.py first."
            )
        if not fingerprint_json or not os.path.isfile(fingerprint_json):
            raise FileNotFoundError(
                f"MultiWindowRefinementDataLoader: fingerprint_json not found: {fingerprint_json!r}"
            )
        self.fp_stats = load_fingerprint_stats(fingerprint_json)
        super().__init__(
            data,
            batch_size,
            patch_size,
            final_patch_size,
            label_manager,
            oversample_foreground_percent,
            sampling_probabilities,
            pad_sides,
            probabilistic_oversampling,
            transforms,
        )

    def determine_shapes(self):
        _, seg, seg_prev, _ = self._data.load_case(self._data.identifiers[0])
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        data_shape = (self.batch_size, 4, *self.patch_size)
        seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = np.asarray(data).shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
            bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

            norm_crop = crop_and_pad_nd(np.asarray(data, dtype=np.float32), bbox, 0)
            prob_path = join(self.prob_dir, f"{i}.npz")
            if not os.path.isfile(prob_path):
                raise FileNotFoundError(f"Missing tumor prob cache for case {i}: {prob_path}")
            prob_raw = np.load(prob_path, mmap_mode="r")
            prob = np.asarray(prob_raw["prob"], dtype=np.float32)
            if prob.ndim == 3:
                prob = prob[np.newaxis, ...]
            if prob.shape != norm_crop.shape:
                raise ValueError(
                    f"Prob shape {prob.shape} != image shape {norm_crop.shape} for case {i}. "
                    "Re-run cache script with the same preprocessed configuration."
                )
            prob_crop = crop_and_pad_nd(prob, bbox, 0.0)
            hw = lim_three_windows_from_norm(np.asarray(norm_crop[0], dtype=np.float32), self.fp_stats)
            merged = np.concatenate([prob_crop, hw], axis=0).astype(np.float32)
            data_all[j] = merged

            seg_cropped = crop_and_pad_nd(np.asarray(seg), bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack(
                    (seg_cropped, crop_and_pad_nd(np.asarray(seg_prev), bbox, -1)[None])
                )
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{"image": data_all[b], "segmentation": seg_all[b]})
                        images.append(tmp["image"])
                        segs.append(tmp["segmentation"])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {"data": data_all, "target": seg_all, "keys": selected_keys}

        return {"data": data_all, "target": seg_all, "keys": selected_keys}
