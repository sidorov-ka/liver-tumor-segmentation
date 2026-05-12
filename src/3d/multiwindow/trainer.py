from __future__ import annotations

import os
from pathlib import Path

import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join

from nnunet_trainer_150_compat import nnUNetTrainer_150

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from multiwindow.config import MultiWindowRefineConfig
from multiwindow.losses import MultiWindowRefinementLoss
from multiwindow.mw_data_loader import MultiWindowRefinementDataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]


class nnUNetTrainer_150_MultiWindowRefine_50epochs(nnUNetTrainer_150):
    """3D refinement: 4 inputs (coarse tumour prob + Lim HU windows) for large under-segmented tumours."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.mw_config = MultiWindowRefineConfig.from_env(str(REPO_ROOT))
        self.num_epochs = self.mw_config.num_epochs
        self.initial_lr = self.mw_config.initial_lr

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized."
            )
        self._set_batch_size_and_oversample()
        self.num_input_channels = 4

        self.network = nnUNetTrainer.build_network_architecture(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.label_manager.num_segmentation_heads,
            self.enable_deep_supervision,
        ).to(self.device)
        if self._do_i_compile():
            self.print_to_log_file("Using torch.compile...")
            self.network = torch.compile(self.network)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = torch.nn.parallel.DistributedDataParallel(
                self.network, device_ids=[self.local_rank]
            )

        self.loss = self._build_loss()
        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True

    def _build_loss(self):
        base_loss = super()._build_loss()
        c = self.mw_config
        return MultiWindowRefinementLoss(
            base_loss=base_loss,
            tumor_label=c.tumor_label,
            tversky_weight=c.tversky_weight,
            tversky_alpha=c.tversky_alpha,
            tversky_beta=c.tversky_beta,
        )

    def _mw_prob_dir(self) -> str:
        d = os.environ.get("NNUNET_MW_PROB_DIR", "").strip()
        if not d:
            raise RuntimeError(
                "NNUNET_MW_PROB_DIR must point to per-case tumour probability caches "
                "(see scripts/cache_tumor_prob_for_multiwindow.py)."
            )
        return os.path.abspath(d)

    def _mw_fingerprint_json(self) -> str:
        fp = (self.mw_config.fingerprint_json or "").strip()
        if fp:
            return os.path.abspath(fp)
        pre = os.environ.get("nnUNet_preprocessed", str(REPO_ROOT / "nnUNet_preprocessed"))
        return os.path.abspath(
            join(pre, "Dataset001_LiverTumor", "dataset_fingerprint.json")
        )

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        prob_dir = self._mw_prob_dir()
        fp_json = self._mw_fingerprint_json()

        dl_tr = MultiWindowRefinementDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            prob_dir=prob_dir,
            fingerprint_json=fp_json,
        )
        dl_val = MultiWindowRefinementDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            prob_dir=prob_dir,
            fingerprint_json=fp_json,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "MultiWindowRefine: 4-channel input (tumor_prob + Lim HU windows [-1000,1000], [0,1000], [400,1000]), "
            f"epochs={self.mw_config.num_epochs}, lr={self.mw_config.initial_lr}, "
            f"Tversky w={self.mw_config.tversky_weight} (alpha={self.mw_config.tversky_alpha}, "
            f"beta={self.mw_config.tversky_beta}), prob_dir={self._mw_prob_dir()}"
        )

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        comp = getattr(self.loss, "last_components", None)
        if comp:
            self.print_to_log_file(
                "MultiWindowRefine loss: "
                + ", ".join(f"{k}={v:.4f}" for k, v in comp.items())
            )
