from __future__ import annotations

import gc
import multiprocessing
import os
import warnings
from pathlib import Path
from time import sleep

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from torch import distributed as dist

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
        # Lighter default than stock nnU-Net (250/50): 4ch batches are heavier on GPU/RAM.
        self.num_iterations_per_epoch = self.mw_config.iterations_per_epoch
        self.num_val_iterations_per_epoch = self.mw_config.val_iterations_per_epoch

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized."
            )
        self._set_batch_size_and_oversample()
        if not self.is_ddp:
            cap = self.mw_config.max_batch_size
            if cap is not None and self.batch_size > cap:
                self.print_to_log_file(
                    f"MultiWindowRefine: batch_size {self.batch_size} -> {cap} (RAM; "
                    f"NNUNET_MW_MAX_BATCH_SIZE=0 to use full plan batch size)"
                )
                self.batch_size = cap
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
                "(see scripts/3d/cache_tumor_prob_for_multiwindow.py)."
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

    def _mw_build_data4_torch(self, data_1ch_zyx: np.ndarray, case_id: str) -> torch.Tensor:
        """Stack tumour prob (cached) + Lim HU windows; same convention as training / infer script."""
        from multiwindow.windows import lim_three_windows_from_norm, load_fingerprint_stats

        prob_path = join(self._mw_prob_dir(), f"{case_id}.npz")
        if not isfile(prob_path):
            raise FileNotFoundError(
                f"MultiWindow validation: missing prob cache for {case_id}: {prob_path}"
            )
        stats = load_fingerprint_stats(self._mw_fingerprint_json())
        data_1ch_zyx = np.asarray(data_1ch_zyx, dtype=np.float32)
        norm = np.asarray(data_1ch_zyx[0], dtype=np.float32)
        pr = np.load(prob_path, mmap_mode="r")["prob"]
        pr = np.asarray(pr, dtype=np.float32)
        if pr.ndim == 3:
            pr = pr[np.newaxis, ...]
        if pr.shape != data_1ch_zyx.shape:
            raise ValueError(
                f"MultiWindow validation: prob shape {pr.shape} != data {data_1ch_zyx.shape} ({case_id})"
            )
        hw = lim_three_windows_from_norm(norm, stats)
        stack = np.concatenate([pr, hw], axis=0).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.from_numpy(stack)

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Full-volume export under ``fold_*/validation`` using 4-channel input (matches training)."""
        if self.is_cascaded:
            raise RuntimeError(
                "nnUNetTrainer_150_MultiWindowRefine: cascaded mode is not supported for full-volume validation."
            )

        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file(
                "WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                "encounter crashes in validation then this is because torch.compile forgets "
                "to trigger a recompilation of the model with deep supervision disabled. "
                "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                "validation with --val (exactly the same as before) and then it will work. "
                "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                "forward pass (where compile is triggered) already has deep supervision disabled. "
                "This is exactly what we need in perform_actual_validation"
            )

        val_on_gpu = self.device.type == "cuda" and os.environ.get(
            "NNUNET_MW_VAL_PERFORM_ON_DEVICE", "1"
        ).strip().lower() not in ("0", "false", "no", "n")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=val_on_gpu,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, "validation")
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank :: dist.get_world_size()]

            dataset_val = self.dataset_class(
                self.preprocessed_dataset_folder,
                val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            )

            next_stages = self.configuration_manager.next_stage_names
            if next_stages is not None:
                _ = [
                    maybe_mkdir_p(join(self.output_folder_base, "predicted_next_stage", n))
                    for n in next_stages
                ]

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(
                    segmentation_export_pool, worker_list, results, allowed_num_queued=2
                )
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        segmentation_export_pool, worker_list, results, allowed_num_queued=2
                    )

                self.print_to_log_file(f"predicting {k}")
                data, _, _seg_prev, properties = dataset_val.load_case(k)
                data = data[:]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data4 = self._mw_build_data4_torch(data, k)

                self.print_to_log_file(f"{k}, shape {data4.shape}, rank {self.local_rank}")
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data4)
                prediction = prediction.cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits,
                        (
                            (
                                prediction,
                                properties,
                                self.configuration_manager,
                                self.plans_manager,
                                self.dataset_json,
                                output_filename_truncated,
                                save_probabilities,
                            ),
                        ),
                    )
                )

                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(
                            nnUNet_preprocessed,
                            self.plans_manager.dataset_name,
                            next_stage_config_manager.data_identifier,
                        )
                        dataset_class_next = infer_dataset_class(expected_preprocessed_folder)

                        try:
                            tmp = dataset_class_next(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k}: "
                                "preprocessed file missing; run preprocessing for this configuration first."
                            )
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, "predicted_next_stage", n)
                        output_file_truncated = join(output_folder, k)

                        results.append(
                            segmentation_export_pool.starmap_async(
                                resample_and_save,
                                (
                                    (
                                        prediction,
                                        target_shape,
                                        output_file_truncated,
                                        self.plans_manager,
                                        self.configuration_manager,
                                        properties,
                                        self.dataset_json,
                                        default_num_processes,
                                        dataset_class_next,
                                    ),
                                ),
                            )
                        )

                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

                del data, data4, prediction
                gc.collect()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
                validation_output_folder,
                join(validation_output_folder, "summary.json"),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else self.label_manager.foreground_labels,
                self.label_manager.ignore_label,
                chill=True,
                num_processes=default_num_processes * dist.get_world_size()
                if self.is_ddp
                else default_num_processes,
            )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file(
                "Mean Validation Dice: ",
                (metrics["foreground_mean"]["Dice"]),
                also_print_to_console=True,
            )

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

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
        nc_tr = max(1, self.mw_config.da_num_cached_train)
        nc_val = max(1, self.mw_config.da_num_cached_val)
        pin = self.device.type == "cuda" and self.mw_config.pin_memory
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=nc_tr,
                seeds=None,
                pin_memory=pin,
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=nc_val,
                seeds=None,
                pin_memory=pin,
                wait_time=0.002,
            )
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "MultiWindowRefine: 4-channel input (tumor_prob + Lim HU windows [-1000,1000], [0,1000], [400,1000]), "
            f"epochs={self.mw_config.num_epochs}, lr={self.mw_config.initial_lr}, batch_size={self.batch_size}, "
            f"iter/epoch={self.num_iterations_per_epoch}, val_iter/epoch={self.num_val_iterations_per_epoch}, "
            f"da_num_cached train/val={self.mw_config.da_num_cached_train}/{self.mw_config.da_num_cached_val}, "
            f"pin_memory={self.mw_config.pin_memory}, "
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
