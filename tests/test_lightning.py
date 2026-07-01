import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uav_iqa.lightning_module import UAVIQALightningModule


class TestLightningModule:
    def test_init(self):
        model = UAVIQALightningModule()
        assert model.model is not None
        assert model.model.use_fab is True
        assert model.model.use_cbam is True
        assert model.model.use_task_conditioning is True

    def test_no_fab(self):
        model = UAVIQALightningModule(use_fab=False)
        assert model.model.use_fab is False

    def test_configure_optimizers(self):
        model = UAVIQALightningModule()
        opt_cfg = model.configure_optimizers()
        assert "optimizer" in opt_cfg
        assert "lr_scheduler" in opt_cfg
        assert opt_cfg["lr_scheduler"]["interval"] == "epoch"

    def test_no_warmup_configure_optimizers(self):
        model = UAVIQALightningModule(warmup_epochs=0)
        opt_cfg = model.configure_optimizers()
        assert "optimizer" in opt_cfg

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_training_step(self):
        model = UAVIQALightningModule(lambda_rank=0.0, lambda_cross_task=0.0)

        batch = {
            "images": torch.randn(4, 3, 256, 256),
            "task_id": torch.randint(0, 14, (4,)),
            "cognitive_score": torch.rand(4),
            "distortion": ["blur_L02", "blur_L02", "noise_L04", "noise_L04"],
        }

        loss = model.training_step(batch, 0)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


class TestLightningDataModule:
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        data_root = tmp_path / "data"
        data_root.mkdir(parents=True)

        def _write_entries(split):
            split_dir = data_root / split
            split_dir.mkdir(parents=True)
            entries = []
            for i in range(20):
                entries.append({
                    "sample_id": f"Sim3__scene_{i:03d}__blur_L02",
                    "subtask_type": "1.1",
                    "subtask_name": "scene_description",
                    "cognitive_score": 0.5,
                    "uav_paths": {"UAV1": f"refs/uav_{i:03d}.png"},
                    "distorted_uav_paths": {"UAV1": f"split/distorted_{i:03d}.png"},
                    "distortion_info": {
                        "type": "blur",
                        "category": "blur",
                        "intensity": 0.2,
                        "level": 2,
                        "seed": 42,
                    },
                })
            with open(split_dir / f"Sim3_VQA_{split}.json", "w") as f:
                json.dump(entries, f)

        _write_entries("train")
        _write_entries("test")
        return str(data_root)

    def test_data_module_setup(self, mock_data_dir):
        from uav_iqa.data_module import UAVIQADataModule

        dm = UAVIQADataModule(
            data_root=mock_data_dir,
            batch_size=4,
            num_workers=0,
        )
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        assert len(dm.train_dataset) == 20
        assert len(dm.val_dataset) == 20
