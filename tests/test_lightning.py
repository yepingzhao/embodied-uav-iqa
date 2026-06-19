import json
import sys
from pathlib import Path

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uav_iqa.lightning_model import UAVQALightningModule


class TestLightningModule:
    def test_init(self):
        model = UAVQALightningModule()
        assert model.model is not None
        assert model.model.use_fab is True
        assert model.model.use_cbam is True
        assert model.model.use_task_conditioning is True

    def test_no_fab(self):
        model = UAVQALightningModule(use_fab=False)
        assert model.model.use_fab is False

    def test_configure_optimizers(self):
        model = UAVQALightningModule()
        opt_cfg = model.configure_optimizers()
        assert "optimizer" in opt_cfg
        assert "lr_scheduler" in opt_cfg
        assert opt_cfg["lr_scheduler"]["interval"] == "epoch"

    def test_no_warmup_configure_optimizers(self):
        model = UAVQALightningModule(warmup_epochs=0)
        opt_cfg = model.configure_optimizers()
        assert "optimizer" in opt_cfg

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_training_step(self):
        model = UAVQALightningModule(lambda_rank=0.0, lambda_cross_task=0.0)
        model.curriculum_stage = "vla"

        batch = {
            "image": torch.randn(4, 3, 256, 256),
            "task_id": torch.randint(0, 4, (4,)),
            "vla_score": torch.rand(4),
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
        for split in ["train", "val", "test"]:
            split_dir = data_root / split
            split_dir.mkdir(parents=True)
            manifest = []
            for i in range(20):
                manifest.append({
                    "path": f"{split}/img_{i:03d}.png",
                    "task": "tracking",
                    "vlm_score": 0.5,
                    "vla_score": 0.6,
                    "execution_score": 0.7,
                    "distortion": "blur_L02",
                    "intensity_level": 0.2,
                    "ref_id": i,
                })
            with open(split_dir / "manifest.json", "w") as f:
                json.dump(manifest, f)
        return str(data_root)

    def test_data_module_setup(self, mock_data_dir):
        from uav_iqa.lightning_data import UAVIQDataModule

        dm = UAVIQDataModule(
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


class TestOverfitIntegration:
    @pytest.mark.slow
    def test_overfit(self):
        model = UAVQALightningModule(
            lambda_rank=0.0,
            lambda_cross_task=0.0,
            warmup_epochs=2,
            total_epochs=30,
        )

        n = 100
        x = torch.randn(n, 3, 256, 256)
        task_ids = torch.randint(0, 4, (n,))
        y = torch.rand(n)

        class OverfitDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(x)

            def __getitem__(self, idx):
                return {
                    "image": x[idx],
                    "task_id": task_ids[idx],
                    "score": y[idx],
                    "vlm_score": y[idx],
                    "vla_score": y[idx],
                    "execution_score": y[idx],
                    "distortion": "overfit",
                }

        dataset = OverfitDataset()
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=30,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        class ManualDataModule(L.LightningDataModule):
            def train_dataloader(self):
                return loader

        trainer.fit(model, datamodule=ManualDataModule())

        final_loss = float(trainer.callback_metrics.get("train/loss_epoch", 999))
        assert final_loss < 0.01, f"Overfit failed: loss={final_loss:.6f}"
