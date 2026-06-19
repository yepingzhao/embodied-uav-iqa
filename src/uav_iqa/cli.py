import json
from pathlib import Path

import lightning as L
from lightning.pytorch.cli import LightningCLI

from .callbacks import CurriculumStageCallback, MetricsHistoryCallback
from .utils import count_parameters


class UAVIQACLI(LightningCLI):
    """Custom LightningCLI with multi-seed support, post-fit test, and result saving."""

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--output_dir", type=str, default="outputs/training")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dry_run", type=bool, default=False)

    def before_fit(self) -> None:
        seed = getattr(self.config, "seed", 42)
        output_dir = Path(getattr(self.config, "output_dir", "outputs/training"))
        run_dir = output_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        L.seed_everything(seed, workers=True)

        max_epochs = self.trainer.max_epochs or 50
        curriculum_cb = CurriculumStageCallback(
            vlm_epochs=20,
            vla_epochs=20,
            execution_epochs=max(0, max_epochs - 40),
        )
        history_cb = MetricsHistoryCallback(output_dir=str(run_dir))
        checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(run_dir),
            filename="best_model",
            monitor="val/srcc",
            mode="max",
            save_top_k=1,
            save_last=False,
        )

        self.trainer.callbacks.append(curriculum_cb)
        self.trainer.callbacks.append(history_cb)
        self.trainer.callbacks.append(checkpoint_cb)

        self._seed = seed
        self._run_dir = run_dir
        self._history_cb = history_cb

        n_params = count_parameters(self.model.model)[0]
        print(f"Seed {seed} | Model params: {n_params:,}")

    def after_fit(self) -> None:
        checkpoint_cb = None
        for cb in self.trainer.callbacks:
            if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint):
                checkpoint_cb = cb
                break

        best_path = checkpoint_cb.best_model_path if checkpoint_cb else None
        if best_path and Path(best_path).exists():
            print(f"Best checkpoint: {best_path}")
            self.trainer.test(
                self.model, datamodule=self.datamodule, ckpt_path=best_path
            )
        else:
            self.trainer.test(self.model, datamodule=self.datamodule)

        test_results = self.model.get_test_results()
        best_val_srcc = self._history_cb.best_val_srcc
        test_metrics = test_results.get(
            "test_metrics", {"srcc": 0.0, "plcc": 0.0, "rmse": 0.0}
        )

        print(
            f"\n  Test: SRCC={test_metrics.get('srcc', 0):.4f} | "
            f"PLCC={test_metrics.get('plcc', 0):.4f} | "
            f"RMSE={test_metrics.get('rmse', 0):.4f}"
        )

        per_task = test_results.get("per_task", {})
        if per_task:
            print(
                "  Per-task SRCC:",
                {k: f"{v['srcc']:.4f}" for k, v in per_task.items()},
            )

        n_params = count_parameters(self.model.model)[0]
        result = {
            "seed": self._seed,
            "n_params": n_params,
            "best_val_srcc": best_val_srcc,
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "per_task": per_task,
        }
        with open(self._run_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)
