import numpy as np
from scipy.stats import pearsonr, spearmanr

from uav_iqa.domain import SUBTASK_NAME_LIST as TASK_NAMES


def compute_srcc(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman Rank Correlation Coefficient."""
    if len(pred) < 3:
        return 0.0
    coef, _ = spearmanr(pred, target)
    return float(coef)


def compute_plcc(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson Linear Correlation Coefficient."""
    if len(pred) < 3:
        return 0.0
    coef, _ = pearsonr(pred, target)
    return float(coef)


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_kendall_tau(pred: np.ndarray, target: np.ndarray) -> float:
    """Kendall's τ rank correlation."""
    from scipy.stats import kendalltau

    if len(pred) < 3:
        return 0.0
    coef, _ = kendalltau(pred, target)
    return float(coef)


def evaluate_iqa(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict:
    """Compute all IQA evaluation metrics (lowercase keys)."""
    return {
        "srcc": compute_srcc(pred, target),
        "plcc": compute_plcc(pred, target),
        "rmse": compute_rmse(pred, target),
        "kendall_tau": compute_kendall_tau(pred, target),
    }


def per_task_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    task_ids: list,
    task_names: tuple = None,
) -> dict:
    """Compute metrics per task.

    Args:
        task_ids: list of int task indices or str task names
        task_names: optional tuple of task name strings (default: TASK_NAMES)
    """
    if task_names is None:
        task_names = TASK_NAMES
    task_map = {name: i for i, name in enumerate(task_names)}
    if isinstance(task_ids[0], str):
        mapped_ids = np.array([task_map.get(t, 0) for t in task_ids])
    else:
        mapped_ids = np.array(task_ids)

    results = {}
    for i, name in enumerate(task_names):
        mask = mapped_ids == i
        if mask.sum() < 3:
            results[name] = {
                "srcc": 0.0,
                "plcc": 0.0,
                "rmse": 0.0,
                "n": int(mask.sum()),
            }
        else:
            r = evaluate_iqa(pred[mask], target[mask])
            results[name] = {
                "srcc": r["srcc"],
                "plcc": r["plcc"],
                "rmse": r["rmse"],
                "n": int(mask.sum()),
            }
    return results


def per_distortion_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    distortion_labels: list,
) -> dict:
    """Compute metrics per distortion type (lowercase sub-keys)."""
    unique = list(set(distortion_labels))
    results = {}
    for dist in unique:
        mask = np.array([d == dist for d in distortion_labels])
        if mask.sum() < 3:
            results[dist] = {
                "srcc": 0.0,
                "plcc": 0.0,
                "rmse": 0.0,
                "n": int(mask.sum()),
            }
        else:
            results[dist] = evaluate_iqa(pred[mask], target[mask])
            results[dist]["n"] = int(mask.sum())
    return results


def per_distortion_category_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    distortion_labels: list,
    uav_dist_names: set = None,
) -> dict:
    """Compute aggregate metrics for UAV-specific vs generic distortion categories.

    Args:
        uav_dist_names: Set of distortion names classified as UAV-specific.
                        Defaults to the 6 known UAV distortion types.
    Returns:
        {'UAV': {...}, 'Generic': {...}}
    """
    if uav_dist_names is None:
        from uav_iqa.distortions import UAV_DISTORTION_NAMES

        uav_dist_names = UAV_DISTORTION_NAMES

    uav_mask = np.array([d in uav_dist_names for d in distortion_labels])
    generic_mask = ~uav_mask

    results = {}
    for cat, mask in [("UAV", uav_mask), ("Generic", generic_mask)]:
        n = int(mask.sum())
        if n < 3:
            results[cat] = {"srcc": 0.0, "plcc": 0.0, "rmse": 0.0, "n": n}
        else:
            results[cat] = evaluate_iqa(pred[mask], target[mask])
            results[cat]["n"] = n
    return results
