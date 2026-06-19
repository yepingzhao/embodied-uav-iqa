import numpy as np
from scipy.stats import pearsonr, spearmanr


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
    """Compute all IQA evaluation metrics."""
    return {
        "SRCC": compute_srcc(pred, target),
        "PLCC": compute_plcc(pred, target),
        "RMSE": compute_rmse(pred, target),
        "KendallTau": compute_kendall_tau(pred, target),
    }


def per_task_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    task_ids: list,
    task_names: list = None,
) -> dict:
    """Compute metrics per task.

    Args:
        task_ids: list of int task indices or str task names
        task_names: optional list of task name strings
    """
    if task_names is None:
        task_names = ["tracking", "inspection", "delivery", "sar"]
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
                "srcc": r["SRCC"],
                "plcc": r["PLCC"],
                "rmse": r["RMSE"],
                "n": int(mask.sum()),
            }
    return results


def compute_metrics(target: np.ndarray, pred: np.ndarray) -> dict:
    """Alias for evaluate_iqa with lowercase keys."""
    r = evaluate_iqa(pred, target)
    return {
        "srcc": r["SRCC"],
        "plcc": r["PLCC"],
        "rmse": r["RMSE"],
        "kendall_tau": r["KendallTau"],
    }


def per_distortion_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    distortion_labels: list,
) -> dict:
    """Compute metrics per distortion type."""
    unique = list(set(distortion_labels))
    results = {}
    for dist in unique:
        mask = np.array([d == dist for d in distortion_labels])
        if mask.sum() < 3:
            results[dist] = {
                "SRCC": 0.0,
                "PLCC": 0.0,
                "RMSE": 0.0,
                "N": int(mask.sum()),
            }
        else:
            results[dist] = evaluate_iqa(pred[mask], target[mask])
            results[dist]["N"] = int(mask.sum())
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
        uav_dist_names = {
            "propeller_vibration_blur",
            "atmospheric_scattering_haze",
            "six_dof_viewpoint_blur",
            "communication_packet_loss",
            "low_res_super_resolution",
            "propeller_shadow",
        }

    uav_mask = np.array([d in uav_dist_names for d in distortion_labels])
    generic_mask = ~uav_mask

    results = {}
    for cat, mask in [("UAV", uav_mask), ("Generic", generic_mask)]:
        n = int(mask.sum())
        if n < 3:
            results[cat] = {"SRCC": 0.0, "PLCC": 0.0, "RMSE": 0.0, "N": n}
        else:
            results[cat] = evaluate_iqa(pred[mask], target[mask])
            results[cat]["N"] = n
    return results
