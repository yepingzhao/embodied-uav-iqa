def count_parameters(model) -> tuple:
    """Return (total_params, trainable_params) for the given model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
