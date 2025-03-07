import torch.nn.functional as F


def custom_loss(preds, targets):
    """
    Custom loss: Computes MSE loss for depth and volume.

    Note: The area_denorm parameter is retained in the signature for compatibility,
    but is not used in the loss calculation.

    Parameters:
        preds (torch.Tensor): Predictions of shape (N, 2) where:
            - Column 0 is predicted depth.
            - Column 1 is predicted volume.
        targets (torch.Tensor): Ground truth tensor of shape (N, 2).
        area_denorm (torch.Tensor): Area normalization factors (not used in this version).

    Returns:
        dict: A dictionary containing:
            - 'total_loss': Sum of depth and volume MSE losses.
            - 'loss_depth': MSE loss for depth.
            - 'loss_volume': MSE loss for volume.
    """
    # Extract depth and volume predictions and targets.
    pred_depth = preds[:, 0]
    pred_volume = preds[:, 1]
    target_depth = targets[:, 0]
    target_volume = targets[:, 1]

    # Compute mean squared error losses directly.
    loss_depth = F.mse_loss(pred_depth, target_depth, reduction='mean')
    loss_volume = F.mse_loss(pred_volume, target_volume, reduction='mean')

    total_loss = loss_depth + loss_volume

    return {
        'total_loss': total_loss,
        'loss_depth': loss_depth,
        'loss_volume': loss_volume,
    }
