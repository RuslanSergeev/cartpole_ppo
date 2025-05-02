import torch

def convert_to_tensor(value, device=None, dtype=None, feature_dim=1):
    """
    Convert a value to a tensor.
    Args:
        value (Any): The value to convert.
    Returns:
        torch.Tensor: The converted tensor.
    """
    if not isinstance(value, torch.Tensor):
        # Single non-tensor value
        value = torch.tensor(value)
    # If squashed, convert to expected shape
    if value.dim() == 1:
        value = value.view(-1, feature_dim)
    # Flatten all the dimensions except the last one:
    value = value.flatten(start_dim=0, end_dim=-2)
    # Move the tensor to the specified device and dtype
    value = value.to(device=device, dtype=dtype)

    return value
