from typing import Dict, Any
import torch
from .logging import logger
from .hardware_manager import Hardware_manager


class Checkpoint:
    """
    Checkpoint class to manage saving and loading of model states.
    """

    @staticmethod
    def save(
        checkpoint_path: str, 
        source: Dict[str, Any]
    ) -> None:
        """
        Save the checkpoint to a file.
        """
        checkpoint = {}
        for key, value in source.items():
            if hasattr(value, "state_dict"):
                checkpoint[key] = value.state_dict()
            else:
                checkpoint[key] = value
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    @staticmethod
    def load(
        checkpoint_path: str,
        target: Dict[str, Any],
        device: torch.device=Hardware_manager.get_device()
    ) -> None:
        """
        Load the checkpoint from a file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        for key, value in target.items():
            if hasattr(value, "load_state_dict"):
                target[key].load_state_dict(checkpoint[key])
            else:
                target[key] = checkpoint[key]
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
