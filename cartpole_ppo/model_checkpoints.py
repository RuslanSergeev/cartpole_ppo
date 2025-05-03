from typing import Dict, Any
import torch
from .logging import logger
from .hardware_manager import Hardware_manager


class Checkpoint:
    """
    Checkpoint class to manage saving and loading of model states.
    """

    def __init__(
        self, 
        checkpoint_path: str, 
        checkpoint_data: Dict[str, Any]
    ) -> None:
        """
        Initialize the Checkpoint class.
        Args:
            checkpoint_path (str): Path to save the checkpoint.
            checkpoint_data (Dict[str, Any]): Data to be saved in the checkpoint,
                on the save method.
        """
        self.path = checkpoint_path
        self.data = checkpoint_data

    def save(self, **kwargs) -> None:
        """
        Save the checkpoint to a file.
        """
        self.data.update(kwargs)
        sanitized_checkpoint = {}
        for key, value in self.data.items():
            if hasattr(value, "state_dict"):
                sanitized_checkpoint[key] = value.state_dict()
            else:
                sanitized_checkpoint[key] = value
        torch.save(sanitized_checkpoint, self.path)
        logger.info(f"Saved checkpoint to {self.path}")

    def load(
        self, 
        device: torch.device=Hardware_manager.get_device()
    ) -> None:
        """
        Load the checkpoint from a file.
        """
        logger.info(f"Loading checkpoint from {self.path}")
        checkpoint = torch.load(
            self.path, 
            map_location=device
        )
        for key, value in self.data.items():
            if hasattr(value, "load_state_dict"):
                value.load_state_dict(checkpoint[key])
            else:
                self.data[key] = checkpoint[key]
