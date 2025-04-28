from typing import Optional, Union
import numpy as np
import torch
import random
from .logging import logger


class Hardware_manager:
    """
    A class to manage hardware settings for PyTorch computations.
    This class allows you to set the device (CPU or GPU) and the random seed
    for reproducibility.
    """
    _device: torch.device = torch.device("cpu")
    _seed: int = 42

    @classmethod
    def configure(
        cls,
        user_device: Optional[Union[torch.device, str]] = None,
        seed: int = 42,
    ) -> None:
        """
        Configure the hardware settings.

        Args:
            user_device (Optional[Union[torch.device, str]]): The device to be used.
                If None, it will automatically select CPU or GPU based on availability.
            seed (int): The random seed for reproducibility.
        """
        cls._set_device(user_device)
        cls._set_seed(seed)

    @classmethod
    def _set_device(
        cls, 
        user_device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Get the device to be used for computation.

        Returns:
            torch.device: The device (CPU or GPU) to be used.
        """
        if user_device is None:
            if torch.cuda.is_available():
                cls._device = torch.device("cuda")
            else:
                cls._device = torch.device("cpu")
        elif isinstance(user_device, str):
            cls._device = torch.device(user_device)

        # check the compatibility with the current hardware
        if cls._device.type == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA is not available. Falling back to CPU."
            )
            cls._device = torch.device("cpu")
        elif cls._device.type == "cpu" and torch.cuda.is_available():
            logger.warning(
                "CUDA is available but CPU is selected. "
                "Consider using GPU for better performance."
            )
        # Report the device being used
        if cls._device.type == "cuda":
            device_index = cls._device.index
            logger.info(
                f"Using GPU: {torch.cuda.get_device_name(device_index)} "
            )
        else:
            logger.info("Using CPU.")

    @classmethod
    def _set_seed(cls, seed: int = 42) -> None:
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The seed value to set.
        """
        cls._seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if cls._device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def get_device(cls) -> torch.device:
        """
        Get the current device being used.

        Returns:
            torch.device: The current device.
        """
        return cls._device

    @classmethod
    def get_seed(cls) -> int:
        """
        Get the current random seed.

        Returns:
            int: The current random seed.
        """
        return cls._seed


Hardware_manager.configure()
