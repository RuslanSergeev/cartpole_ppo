from dataclasses import dataclass
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from .actor import Actor
from .critic import Critic
from .logging import logger


@dataclass
class Checkpoint:
    """
    Checkpoint class to manage saving and loading of model states.
    """

    actor: Actor
    critic: Critic
    actor_optimizer: Adam
    critic_optimizer: Adam
    actor_scheduler: StepLR
    critic_scheduler: StepLR
    episode: int
    checkpoint_path: str


    def save(self) -> None:
        """
        Save the checkpoint to a file.
        """
        # move the nn.Module to CPU before saving
        checkpoint = {
            "actor": self.actor.copy().state_dict().cpu().eval(),
            "critic": self.critic.copy().state_dict().cpu().eval(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
        }
        logger.info(f"Saving checkpoint to {self.checkpoint_path}")
        torch.save(checkpoint, self.checkpoint_path)

    def load(self) -> None:
        """
        Load the checkpoint from a file.
        """
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler"])
