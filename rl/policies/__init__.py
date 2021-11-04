from .mlp_actor_critic import MlpActor, MlpCritic
from .asym_actor_critic import AsymActor, AsymCritic
from .asym_ddpg_actor_critic import AsymDDPGActor, AsymDDPGCritic
from .image_actor_critic import ImageActor, ImageCritic

def get_actor_critic_by_name(name):
    if name == "mlp":
        return MlpActor, MlpCritic
    elif name == "asym":
        return AsymActor, AsymCritic
    elif name == "asym-ddpg":
        return AsymDDPGActor, AsymDDPGCritic
    elif name == "full-image":
        return ImageActor, ImageCritic
    else:
        raise NotImplementedError()
