from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
import torch

from ..config import cfg


def prepare_vae():
    image_processor = VaeImageProcessor(vae_scale_factor=8)

    if cfg.main_process:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            pretrained_model_name_or_local_dir=cfg.pretrained_model,
            subfolder="vae",
            torch_dtype=torch.float16,
            variant="fp16",
        )  # type: AutoencoderKLTemporalDecoder

        vae.eval()

        scheduler = EulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_local_dir=cfg.pretrained_model,
            subfolder="scheduler",
        )  # type: EulerDiscreteScheduler
    else:
        vae = None
        scheduler = None

    return image_processor, vae, scheduler
