import os
import argparse
from StableVideo.Utils.decode import decode_latents_and_save
from StableVideo.dataset import DragVideoDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf


def render_all(
    pretrained_model_name_or_local_dir,
    dataset_args,
    output_dir,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading VAE from {pretrained_model_name_or_local_dir}")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name_or_local_dir,
        subfolder="vae",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    vae.eval()

    image_processor = VaeImageProcessor(vae_scale_factor=8)

    for k in dataset_args:
        if "roots" in k:
            from glob import glob
            dataset_args[k] = sorted(glob(dataset_args[k]))
    
    dataset = DragVideoDataset(**dataset_args)
    dataloader = DataLoader(
        dataset,
        shuffle=False
    )
    
    print(f"Starting to render {len(dataset)} samples to {output_dir}")
    
    for i, batch in enumerate(tqdm(dataloader, desc="Rendering progress")):
        for j in range(len(batch["latents"])):
            sample_idx = i + j
            
            latents = batch["latents"][j].to(device)
            drags = batch["drags"][j].to(device)
            
            output_path = os.path.join(output_dir, f"sample_{sample_idx:05d}_{batch['name']}.png")
            decode_latents_and_save(
                vae, 
                image_processor, 
                latents, 
                output_path, 
                drags
            )
    
    print(f"All samples have been rendered and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/c_capzw/notebooks/configs/train-puppet-master.yaml")
    parser.add_argument("--output_dir", type=str, default="/home/c_capzw/c_cape3d_scratch/data/rendered")
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    render_all(
        pretrained_model_name_or_local_dir=config["pretrained_model_name_or_local_dir"],
        dataset_args=config["dataset_args"],
        output_dir=args.output_dir,
    )
