import torch
import numpy as np
import cv2
from PIL import Image

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_gif


def tensor2vid(video: torch.Tensor, processor: VaeImageProcessor, output_type: str = "np"):
    batch_size = video.shape[0]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


def decode_latents_and_save(vae, image_processor, latents, save_path, drags=None):
    with torch.no_grad():
        frames = vae.decode(latents.to(torch.float16) / 0.18215, num_frames=14).sample.float()
    frames = tensor2vid(frames[None].detach().permute(0, 2, 1, 3, 4), image_processor, output_type="pil")[0]
    
    # Add drag visualizations.
    if drags is not None:
        final_video = []
        for fid, frame in enumerate(frames):
            frame_np = np.array(frame).copy()
            for pid in range(drags.shape[1]):
                if (drags[fid, pid] != 0).any():
                    frame_np = cv2.circle(
                        frame_np, 
                        (int(drags[fid, pid, 0] * 256), int(drags[fid, pid, 1] * 256)), 
                        3, (255, 0, 0), -1
                    )
                    frame_np = cv2.circle(
                        frame_np, 
                        (int(drags[fid, pid, 2] * 256), int(drags[fid, pid, 3] * 256)), 
                        3, (0, 255, 0), -1
                    )
                    frame_np = cv2.line(
                        frame_np, 
                        (int(drags[fid, pid, 0] * 256), int(drags[fid, pid, 1] * 256)), 
                        (int(drags[fid, pid, 2] * 256), int(drags[fid, pid, 3] * 256)), 
                        (0, 0, 255), 
                        2
                    )
            final_video.append(Image.fromarray(frame_np))
    else:
        final_video = frames

    export_to_gif(final_video, save_path)
