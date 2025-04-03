import torch
from copy import deepcopy
from tqdm import tqdm


def sample_from_noise(model, scheduler, cond_latent, cond_embedding, drags,
                      min_guidance=1.0, max_guidance=3.0, num_inference_steps=25, num_frames=14):
    model.eval()

    scheduler_inference = deepcopy(scheduler)
    scheduler_inference.set_timesteps(num_inference_steps, device=cond_latent.device)
    timesteps = scheduler_inference.timesteps
    do_classifier_free_guidance = max_guidance > 1
    latents = torch.randn((1, num_frames, 4, 32, 32)).to(cond_latent) * scheduler_inference.init_noise_sigma
    guidance_scale = torch.linspace(min_guidance, max_guidance, num_frames).unsqueeze(0).to(cond_latent)[..., None, None, None]

    for i, t in tqdm(enumerate(timesteps)):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler_inference.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = model(
                latent_model_input,
                t,
                image_latents=torch.cat([cond_latent, torch.zeros_like(cond_latent)]) if do_classifier_free_guidance else cond_latent,
                encoder_hidden_states=torch.cat([cond_embedding, torch.zeros_like(cond_embedding)]) if do_classifier_free_guidance else cond_embedding,
                added_time_ids=torch.FloatTensor([[6, 127, 0.02] * 2]).to(cond_latent) if do_classifier_free_guidance else torch.FloatTensor([[6, 127, 0.02]]).to(cond_latent),
                drags=torch.cat([drags, torch.zeros_like(drags)]) if do_classifier_free_guidance else drags,
            )

        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        latents = scheduler_inference.step(noise_pred, t, latents).prev_sample
    
    return latents
