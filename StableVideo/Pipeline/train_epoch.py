import torch
import torch.distributed
from ..Networks.DragSpatioModel import UNetDragSpatioTemporalConditionModel

from ..config import cfg


def train_epoch(
        model: UNetDragSpatioTemporalConditionModel,
        batch: torch.Tensor,
        device: torch.device,
):
    model.train()

    sample = batch["sample"].to(device)
    cond_latent = batch["cond_latent"].to(device)
    encoder_hidden_states = batch["embedding"].to(device)
    drags = batch["drags"].to(device)

    batch_size = sample.shape[0]
    
    log_sigmas = torch.randn(batch_size, device=device) * cfg.log_sigma_std + cfg.log_sigma_mean
    timesteps = log_sigmas * 0.25
    sigmas = torch.exp(log_sigmas).to(sample)
    sigmas = sigmas[(...,) + (None,) * (len(sample.shape) - 1)]
    
    model_kwargs = dict(
        image_latents=cond_latent,
        encoder_hidden_states=encoder_hidden_states,
        added_time_ids=torch.FloatTensor([[6, 127, 0.02] * batch_size]).to(device),
        drags=drags,
    )

    noise = torch.randn_like(sample)
    noised_sample = sample + sigmas * noise

    model_output = model(noised_sample, timesteps, **model_kwargs)
    pred = model_output * (-sigmas / (sigmas ** 2 + 1) ** 0.5) + noised_sample / (sigmas ** 2 + 1)
    
    loss_weighting = 1 + 1 / sigmas ** 2
    loss_weighting = loss_weighting.repeat(1, 14, 1, 1, 1)

    if cfg.weight_increasing and cfg.non_first_frame_weight > 1:
        loss_weighting = loss_weighting * torch.linspace(1, cfg.non_first_frame_weight, 14)[..., None, None, None].to(device)
    elif cfg.non_first_frame_weight > 1:
        loss_weighting[:, 1:] = loss_weighting[:, 1:] * cfg.non_first_frame_weight
    
    loss = (loss_weighting * torch.nan_to_num((sample * 10.0 - pred * 10.0) ** 2, nan=0.0)).mean()
    is_nan = torch.tensor(1.0 if torch.isnan(loss) or loss.item() < 0 else 0.0, device=device)
    
    if cfg.distributed:
        torch.distributed.all_reduce(is_nan, op=torch.distributed.ReduceOp.SUM)
    
    global_nan = is_nan.item()
    del is_nan, loss_weighting, noise, noised_sample, pred, model_output, sample, cond_latent, encoder_hidden_states, drags, log_sigmas, timesteps, sigmas, model_kwargs
    return loss, global_nan
