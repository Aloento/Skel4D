import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from einops import rearrange
import wandb

from src.utils.train_util import instantiate_from_config
from src.utils.attention_extraction import extract_sk_attention_auto_dimensions
from src.utils.keypoint_losses import compute_sharpening_loss_batch, find_top_k_gaussian_batch, furthest_point_sampling_batch
from src.utils.keypoint_visualization import extract_and_visualize_keypoints_from_sk_ref
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, UNet2DConditionModel
from .pipeline import RefOnlyNoisedUNet
from StableKeypoints.optimization.losses import equivariance_loss as sk_equivariance_loss


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class MVDiffusion(pl.LightningModule):
    def __init__(
        self,
        stable_diffusion_config,
        drop_cond_prob=0.1,
        # StableKeypoints configuration
        use_stable_keypoints=False,
        num_learnable_tokens=16,
        sk_loss_weights=None,
        # M1 toggles
        use_equivariance_mode: bool = True,
        equivariance_transform_cfg: dict | None = None,
    ):
        super(MVDiffusion, self).__init__()

        self.drop_cond_prob = drop_cond_prob
        
        # StableKeypoints setup
        self.use_stable_keypoints = use_stable_keypoints
        self.num_learnable_tokens = num_learnable_tokens
        self.sk_loss_weights = sk_loss_weights
        # M1 flags
        self.use_equivariance_mode = use_equivariance_mode
        self.equivariance_transform_cfg = equivariance_transform_cfg or {
            'degrees': 15.0, 'scale': (0.95, 1.05), 'translate': (0.05, 0.05)
        }

        self.register_schedule()

        # init modules
        pipeline_kwargs = dict(**stable_diffusion_config)
        if self.use_stable_keypoints:
            # Add StableKeypoints parameters to pipeline
            pipeline_kwargs.update({
                'use_learnable_embeddings': True,
                'num_learnable_tokens': self.num_learnable_tokens,
            })
            
        pipeline = DiffusionPipeline.from_pretrained(**pipeline_kwargs)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.pipeline = pipeline

        # Enable StableKeypoints training mode if configured
        if self.use_stable_keypoints:
            #self.pipeline.enable_keypoint_learning()
            print(f"StableKeypoints enabled with {self.num_learnable_tokens} learnable tokens")
            print(f"SK loss weights: {self.sk_loss_weights}")

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = RefOnlyNoisedUNet(self.pipeline.unet, train_sched, self.pipeline.scheduler)

        self.train_scheduler = train_sched      # use ddpm scheduler during training

        self.unet = pipeline.unet

        # validation output buffer
        self.validation_step_outputs = []

    def log_wandb_metrics(self, dictionary):
        """Enhanced wandb logging with custom metrics"""
        if (self.global_rank == 0 and hasattr(self.logger, 'experiment') and 
            hasattr(self.logger.experiment, 'log')):
            # Create a filtered dict for wandb with additional computed metrics
            wandb_dict = {}
            
            for key, value in dictionary.items():
                # Convert tensor values to float for wandb
                if isinstance(value, torch.Tensor):
                    wandb_dict[key] = value.item()
                else:
                    wandb_dict[key] = value
            
            # Add custom metrics for StableKeypoints
            if self.use_stable_keypoints:
                if 'train/loss' in wandb_dict and 'train/nvs_loss' in wandb_dict:
                    # Compute SK loss contribution
                    sk_contribution = wandb_dict['train/loss'] - wandb_dict['train/nvs_loss']
                    wandb_dict['train/sk_loss_contribution'] = sk_contribution
                
                if 'train/sk_sharpening_loss' in wandb_dict:
                    # Compute normalized sharpening loss (for better tracking)
                    normalized_sharpening = wandb_dict['train/sk_sharpening_loss'] / self.sk_loss_weights['sharpening']
                    wandb_dict['train/sk_sharpening_normalized'] = normalized_sharpening
            
            # Log to wandb with error handling - only on rank 0 to avoid step conflicts
            if self.global_rank == 0:
                try:
                    self.logger.experiment.log(wandb_dict, step=self.global_step)
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")

    def register_schedule(self):
        self.num_timesteps = 1000

        # replace scaled_linear schedule with linear schedule as Zero123++
        beta_start = 0.00085
        beta_end = 0.0120
        betas = torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).float())
        
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())
    
    def on_fit_start(self):
        device = torch.device(f'cuda:{self.global_rank}')
        self.pipeline.to(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)
    
    def prepare_batch_data(self, batch):
        # prepare stable diffusion input
        cond_imgs = batch['cond_imgs']      # (B, C, H, W)        

        # random resize the condition image
        cond_size = np.random.randint(128, 513)
        cond_imgs = v2.functional.resize(cond_imgs, cond_size, interpolation=3, antialias=True).clamp(0, 1)

        target_imgs = batch['target_imgs']  # (B, 6, C, H, W)
        target_imgs = v2.functional.resize(target_imgs, 320, interpolation=3, antialias=True).clamp(0, 1)

        if not getattr(self, 'use_equivariance_mode', False):
            target_grids = rearrange(target_imgs, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=2)
            cond_imgs = cond_imgs.to(self.device)
            target_imgs = target_imgs.to(self.device)
            target_grids = target_grids.to(self.device)
            return cond_imgs, target_imgs, target_grids

        # Equivariance mode: build 3+3 randomized grid using invertible transforms (SK style)
        from src.utils.equivariance import build_random_invertible_transform
        B, V, C, H, W = target_imgs.shape  # V should be 6
        device = self.device
        eq_pairs = []
        eq_transforms = []
        new_views = []  # per-sample list of 6 views

        for b in range(B):
            # choose 3 unique indices from the 6 available target views
            idx = torch.randperm(V)[:3].tolist()
            originals = [target_imgs[b, i] for i in idx]  # list of 3 tensors [C,H,W]

            # build per-sample transform and apply to the batch of 3 originals (to keep 3 thetas)
            transform = build_random_invertible_transform(
                **self.equivariance_transform_cfg
            )
            originals_b = torch.stack(originals, dim=0)  # [3,C,H,W]
            transformed_b = transform(originals_b)       # [3,C,H,W], stores theta per item
            transformed = [transformed_b[i] for i in range(3)]

            # concatenate originals and transformed
            six_views = originals + transformed  # list of 6 [C,H,W]

            # random permutation of 6
            perm = torch.randperm(6).tolist()
            six_views_perm = [six_views[p] for p in perm]

            # compute paired indices in permuted order
            # originals at 0,1,2 correspond to transformed at 3,4,5 pre-permutation
            pre_pairs = [(i, i+3) for i in range(3)]  # order defines theta index
            # map to permuted indices
            perm_pairs = [(perm.index(a), perm.index(b)) for (a, b) in pre_pairs]

            # collect for this sample
            eq_pairs.append(perm_pairs)       # same order as theta indices 0..2
            eq_transforms.append(transform)   # has last_params.theta of shape [3,2,3]
            new_views.append(torch.stack(six_views_perm, dim=0))  # [6,C,H,W]

        # stack into batch
        target_imgs_eq = torch.stack(new_views, dim=0)  # [B,6,C,H,W]
        target_grids = rearrange(target_imgs_eq, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=2)

        # move to device
        cond_imgs = cond_imgs.to(device)
        target_imgs_eq = target_imgs_eq.to(device)
        target_grids = target_grids.to(device)

        eq_meta = {
            'pairs': eq_pairs,             # List[B] of List[3] of (o_idx, t_idx) in permuted grid
            'transforms': eq_transforms,   # List[B] of RandomAffineWithInverse (theta per 3 originals)
        }
        return cond_imgs, target_imgs_eq, target_grids, eq_meta
    
    @torch.no_grad()
    def forward_vision_encoder(self, images):
        dtype = next(self.pipeline.vision_encoder.parameters()).dtype
        image_pil = [v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])]
        image_pt = self.pipeline.feature_extractor_clip(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        global_embeds = self.pipeline.vision_encoder(image_pt, output_hidden_states=False).image_embeds
        global_embeds = global_embeds.unsqueeze(-2)

        # Use the newer encode_prompt method instead of deprecated _encode_prompt
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt="", 
            device=self.device, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=False
        )
        encoder_hidden_states = prompt_embeds
        ramp = global_embeds.new_tensor(self.pipeline.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp

        return encoder_hidden_states
    
    @torch.no_grad()
    def encode_condition_image(self, images):
        #print(f"Encoding condition images with shape: {images.shape}")
        dtype = next(self.pipeline.vae.parameters()).dtype
        image_pil = [v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])]
        image_pt = self.pipeline.feature_extractor_vae(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        latents = self.pipeline.vae.encode(image_pt).latent_dist.sample()
        return latents
    
    @torch.no_grad()
    def encode_target_grids(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8   # [-0.625, 0.625]
        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents
    
    @torch.no_grad()
    def encode_target_images(self, images):
        """
        Encode target images for StableKeypoints processing following the original SK approach.
        
        Args:
            images: Target images tensor with shape (batch_size, num_views, channels, height, width)
                   or (num_views, channels, height, width)
                   Expected to be in range [0, 1]
                   
        Returns:
            Encoded target latents ready for StableKeypoints processing
        """
        if not self.use_stable_keypoints:
            return None
            
        dtype = next(self.pipeline.vae.parameters()).dtype
        original_shape = images.shape
        
        
        batch_size, num_views, channels, height, width = images.shape
        # Reshape for VAE encoding: (batch_size * num_views, channels, height, width)
        images_flat = images.view(batch_size * num_views, channels, height, width)        
        
        # Convert from [0, 1] to [-1, 1] range (following SK approach: image * 2 - 1)
        images_normalized = images_flat * 2.0 - 1.0
        images_normalized = images_normalized.to(dtype)
        
        # Encode to latent space following StableKeypoints approach
        # SK uses: model.vae.encode(image)["latent_dist"].mean * 0.18215
        latent_dist = self.pipeline.vae.encode(images_normalized).latent_dist
        latents = latent_dist.mean * 0.18215  # Use mean and SK scaling factor
        
        # Restructure back to original batch format
        # latents shape: (batch_size * num_views, latent_channels, latent_height, latent_width)
        latent_channels, latent_height, latent_width = latents.shape[1], latents.shape[2], latents.shape[3]
        
        
        # Reshape back to: (batch_size, num_views, latent_channels, latent_height, latent_width)
        latents = latents.view(batch_size, num_views, latent_channels, latent_height, latent_width)
        
        
        # Return latents in format matching original input structure
        return latents
    
    def forward_unet(self, latents_noisy, t, prompt_embeds, cond_latents):
        dtype = next(self.pipeline.unet.parameters()).dtype
        latents_noisy = latents_noisy.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        cond_latents = cond_latents.to(dtype)
        cross_attention_kwargs = dict(cond_lat=cond_latents)
        pred_noise = self.pipeline.unet(
            latents_noisy,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        return pred_noise
    
    def forward_unet_with_sk(self, latents_noisy, target_latents, t, prompt_embeds, cond_latents):
        """
        Forward pass with StableKeypoints integration.
        Returns both UNet output and StableKeypoints losses.
        """
        dtype = next(self.pipeline.unet.parameters()).dtype
        latents_noisy = latents_noisy.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        cond_latents = cond_latents.to(dtype)
        target_latents = target_latents.to(dtype)

        sk_ref_dict = {}
        learnable_embeddings=self.pipeline.get_learnable_embeddings().to(self.device)
                
        # Create cross_attention_kwargs with learnable embeddings
        cross_attention_kwargs = dict(
            cond_lat=cond_latents,
            sk_ref_dict=sk_ref_dict,  # Pass sk_ref_dict for SK processing
            target_lat=target_latents,  # Use latents as target_lat for SK training
            learnable_embeddings=learnable_embeddings,
        )

        #print(f"Forwarding UNet with SK enabled, latents shape: {latents_noisy.shape}, t: {t.shape}, prompt_embeds shape: {prompt_embeds.shape}, cond_latents shape: {cond_latents.shape}, learnable_embeddings shape: {learnable_embeddings.shape}")
        
        # Forward pass through UNet (SK losses computed internally)
        pred_noise = self.pipeline.unet(
            latents_noisy,
            t,
            encoder_hidden_states=torch.cat([prompt_embeds, learnable_embeddings], dim=1),
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]   
        
        return pred_noise, sk_ref_dict
    
    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
    
    def training_step(self, batch, batch_idx):
        # get input
        out = self.prepare_batch_data(batch)
        if getattr(self, 'use_equivariance_mode', False):
            cond_imgs, target_imgs, target_grids, eq_meta = out
        else:
            cond_imgs, target_imgs, target_grids = out
            eq_meta = None

        # sample random timestep
        B = cond_imgs.shape[0]
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)

        # classifier-free guidance
        if np.random.rand() < self.drop_cond_prob:
            # Use the newer encode_prompt method for empty prompts during classifier-free guidance
            prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=[""] * B, 
                device=self.device, 
                num_images_per_prompt=1, 
                do_classifier_free_guidance=False
            )
            cond_latents = self.encode_condition_image(torch.zeros_like(cond_imgs))            
        else:
            #print("Using condition images for training")
            #print(f"Condition images shape: {cond_imgs.shape}")
            prompt_embeds = self.forward_vision_encoder(cond_imgs)
            cond_latents = self.encode_condition_image(cond_imgs)

        # Encode target images
        target_latents = self.encode_target_images(target_imgs)

        latents = self.encode_target_grids(target_grids)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)
        
        # Forward pass with potential StableKeypoints integration
        if self.use_stable_keypoints:
            # Use StableKeypoints-enabled forward pass
            v_pred, sk_ref_dict = self.forward_unet_with_sk(latents_noisy, target_latents, t, prompt_embeds, cond_latents)
        else:
            # Standard forward pass
            v_pred = self.forward_unet(latents_noisy, t, prompt_embeds, cond_latents)
            sk_ref_dict = None

        v_target = self.get_v(latents, noise, t)

        loss, loss_dict = self.compute_loss(v_pred, v_target, sk_ref_dict, eq_meta)

        # Attach basic eq metadata stats for logging (debug)
        if eq_meta is not None and self.global_rank == 0:
            loss_dict['train/eq_pairs_count'] = torch.tensor(sum(len(p) for p in eq_meta['pairs'])/B, device=self.device, dtype=torch.float32)

        # logging
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        # Enhanced wandb logging
        self.log_wandb_metrics(loss_dict)

        if self.global_step % 100 == 0 and self.global_rank == 0:
            with torch.no_grad():
                latents_pred = self.predict_start_from_z_and_v(latents_noisy, t, v_pred)

                latents = unscale_latents(latents_pred)
                images = unscale_image(self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
                images = (images * 0.5 + 0.5).clamp(0, 1)
                images = torch.cat([target_grids, images], dim=-2)

                # Save standard reconstruction visualization
                grid = make_grid(images, nrow=images.shape[0], normalize=True, value_range=(0, 1))
                save_image(grid, os.path.join(self.logdir, 'images', f'train_{self.global_step:07d}.png'))
                
                # Log to wandb with error handling - only on rank 0 to avoid step conflicts
                if (self.global_rank == 0 and hasattr(self.logger, 'experiment') and 
                    hasattr(self.logger.experiment, 'log')):
                    try:
                        self.logger.experiment.log({
                            "train_reconstruction": wandb.Image(grid, caption=f"Training reconstruction at step {self.global_step}")
                        })
                    except Exception as e:
                        print(f"Warning: Failed to log train image to wandb: {e}")
                
                # Save keypoint visualization if StableKeypoints is enabled and sk_ref_dict is available
                if self.use_stable_keypoints and sk_ref_dict:
                    keypoint_result = extract_and_visualize_keypoints_from_sk_ref(sk_ref_dict, target_imgs)
                    if keypoint_result is not None:
                        keypoint_grid, attention_stats = keypoint_result
                        save_image(keypoint_grid, os.path.join(self.logdir, 'images', f'train_keypoints_{self.global_step:07d}.png'))
                        
                        # Log keypoint visualization to wandb with error handling - only on rank 0
                        if (self.global_rank == 0 and hasattr(self.logger, 'experiment') and 
                            hasattr(self.logger.experiment, 'log')):
                            try:
                                self.logger.experiment.log({
                                    "keypoint_heatmaps": wandb.Image(keypoint_grid, caption=f"Keypoint heatmaps at step {self.global_step}")
                                })
                                # Log attention statistics as well
                                if attention_stats:
                                    self.logger.experiment.log(attention_stats)
                            except Exception as e:
                                print(f"Warning: Failed to log keypoint image to wandb: {e}")
                        
                        print(f"Saved keypoint visualization for step {self.global_step}")

        return loss
        
    def compute_loss(self, noise_pred, noise_gt, sk_ref_dict=None, eq_meta=None):
        # Standard NVS loss
        nvs_loss = F.mse_loss(noise_pred, noise_gt)

        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/nvs_loss': nvs_loss})
        
        total_loss = nvs_loss * self.sk_loss_weights['nvs']

        if sk_ref_dict and self.use_stable_keypoints:
            sk_losses = self.compute_sk_losses_from_attention(sk_ref_dict, eq_meta)
            
            if sk_losses:
                sharpening_loss = sk_losses.get('sharpening', torch.tensor(0.0, device=self.device))
                
                loss_dict[f'{prefix}/sk_sharpening_loss'] = sharpening_loss
                total_loss += sharpening_loss * self.sk_loss_weights['sharpening']

                if 'equivariance' in sk_losses:
                    eq_loss_val = sk_losses['equivariance']
                    loss_dict[f'{prefix}/sk_equivariance_loss'] = eq_loss_val
                    total_loss += eq_loss_val * self.sk_loss_weights['equivariance']
        
        loss_dict.update({f'{prefix}/loss': total_loss})

        return total_loss, loss_dict
    
    def compute_sk_losses_from_attention(self, sk_ref_dict, eq_meta=None):
        """
        Compute StableKeypoints losses from collected attention maps, including optional equivariance.
        
        Args:
            sk_ref_dict: Dict[layer_name -> attention tensor]
            eq_meta: Optional metadata with pairs and transforms when equivariance mode is enabled
        
        Returns:
            Dict with 'sharpening' and optionally 'equivariance'
        """

        
        
        # Filter for high-resolution layers (most informative for keypoints)
        target_layers = ['down_blocks.0', 'up_blocks.3']  # 9600 spatial resolution
        
        aggregated_attention_maps = []
        
        for layer_name, attention_tensor in sk_ref_dict.items():
            # Check if this is a target layer
            if any(target in layer_name for target in target_layers):
                # Extract SK attention maps with individual view separation
                sk_data = extract_sk_attention_auto_dimensions(attention_tensor)
                # sk_data['individual_views'] has shape [6, view_spatial, 16]
                aggregated_attention_maps.append(sk_data['individual_views'])
        
        
        
        if not aggregated_attention_maps:
            return {}
        
        # Average attention maps across layers: [6, view_spatial, 16]
        avg_attention_maps = torch.mean(torch.stack(aggregated_attention_maps), dim=0)
        
        # Compute sharpening loss for each view separately
        total_sharpening_loss = 0.0
        num_views = avg_attention_maps.shape[0]
        spatial_dim = avg_attention_maps.shape[1]
        # determine H,W
        if spatial_dim == 1600:
            view_h, view_w = 40, 40
        elif spatial_dim == 400:
            view_h, view_w = 20, 20
        elif spatial_dim == 100:
            view_h, view_w = 10, 10
        else:
            side = int(spatial_dim ** 0.5)
            if side * side != spatial_dim:
                return {'sharpening': torch.tensor(0.0, device=avg_attention_maps.device)}
            view_h, view_w = side, side
        
        spatial_views = []  # list of [16,H,W]
        for view_idx in range(num_views):
            view_attention = avg_attention_maps[view_idx]  # [view_spatial, 16]
            spatial_attention = view_attention.T.reshape(16, view_h, view_w)
            spatial_views.append(spatial_attention)
            view_sharpening_loss = compute_sharpening_loss_batch(
                spatial_attention.unsqueeze(0),
                apply_diversity_filter=True,
                top_k=10,
                furthest_point_num_samples=16,
                sigma=1.0
            )
            total_sharpening_loss += view_sharpening_loss
        
        avg_sharpening_loss = total_sharpening_loss / num_views

        losses = {'sharpening': avg_sharpening_loss}

        # Equivariance loss (on corresponding pairs only)
        if eq_meta is not None and 'pairs' in eq_meta and 'transforms' in eq_meta and len(eq_meta['pairs']) > 0:
            # Use first sample's mapping due to aggregation across batch
            pairs = eq_meta['pairs'][0]          # List[(o_idx, t_idx)] length 3
            transform = eq_meta['transforms'][0] # RandomAffineWithInverse with theta for 3 originals
            eq_losses = []
            for i, (o_idx, t_idx) in enumerate(pairs):
                A_o = spatial_views[o_idx]  # [16,H,W]
                A_t = spatial_views[t_idx]  # [16,H,W]
                # Normalize per token across spatial for stability
                A_o_flat = A_o.view(16, -1)
                A_t_flat = A_t.view(16, -1)
                A_o_norm = F.softmax(A_o_flat, dim=-1).view_as(A_o)
                A_t_norm = F.softmax(A_t_flat, dim=-1).view_as(A_t)
                # Candidate selection
                candidates = find_top_k_gaussian_batch(A_o_norm, top_k=16, sigma=1.0)
                selected = furthest_point_sampling_batch(A_o_norm, top_k=min(10, candidates.shape[0]), initial_candidates=candidates)
                if selected.numel() == 0:
                    continue
                Emb_o = A_o_norm[selected]  # [K,H,W]
                Emb_t = A_t_norm[selected]  # [K,H,W]
                # Build a batch along transform dimension (size 3) to match theta indexing
                # repeat Emb_t across dim0 to length 3 so inverse can pick index i
                Emb_t_batched = Emb_t.unsqueeze(0).repeat(3, 1, 1, 1)  # [3,K,H,W]
                # Compute SK equivariance loss (will inverse with transform and pick [index])
                loss_eq = sk_equivariance_loss(Emb_o, Emb_t_batched, transform, i)
                eq_losses.append(loss_eq)
            if len(eq_losses) > 0:
                losses['equivariance'] = torch.stack(eq_losses).mean()
        
        return losses
            

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # get input
        cond_imgs, _, target_imgs = self.prepare_batch_data(batch)

        images_pil = [v2.functional.to_pil_image(cond_imgs[i]) for i in range(cond_imgs.shape[0])]

        outputs = []
        for cond_img in images_pil:
            latent = self.pipeline(cond_img, num_inference_steps=75, output_type='latent').images
            image = unscale_image(self.pipeline.vae.decode(latent / self.pipeline.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
            image = (image * 0.5 + 0.5).clamp(0, 1)
            outputs.append(image)
        outputs = torch.cat(outputs, dim=0).to(self.device)
        images = torch.cat([target_imgs, outputs], dim=-2)
        
        self.validation_step_outputs.append(images)
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=0)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

        if self.global_rank == 0:
            grid = make_grid(all_images, nrow=8, normalize=True, value_range=(0, 1))
            save_image(grid, os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png'))
            
            # Log validation images to wandb with error handling - only on rank 0 to avoid step conflicts
            if (self.global_rank == 0 and hasattr(self.logger, 'experiment') and 
                hasattr(self.logger.experiment, 'log')):
                try:
                    self.logger.experiment.log({
                        "validation_reconstruction": wandb.Image(grid, caption=f"Validation reconstruction at step {self.global_step}")
                    })
                except Exception as e:
                    print(f"Warning: Failed to log validation image to wandb: {e}")

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        lr = self.learning_rate

        # Collect parameters to optimize
        params_to_optimize = list(self.unet.parameters())
        
        # Add learnable embeddings if StableKeypoints is enabled
        if self.use_stable_keypoints:
            sk_params = self.pipeline.get_learnable_embedding_parameters()
            if sk_params:
                params_to_optimize.extend(sk_params)
                print(f"Added {len(sk_params)} StableKeypoints parameters to optimizer")

        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3000, eta_min=lr/4)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}