class SkelConf:
    def __init__(self):
        self.pretrained_model = "stabilityai/stable-video-diffusion-img2vid"
        
        self.results_dir = "/home/c_capzw/c_cape3d_scratch/runs"
        self.test_dir = None
        self.resume_checkpoint = "/home/c_capzw/c_cape3d_scratch/runs/009/checkpoints/0046800-ckpt.pt"
        self.h5_path = "/home/c_capzw/c_cape3d_scratch/data/latent_states_embeddings_and_images_1000_v2.h5"
        
        self.num_max_drags = 50
        self.num_frames = 14

        self.random_seed = 1024
        self.learning_rate = 1.e-5
        self.num_steps = 1500000
        self.global_batch_size = 1
        self.num_workers = 2

        self.log_every = 50
        self.visualize_every = 300
        self.ckpt_every = 600
        self.use_wandb = True

        self.log_sigma_std = 1.6
        self.log_sigma_mean = 0.7
        self.zero_init = True
        
        self.cond_dropout_prob = 0.1
        self.drag_token_cross_attn = True
        self.use_modulate = True
        self.pos_embed_dim = 64
        self.drag_embedder_out_channels = [256, 320, 320]
        
        self.enable_gradient_checkpointing = True
        self.non_first_frame_weight = 1.0
        self.weight_increasing = True
        self.max_grad_norm = 5.0
