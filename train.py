import os, sys
import argparse
import shutil
import subprocess
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

from src.utils.train_util import instantiate_from_config


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="only resume model weights",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="base_config.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="experiment name",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes to use",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,",
        help="gpu ids to use",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="/logs",
        help="directory for logging data",
    )
    return parser


class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "project.yaml"))


class CodeSnapshot(Callback):
    """
    Modified from https://github.com/threestudio-project/threestudio/blob/main/threestudio/utils/callbacks.py#L60
    """
    def __init__(self, savedir):
        self.savedir = savedir

    def get_file_list(self):
        return [
            b.decode()
            for b in set(
                subprocess.check_output(
                    'git ls-files -- ":!:configs/*"', shell=True
                ).splitlines()
            )
            | set(  # hard code, TODO: use config to exclude folders or files
                subprocess.check_output(
                    "git ls-files --others --exclude-standard", shell=True
                ).splitlines()
            )
        ]

    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except:
            rank_zero_warn(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


if __name__ == "__main__":
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())

    # Early W&B mode enforcement: if WANDB_MODE explicitly 'online', clear offline markers
    if os.environ.get('WANDB_MODE', '').lower() == 'online':
        # Remove any offline marker file that could force offline behavior
        offline_marker = os.path.join(os.environ.get('WANDB_DIR', '.'), 'wandb', 'offline-run-*')
        # We won't glob-delete here to avoid accidental removal; rely on WANDB_MODE precedence.
        os.environ.pop('WANDB_DISABLED', None)
        os.environ['WANDB_START_METHOD'] = 'thread'

    # Add CUDA memory management and debugging
    import torch
    if torch.cuda.is_available():
        # Set memory allocation strategy for better debugging
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # Enable memory debugging
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        rank_zero_print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            rank_zero_print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            rank_zero_print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    cfg_fname = os.path.split(opt.base)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    exp_name = "-" + opt.name if opt.name != "" else ""
    logdir = os.path.join(opt.logdir, cfg_name+exp_name)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    codedir = os.path.join(logdir, "code")
    seed_everything(opt.seed)

    # init configs
    config = OmegaConf.load(opt.base)
    lightning_config = config.lightning
    trainer_config = lightning_config.trainer
    
    trainer_config["accelerator"] = "gpu"
    rank_zero_print(f"Running on GPUs {opt.gpus}")
    ngpu = len(opt.gpus.strip(",").split(','))
    trainer_config['devices'] = ngpu

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)
    if opt.resume and opt.resume_weights_only:
        model = model.__class__.load_from_checkpoint(opt.resume, **config.model.params)
    
    model.logdir = logdir

    # trainer and callbacks
    trainer_kwargs = dict()

    # logger
    # Create a serializable config for wandb (exclude complex objects)
    wandb_config = {}
    '''if hasattr(config, 'model') and hasattr(config.model, 'params'):
        model_params = config.model.params
        wandb_config.update({
            'learning_rate': getattr(config.model, 'base_learning_rate', None),
            'use_stable_keypoints': getattr(model_params, 'use_stable_keypoints', False),
            'num_learnable_tokens': getattr(model_params, 'num_learnable_tokens', 16),
            'drop_cond_prob': getattr(model_params, 'drop_cond_prob', 0.1),
        })
        
        # Add SK loss weights if they exist
        if hasattr(model_params, 'sk_loss_weights'):
            sk_weights = model_params.sk_loss_weights
            if isinstance(sk_weights, dict):
                for key, value in sk_weights.items():
                    wandb_config[f'sk_loss_weight_{key}'] = value
    
    if hasattr(config, 'data') and hasattr(config.data, 'params'):
        data_params = config.data.params
        wandb_config.update({
            'batch_size': getattr(data_params, 'batch_size', None),
            'num_workers': getattr(data_params, 'num_workers', None),
        })'''
    
    # Use dedicated wandb directory if available, otherwise use logdir
    wandb_save_dir = os.environ.get('WANDB_DIR', logdir)
    wandb_mode = os.environ.get('WANDB_MODE', '').lower()
    wandb_offline = wandb_mode == 'offline'  # online if not explicitly 'offline'

    # Force no resume to avoid extra GraphQL resume status call (helps with strict TLS)
    os.environ.setdefault('WANDB_RESUME', 'never')
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "name": f"superstable-keypoint-{opt.name}",
            "project": "superstable-keypoints",
            "save_dir": wandb_save_dir,
            "config": wandb_config,
            "tags": ["zero123plus", "stable-keypoints", "multi-view"],
            "offline": wandb_offline,
            "resume": "never",
        }
    }
    logger_cfg = OmegaConf.merge(default_logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # model checkpoint
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:08}",
            "verbose": True,
            "save_last": True,
            "every_n_train_steps": 5000,
            "save_top_k": -1,   # save all checkpoints
        }
    }

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # callbacks
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train.SetupCallback",
            "params": {
                "resume": opt.resume,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "code_snapshot": {
            "target": "train.CodeSnapshot",
            "params": {
                "savedir": codedir,
            }
        },
    }
    default_callbacks_cfg["checkpoint_callback"] = modelckpt_cfg

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    # Use 16-bit mixed precision to save memory
    trainer_kwargs['precision'] = '16-mixed'
    # Configure DDP strategy with memory optimizations
    trainer_kwargs["strategy"] = DDPStrategy(
        find_unused_parameters=True,
        gradient_as_bucket_view=True,  # Memory optimization
        static_graph=False  # Allow dynamic graphs for StableKeypoints
    )

    # trainer
    trainer = Trainer(**trainer_config, **trainer_kwargs, num_nodes=opt.num_nodes)
    trainer.logdir = logdir

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup("fit")

    # configure learning rate
    base_lr = config.model.base_learning_rate
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    model.learning_rate = base_lr
    rank_zero_print("++++ NOT USING LR SCALING ++++")
    rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")

    # run training loop
    if opt.resume and not opt.resume_weights_only:
        trainer.fit(model, data, ckpt_path=opt.resume)
    else:
        trainer.fit(model, data)