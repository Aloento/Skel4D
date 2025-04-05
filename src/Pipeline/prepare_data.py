from torch.utils.data import DataLoader, DistributedSampler

from ..dataset import DragVideoDataset
from ..config import cfg


def get_generator(loader):
    while True:
        for batch in loader:
            yield batch


def prepare_data():
    dataset_train = DragVideoDataset(
        h5_path=cfg.h5_path,
        num_max_drags=cfg.num_max_drags,
        num_frames=cfg.num_frames,
    )

    if cfg.distributed:
        train_sampler = DistributedSampler(
            dataset_train,
            num_replicas=cfg.world_size,
            rank=cfg.global_rank,
            shuffle=True,
        )
    else:
        train_sampler = None

    loader = DataLoader(
        dataset_train,
        batch_size=4,
        shuffle=not cfg.distributed,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
    )

    if cfg.main_process:
        val_generator = get_generator(val_loader)
    else:
        val_generator = None

    return loader, val_loader, val_generator
