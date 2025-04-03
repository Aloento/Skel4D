from typing import Tuple

from ..Utils.pos_embed import zero_module
import torch.nn as nn
import torch.nn.functional as F

class DragEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,  # out channel
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        conditioning_ndims = len(conditioning.shape)
        if conditioning_ndims == 5:
            batch_size, num_frames, num_channels, h, w = conditioning.shape
            conditioning = conditioning.flatten(0, 1)

        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)
        if conditioning_ndims == 5:
            embedding = embedding.view(batch_size, num_frames, *embedding.shape[1:])

        return embedding
   