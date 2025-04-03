from typing import Dict, Optional, Tuple, Union, Any

from ..Networks.DragEmbedding import DragEmbedding
from ..Utils.pos_embed import get_2d_sincos_pos_embed
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils.torch_utils import is_torch_version
from diffusers.models.resnet import (
    Downsample2D,
    SpatioTemporalResBlock,
)
from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel


class CrossAttnDownBlockSpatioTemporalWithFlow(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        flow_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        add_downsample: bool = True,
        num_frames: int = 14,
        pos_embed_dim: int = 64,
        drag_token_cross_attn: bool = True,
        use_modulate: bool = True,
        drag_embedder_out_channels = (256, 320, 320),
        num_max_drags: int = 5,
    ):
        super().__init__()
        resnets = []
        attentions = []
        flow_convs = []
        if drag_token_cross_attn:
            drag_token_mlps = []
        self.num_max_drags = num_max_drags
        self.num_frames = num_frames
        self.pos_embed_dim = pos_embed_dim
        self.drag_token_cross_attn = drag_token_cross_attn

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.use_modulate = use_modulate
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-6,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                )
            )
            flow_convs.append(
                DragEmbedding(
                    conditioning_channels=flow_channels, 
                    conditioning_embedding_channels=out_channels * 2 if use_modulate else out_channels,
                    block_out_channels = drag_embedder_out_channels,
                )
            )
            if drag_token_cross_attn:
                drag_token_mlps.append(
                    nn.Sequential(
                        nn.Linear(pos_embed_dim * 2 + out_channels * 2, cross_attention_dim),
                        nn.SiLU(),
                        nn.Linear(cross_attention_dim, cross_attention_dim),
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.flow_convs = nn.ModuleList(flow_convs)
        if drag_token_cross_attn:
            self.drag_token_mlps = nn.ModuleList(drag_token_mlps)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=1,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.pos_embedding = {res: torch.tensor(get_2d_sincos_pos_embed(self.pos_embed_dim, res)) for res in [32, 16, 8, 4, 2]}
        self.pos_embedding_prepared = False

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        drag_original: Optional[torch.Tensor] = None,  # (batch_frame, num_points, 4)
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        batch_frame = hidden_states.shape[0]

        if self.drag_token_cross_attn:
            encoder_hidden_states_ori = encoder_hidden_states

        if not self.pos_embedding_prepared:
            for res in self.pos_embedding:
                self.pos_embedding[res] = self.pos_embedding[res].to(hidden_states)
            self.pos_embedding_prepared = True

        blocks = list(zip(self.resnets, self.attentions, self.flow_convs))
        for bid, (resnet, attn, flow_conv) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )

                if flow is not None:
                    # flow shape is (batch_frame, 40, h, w)
                    drags = flow.view(-1, self.num_frames, *flow.shape[1:])
                    drags = drags.chunk(self.num_max_drags, dim=2)  # (batch, frame, 4, h, w) x 10
                    drags = torch.stack(drags, dim=0)  # 10, batch, frame, 4, h, w
                    invalid_flag = torch.all(drags == -1, dim=(2, 3, 4, 5))
                    if self.use_modulate:
                        scale, shift = flow_conv(flow).chunk(2, dim=1)
                    else:
                        scale = 0
                        shift = flow_conv(flow)
                    hidden_states = hidden_states * (1 + scale) + shift
                    # print(self.drag_token_cross_attn)
                    if self.drag_token_cross_attn:
                        drag_token_mlp = self.drag_token_mlps[bid]
                        pos_embed = self.pos_embedding[scale.shape[-1]]
                        pos_embed = pos_embed.reshape(1, scale.shape[-1], scale.shape[-1], -1).permute(0, 3, 1, 2)
                        grid = (drag_original[..., :2] * 2 - 1)[:, None]
                        grid_end = (drag_original[..., 2:] * 2 - 1)[:, None]
                        drags_pos_start = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        drags_pos_end = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features = F.grid_sample(hidden_states.detach(), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features_end = F.grid_sample(hidden_states.detach(), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)

                        drag_token_in = torch.cat([features, features_end, drags_pos_start, drags_pos_end], dim=1).permute(0, 2, 1)
                        drag_token_out = drag_token_mlp(drag_token_in)
                        # Mask the invalid drags
                        drag_token_out = drag_token_out.view(batch_frame // self.num_frames, self.num_frames, self.num_max_drags, -1)
                        drag_token_out = drag_token_out.permute(2, 0, 1, 3)
                        drag_token_out = drag_token_out.masked_fill(invalid_flag[..., None, None].expand_as(drag_token_out), 0)
                        drag_token_out = drag_token_out.permute(1, 2, 0, 3).flatten(0, 1)
                        encoder_hidden_states = torch.cat([encoder_hidden_states_ori, drag_token_out], dim=1)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                if flow is not None:
                    # flow shape is (batch_frame, 40, h, w)
                    drags = flow.view(-1, self.num_frames, *flow.shape[1:])
                    drags = drags.chunk(self.num_max_drags, dim=2)  # (batch, frame, 4, h, w) x 10
                    drags = torch.stack(drags, dim=0)  # 10, batch, frame, 4, h, w
                    invalid_flag = torch.all(drags == -1, dim=(2, 3, 4, 5))
                    if self.use_modulate:
                        scale, shift = flow_conv(flow).chunk(2, dim=1)
                    else:
                        scale = 0
                        shift = flow_conv(flow)
                    hidden_states = hidden_states * (1 + scale) + shift
                    if self.drag_token_cross_attn:
                        drag_token_mlp = self.drag_token_mlps[bid]
                        pos_embed = self.pos_embedding[scale.shape[-1]]
                        pos_embed = pos_embed.reshape(1, scale.shape[-1], scale.shape[-1], -1).permute(0, 3, 1, 2)
                        grid = (drag_original[..., :2] * 2 - 1)[:, None]
                        grid_end = (drag_original[..., 2:] * 2 - 1)[:, None]
                        drags_pos_start = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        drags_pos_end = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features = F.grid_sample(hidden_states.detach(), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features_end = F.grid_sample(hidden_states.detach(), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)

                        drag_token_in = torch.cat([features, features_end, drags_pos_start, drags_pos_end], dim=1).permute(0, 2, 1)
                        drag_token_out = drag_token_mlp(drag_token_in)
                        # Mask the invalid drags
                        drag_token_out = drag_token_out.view(batch_frame // self.num_frames, self.num_frames, self.num_max_drags, -1)
                        drag_token_out = drag_token_out.permute(2, 0, 1, 3)
                        drag_token_out = drag_token_out.masked_fill(invalid_flag[..., None, None].expand_as(drag_token_out), 0)
                        drag_token_out = drag_token_out.permute(1, 2, 0, 3).flatten(0, 1)
                        encoder_hidden_states = torch.cat([encoder_hidden_states_ori, drag_token_out], dim=1)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states
