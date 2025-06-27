from ..Networks.CrossAttnDown import CrossAttnDownBlockSpatioTemporalWithFlow
from ..Networks.CrossAttnUp import CrossAttnUpBlockSpatioTemporalWithFlow
from diffusers.models.unets.unet_3d_blocks import (
    get_down_block as gdb, 
    get_up_block as gub,
)


def get_down_block(
    with_concatenated_flow: bool = False,
    *args,
    **kwargs,
):
    NEEDED_KEYS = [
        "in_channels",
        "out_channels",
        "temb_channels",
        "flow_channels",
        "num_layers",
        "transformer_layers_per_block",
        "num_attention_heads",
        "cross_attention_dim",
        "add_downsample",
        "pos_embed_dim",
        'use_modulate',
        "drag_token_cross_attn",
        "drag_embedder_out_channels",
        "num_max_drags",
    ]
    if not with_concatenated_flow or args[0] == "DownBlockSpatioTemporal":
        kwargs.pop("flow_channels", None)
        kwargs.pop("pos_embed_dim", None)
        kwargs.pop("use_modulate", None)
        kwargs.pop("drag_token_cross_attn", None)
        kwargs.pop("drag_embedder_out_channels", None)
        kwargs.pop("num_max_drags", None)
        return gdb(*args, **kwargs)
    elif args[0] == "CrossAttnDownBlockSpatioTemporal":
        for key in list(kwargs.keys()):
            if key not in NEEDED_KEYS:
                kwargs.pop(key, None)
        return CrossAttnDownBlockSpatioTemporalWithFlow(*args[1:], **kwargs)
    else:
        raise ValueError(f"Unknown block type {args[0]}")
    

def get_up_block(
    with_concatenated_flow: bool = False,
    *args,
    **kwargs,
):
    NEEDED_KEYS = [
        "in_channels",
        "out_channels",
        "prev_output_channel",
        "temb_channels",
        "flow_channels",
        "resolution_idx",
        "num_layers",
        "transformer_layers_per_block",
        "resnet_eps",
        "num_attention_heads",
        "cross_attention_dim",
        "add_upsample",
        "pos_embed_dim",
        "use_modulate",
        "drag_token_cross_attn",
        "drag_embedder_out_channels",
        "num_max_drags",
    ]
    if not with_concatenated_flow or args[0] == "UpBlockSpatioTemporal":
        kwargs.pop("flow_channels", None)
        kwargs.pop("pos_embed_dim", None)
        kwargs.pop("use_modulate", None)
        kwargs.pop("drag_token_cross_attn", None)
        kwargs.pop("drag_embedder_out_channels", None)
        kwargs.pop("num_max_drags", None)
        return gub(*args, **kwargs)
    elif args[0] == "CrossAttnUpBlockSpatioTemporal":
        for key in list(kwargs.keys()):
            if key not in NEEDED_KEYS:
                kwargs.pop(key, None)
        return CrossAttnUpBlockSpatioTemporalWithFlow(*args[1:], **kwargs)
    else:
        raise ValueError(f"Unknown block type {args[0]}")
