import torch
from diffusers.models.attention_processor import XFormersAttnProcessor

from ..Networks.XFormersProcessor import AllToFirstXFormersAttnProcessor
from ..Networks.DragSpatioModel import UNetDragSpatioTemporalConditionModel
from ..config import cfg


def prepare_model():
    model = UNetDragSpatioTemporalConditionModel.from_pretrained(
        cfg.pretrained_model,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
        num_drags=cfg.num_drags,
        variant="fp16",
        torch_dtype=torch.float16,

        cond_dropout_prob=cfg.cond_dropout_prob,
        drag_token_cross_attn=cfg.drag_token_cross_attn,
        use_modulate=cfg.use_modulate,
        pos_embed_dim=cfg.pos_embed_dim,
        drag_embedder_out_channels=cfg.drag_embedder_out_channels,
    )  # type: UNetDragSpatioTemporalConditionModel

    if cfg.zero_init:
        model.zero_init()

    model.enable_xformers_memory_efficient_attention()

    if cfg.enable_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    # Set up all-to-first attention processors.
    attn_processors_dict={
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "mid_block.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "mid_block.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "mid_block.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "mid_block.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
    }

    model_attn_processor_dict = model.attn_processors
    for key in model_attn_processor_dict.keys():
        if key not in attn_processors_dict:
            attn_processors_dict[key] = model_attn_processor_dict[key]
    model.set_attn_processor(attn_processors_dict)

    return model
