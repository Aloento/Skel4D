"""
Attention control mechanisms for Stable Diffusion
"""

import abc
import torch
import torch.nn.functional as F


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, dict, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, dict, is_cross: bool, place_in_unet: str):
        dict = self.forward(dict, is_cross, place_in_unet)
        return dict['attn']

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "attn": [],
        }

    def forward(self, dict, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 32**2:  # avoid memory overhead
        self.step_store["attn"].append(dict['attn'])
        return dict

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()


def register_attention_control(model, controller, feature_upsample_res=256):
    """Register attention control hooks in the model"""
    
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None

            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim = sim.masked_fill(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = torch.nn.Softmax(dim=-1)(sim)
            attn = attn.clone()

            out = torch.matmul(attn, v)

            if (
                is_cross
                and sequence_length <= 32**2
                and len(controller.step_store["attn"]) < 4
            ):
                x_reshaped = x.reshape(
                    batch_size,
                    int(sequence_length**0.5),
                    int(sequence_length**0.5),
                    dim,
                ).permute(0, 3, 1, 2)
                # upsample to feature_upsample_res**2
                x_reshaped = (
                    F.interpolate(
                        x_reshaped,
                        size=(feature_upsample_res, feature_upsample_res),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(batch_size, -1, dim)
                )

                q = self.to_q(x_reshaped)
                q = self.reshape_heads_to_batch_dim(q)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                attn = torch.nn.Softmax(dim=-1)(sim)
                attn = attn.clone()

                attn = controller({"attn": attn}, is_cross, place_in_unet)

            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "CrossAttention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")

    controller.num_att_layers = cross_att_count

    # create assertion with message
    assert cross_att_count != 0, "No cross attention layers found in the model. Please check to make sure you're using diffusers==0.8.0."
