from .raunet import ApplyRAUNet, ApplyRAUNetSimple, UPSCALE_METHODS
from .msw_msa_attention import ApplyMSWMSAAttention, ApplyMSWMSAAttentionSimple
from .utils import *

NODE_CLASS_MAPPINGS = {
    "ApplyRAUNet": ApplyRAUNet,
    "ApplyRAUNetSimple": ApplyRAUNetSimple,
    "ApplyMSWMSAAttention": ApplyMSWMSAAttention,
    "ApplyMSWMSAAttentionSimple": ApplyMSWMSAAttentionSimple,
}

__all__ = ['ApplyRAUNet', 'ApplyRAUNetSimple', 'ApplyMSWMSAAttention', 'ApplyMSWMSAAttentionSimple', 'UPSCALE_METHODS']