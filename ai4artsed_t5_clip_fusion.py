import torch
from comfy.model_patcher import ModelPatcher

"""
ai4artsed_t5_clip_fusion.py
---------------------------
This node is opensource provided by the ai4artsed project.
Conceptualization: Benjamin Jörissen, https://github.com/joeriben, coding: ChatGPT o3

**Purpose**
    Combine the stylistic precision of a 77‑token CLIP conditioning with the
    semantic depth of a long‑context T5 encoder without changing the expected
    token length of Stable‑Diffusion’s UNet.

How it differs from *Triple CLIP*:
    • *Triple CLIP* runs **three** CLIP text encoders and **averages** their 77×768
      outputs.  All content beyond the first 77 tokens is discarded during
      CLIP tokenisation.

    • *ai4artsed_t5_clip_fusion* keeps **one** CLIP conditioning (any 77×768
      tensor) **and** injects additional semantics from **all** T5 tokens
      (≤ 512).  The T5 sequence is **mean‑pooled to a single 768‑d vector**,
      L2‑normalised and then broadcast‑added onto **each of the 77 CLIP token
      rows**.

Token handling
    • The first 77 tokens of the *Prompt‑A* (CLIP) remain unmodified except for
      the additive shift.
    • The remaining tokens of *Prompt‑B* (T5) are not appended as extra rows;   
      their information is condensed into the pooled vector.

Alpha parameter
    • `alpha` (0‒1) scales **only the pooled T5 vector** before it is added.
      Setting α=0 reproduces vanilla CLIP; α=1 gives full T5 influence.

No training required
    • Operations are arithmetic (mean, norm, add).  No weights, no fine‑tuning.
Category
    • Registered under *"ai4artsed"* in ComfyUI’s node tree.
"""

NODE_NAME = "AI4ArtsEd T5‑CLIP Fusion"

class AI4ArtsEdT5ClipFusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_cond": ("CONDITIONING",),
                "t5_embeds": ("EMBED",),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "fuse"
    CATEGORY = "AI4ArtsEd"

    def fuse(self, clip_cond, t5_embeds, alpha):
        # clip_cond is a tuple (tokens, pooled)  where tokens: (B,77,768)
        clip_tokens, clip_pooled = clip_cond

        # t5_embeds shape: (B,Seq,768)
        # Mean‑pool along sequence dimension
        t5_vec = t5_embeds.mean(dim=1)  # (B,768)
        # L2‑norm
        t5_vec = t5_vec / (t5_vec.norm(dim=-1, keepdim=True) + 1e-8)
        # Broadcast to (B,77,768)
        t5_tokens = t5_vec.unsqueeze(1).repeat(1, clip_tokens.size(1), 1)

        fused_tokens = clip_tokens + alpha * t5_tokens
        return ((fused_tokens, clip_pooled),)


# ---- ComfyUI registration ---------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "AI4ArtsEdT5ClipFusion": AI4ArtsEdT5ClipFusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AI4ArtsEdT5ClipFusion": NODE_NAME,
}
