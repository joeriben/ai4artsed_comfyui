import torch
from comfy.model_patcher import ModelPatcher

"""
ai4artsed_t5_clip_fusion.py
---------------------------
This node is opensource provided by the ai4artsed project.
Conceptualization: Benjamin Jörissen, https://github.com/joeriben, coding: ChatGPT o3

**Purpose**
    Combine the stylistic precision of CLIP conditioning (from CLIP-L and CLIP-G) 
    with the semantic depth of a long‑context T5 encoder without changing the 
    expected token length of Stable‑Diffusion's UNet.

Workflow:
    1. A prompt is manipulated by two intercept nodes:
       - Node 1 optimizes the prompt for CLIP-L and CLIP-G
       - Node 2 manipulates the prompt for T5
    2. The prompts are embedded separately:
       - One with CLIP (using CLIP Text Encode or DUAL-CLIP loader)
       - One with T5 (using T5 Text Encode)
    3. This fusion node combines both embeddings with an alpha parameter to weight the T5 contribution

How it works:
    • Takes CLIP conditioning from any CLIP encoder (CLIP-L, CLIP-G, or DUAL-CLIP)
    • Takes T5 conditioning from a CLIPTextEncode node using a T5 model
    • Handles various input formats robustly:
      - Standard CONDITIONING format (tuple or list of tokens and pooled)
      - Single tensor format
      - Single-element tuple or list containing a tensor
      - Nested list structures from ComfyUI
    • The T5 tokens are **mean‑pooled to a single 768‑d vector**,
      L2‑normalised and then broadcast‑added onto **each of the 77 CLIP token
      rows**.

Token handling
    • The CLIP tokens (77×768) remain unmodified except for the additive shift.
    • The T5 tokens are not appended as extra rows; their information is 
      condensed into the pooled vector.

Alpha parameter
    • `alpha` (0‒1) scales **only the pooled T5 vector** before it is added.
      Setting α=0 reproduces vanilla CLIP; α=1 gives full T5 influence.

No training required
    • Operations are arithmetic (mean, norm, add). No weights, no fine‑tuning.
Category
    • Registered under *"AI4ArtsEd"* in ComfyUI's node tree.
"""

NODE_NAME = "AI4ArtsEd T5‑CLIP Fusion"

class ai4artsed_t5_clip_fusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_conditioning": ("CONDITIONING",),
                "t5_conditioning": ("CONDITIONING",),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "fuse"
    CATEGORY = "AI4ArtsEd"

    def fuse(self, clip_conditioning, t5_conditioning, alpha):
        # Debug information
        print("DEBUG INFO FOR ai4artsed_t5_clip_fusion:")
        print(f"clip_conditioning type: {type(clip_conditioning)}")
        print(f"clip_conditioning length: {len(clip_conditioning) if hasattr(clip_conditioning, '__len__') else 'N/A'}")
        
        # Extract clip tokens and pooled output
        clip_tokens = None
        clip_pooled = None
        
        try:
            # Handle nested list structure: [[tensor, dict]]
            if isinstance(clip_conditioning, list) and len(clip_conditioning) == 1:
                if isinstance(clip_conditioning[0], list) and len(clip_conditioning[0]) == 2:
                    inner_list = clip_conditioning[0]
                    if isinstance(inner_list[0], torch.Tensor) and isinstance(inner_list[1], dict):
                        clip_tokens = inner_list[0]
                        if 'pooled_output' in inner_list[1]:
                            clip_pooled = inner_list[1]['pooled_output']
                        else:
                            # Create a default pooled output if not available
                            clip_pooled = torch.zeros(1, clip_tokens.size(-1), dtype=clip_tokens.dtype, device=clip_tokens.device)
            
            # Handle standard tuple/list structure: (tensor, tensor)
            elif isinstance(clip_conditioning, (tuple, list)) and len(clip_conditioning) >= 2:
                clip_tokens, clip_pooled = clip_conditioning[0], clip_conditioning[1]
            
            # Handle single tensor
            elif isinstance(clip_conditioning, torch.Tensor):
                clip_tokens = clip_conditioning
                clip_pooled = torch.zeros(1, clip_tokens.size(-1), dtype=clip_tokens.dtype, device=clip_tokens.device)
            
            # If we couldn't extract clip_tokens, raise an error
            if clip_tokens is None:
                raise ValueError(f"Could not extract tensor from clip_conditioning: {type(clip_conditioning)}")
            
            print(f"Successfully extracted clip_tokens with shape: {clip_tokens.shape}")
            if clip_pooled is not None:
                print(f"Successfully extracted clip_pooled with shape: {clip_pooled.shape}")
            
            # Extract t5 embeddings
            t5_embeds = None
            
            # Handle nested list structure: [[tensor, dict]]
            if isinstance(t5_conditioning, list) and len(t5_conditioning) == 1:
                if isinstance(t5_conditioning[0], list) and len(t5_conditioning[0]) == 2:
                    inner_list = t5_conditioning[0]
                    if isinstance(inner_list[0], torch.Tensor):
                        t5_embeds = inner_list[0]
            
            # Handle standard tuple/list structure: (tensor, tensor)
            elif isinstance(t5_conditioning, (tuple, list)) and len(t5_conditioning) >= 2:
                t5_embeds = t5_conditioning[0]
            
            # Handle single tensor
            elif isinstance(t5_conditioning, torch.Tensor):
                t5_embeds = t5_conditioning
            
            # Handle single-element tuple/list containing a tensor
            elif isinstance(t5_conditioning, (tuple, list)) and len(t5_conditioning) == 1:
                if isinstance(t5_conditioning[0], torch.Tensor):
                    t5_embeds = t5_conditioning[0]
            
            # If we couldn't extract t5_embeds, raise an error
            if t5_embeds is None:
                raise ValueError(f"Could not extract tensor from t5_conditioning: {type(t5_conditioning)}")
            
            # Ensure t5_embeds is a tensor with the right shape
            if not isinstance(t5_embeds, torch.Tensor):
                raise ValueError(f"t5_embeds must be a tensor, got {type(t5_embeds)}")
            
            print(f"Successfully extracted t5_embeds with shape: {t5_embeds.shape}")
            
            # t5_embeds shape should be (B,Seq,768) where Seq could be 77 for CLIP or another length for T5
            # Mean‑pool along sequence dimension (token dimension)
            t5_vec = t5_embeds.mean(dim=1)  # (B,768)
            
            # L2‑norm
            t5_vec = t5_vec / (t5_vec.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Broadcast to (B,77,768)
            t5_tokens = t5_vec.unsqueeze(1).repeat(1, clip_tokens.size(1), 1)

            # Fuse the embeddings with alpha weighting
            fused_tokens = clip_tokens + alpha * t5_tokens
            
            # Return the fused conditioning in the same format as the input
            if isinstance(clip_conditioning, list) and len(clip_conditioning) == 1 and isinstance(clip_conditioning[0], list):
                # Return in nested list format
                result = [[fused_tokens, clip_conditioning[0][1]]]
                print(f"Returning result in nested list format: {type(result)}")
                return (result,)
            else:
                # Return in standard tuple format
                result = (fused_tokens, clip_pooled)
                print(f"Returning result in standard tuple format: {type(result)}")
                return (result,)
            
        except Exception as e:
            print(f"ERROR in ai4artsed_t5_clip_fusion: {str(e)}")
            # Return original conditioning as fallback
            return (clip_conditioning,)


# ---- ComfyUI registration ---------------------------------------------------

NODE_CLASSES = {
    "ai4artsed_t5_clip_fusion": ai4artsed_t5_clip_fusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ai4artsed_t5_clip_fusion": NODE_NAME,
}
