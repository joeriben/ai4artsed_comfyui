import os
from pathlib import Path


class ai4artsed_openrouter_key:
    """Securely provides the Stability AI API key to downstream nodes.

    The key is read **only** from a text file named ``stabilityai.key`` that
    lives in the same directory as this node file.  The first non‑empty line
    of that file is treated as the key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_key"
    CATEGORY = "AI4ArtsEd"

    def get_key(self):
        key_path = Path(__file__).with_name("stabilityai.key")
        if not key_path.exists():
            raise RuntimeError("Missing 'stabilityai.key' in node directory.")

        with key_path.open("r", encoding="utf-8") as f:
            key_line = f.readline().strip()

        if not key_line or not key_line.startswith("sk-or-"):
            raise RuntimeError("Invalid or empty key in 'stabilityai.key'.")

        return (key_line,)


NODE_CLASS_MAPPINGS = {
    "ai4artsed_stabilitai_key": ai4artsed_openrouter_key
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ai4artsed_stabilitai_key": "Secure Access to Stability AI API Key"
}
