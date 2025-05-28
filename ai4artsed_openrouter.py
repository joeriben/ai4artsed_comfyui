import os
import requests
import json

class ai4artsed_openrouter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_prompt": ("STRING", {"forceInput": True, "multiline": True}),
                "input_context": ("STRING", {
                    "default": "Input CONTEXT here",
                    "multiline": True
                 }),
                "style_prompt": ("STRING", {
                    "default": "Translate the prompt according to the context. Translate epistemic, cultural, and aesthetic, as well as value-related contexts.",
                    "multiline": True
                }),
                "api_key": ("STRING", {"multiline": False, "password": True}),
                "model": ([
                    "anthropic/claude-sonnet-4", #200K context $3/M input tokens $15/M output tokens $4.80/K input img
                    "deepseek/deepseek-chat-v3-0324", #164K context $ 0.30/M input tokens $0.88/M output tokens
                    "deepseek/deepseek-r1", #164K context $0.50/M input tokens $2.18/M output tokens
                    "google/gemini-2.5-pro-preview", #1.05M context $1.25/M input tokens $10/M output tokens $5.16/K input imgs
                    "meta-llama/llama-3.3-70b-instruct", #131K context $0.07/M input tokens $0.25/M output tokens
                    "meta-llama/llama-guard-3-8b", #131K context $0.02/M input tokens $0.06/M output tokens
                    "mistralai/mistral-medium-3", #131,072 context $0.40/M input tokens $2/M output tokens
                    "mistralai/mistral-7b-instruct", #33K context$0.028/M input tokens $0.054/M output tokens
                    "openai/o3", #200K context $10/M input tokens $40/M output tokens$7.65/K input imgs
                ],),
                "debug": (["enable", "disable"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "AI4ArtsEd"

    def get_api_key(self, user_input_key):
        if user_input_key.strip():
            return user_input_key.strip()

        # Suche im selben Verzeichnis wie dieses Skript
        key_path = os.path.join(os.path.dirname(__file__), "openrouter.key")
        try:
            with open(key_path, "r") as f:
                return f.read().strip()
        except Exception:
            raise Exception("[AI4ArtsEd OpenRouter Node] No API key provided and openrouter.key not found.")

    def run(self, input_prompt, input_context, style_prompt, api_key, model, debug):
        full_prompt = f"Task:\n{style_prompt.strip()}\n\nContext:\n{input_context.strip()}\nPrompt:\n{input_prompt.strip()}"

        messages = [
            {"role": "system", "content": "You are a fresh assistant instance. Forget all previous conversation history."},
            {"role": "user", "content": full_prompt}
        ]

        headers = {
            "Authorization": f"Bearer {self.get_api_key(api_key)}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7
        }

        url = "https://openrouter.ai/api/v1/chat/completions"
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            raise Exception(f"[AI4ArtsEd OpenRouter Node] API Error: {response.status_code}\n{response.text}")

        result = response.json()
        output_text = result["choices"][0]["message"]["content"]

        if debug == "enable":
            print(">>> AI4ARTSED OPENROUTER NODE <<<")
            print("Model:", model)
            print("Prompt sent:\n", full_prompt)
            print("Response received:\n", output_text)

        return (output_text,)
