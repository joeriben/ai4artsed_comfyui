from .ai4artsed_ollama import ai4artsed_ollama
from .ai4artsed_ollama_imageanalysis import ai4artsed_ollama_imageanalysis
from .ai4artsed_openrouter import ai4artsed_openrouter
from .ai4artsed_openrouter_imageanalysis import ai4artsed_openrouter_imageanalysis
from .ai4artsed_text_remix import ai4artsed_text_remix
from .ai4artsed_random_instruction_generator import ai4artsed_random_instruction_generator
from .ai4artsed_random_artform_generator import ai4artsed_random_artform_generator
from .ai4artsed_random_language_selector import ai4artsed_random_language_selector
from .ai4artsed_openrouter_key import ai4artsed_openrouter_key

NODE_CLASS_MAPPINGS = {
    "ai4artsed_ollama": ai4artsed_ollama,
    "ai4artsed_ollama_imageanalysis": ai4artsed_ollama_imageanalysis,
    "ai4artsed_openrouter": ai4artsed_openrouter,
    "ai4artsed_openrouter_imageanalysis": ai4artsed_openrouter_imageanalysis,
    "ai4artsed_text_remix": ai4artsed_text_remix,
    "ai4artsed_random_instruction_generator": ai4artsed_random_instruction_generator,
    "ai4artsed_random_artform_generator": ai4artsed_random_artform_generator,
    "ai4artsed_random_language_selector": ai4artsed_random_language_selector,
    "ai4artsed_openrouter_key": ai4artsed_openrouter_key,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ai4artsed_ollama": "AI4ArtsEd Ollama Promptinterception",
    "ai4artsed_ollama_imageanalysis": "AI4ArtsEd Ollama Image Analysis",
    "ai4artsed_openrouter": "AI4ArtsEd OpenRouter Promptinterception",
    "ai4artsed_openrouter_imageanalysis": "AI4ArtsEd OpenRouter Image Analysis",
    "ai4artsed_text_remix": "AI4ArtsEd Text Remix",
    "ai4artsed_random_instruction_generator": "Random Instruction Generator",
    "ai4artsed_random_artform_generator": "Random Artform Generator",
    "ai4artsed_random_language_selector": "Random Language Selector",
    "ai4artsed_openrouter_key": "Secure Access to OpenRouter API Key",
}
