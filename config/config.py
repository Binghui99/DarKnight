"""
config.py - Configuration for Dark Knight
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Model Endpoints
# Set these in .env — see README.md "Setup" for the required keys.

# GPUStack endpoint (OpenAI-compatible)
GPUSTACK_BASE_URL = os.getenv("GPUSTACK_BASE_URL")
GPUSTACK_API_KEY  = os.getenv("ZIYI_CAPTCHA_API_KEY")

# Anthropic via Azure APIM
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
ANTHROPIC_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")

# Models (used as BOTH planner and OCR agent.
MODELS = [
    # {
    #     "label": "hunyuan-ocr",
    #     "model": "hunyuanocr",
    #     "endpoint": "gpustack",
    # },
    # {
    #     "label": "qwen3-vl-4b-instruct",
    #     "model": "qwen3-vl-4b-instruct",
    #     "endpoint": "gpustack",
    # },
    # {
    #     "label": "qwen3-vl-8b-instruct",
    #     "model": "qwen3-vl-8b-instruct",
    #     "endpoint": "gpustack",
    # },
    # {
    #     "label": "step3-vl-10b",
    #     "model": "step3-vl-10b",
    #     "endpoint": "gpustack",
    # },
    # {
    #     "label": "gemma4-e4b-it",
    #     "model": "gemma-4-e4b-it",
    #     "endpoint": "gpustack",
    # },
    # {
    #     "label": "qwen3.5-9b",
    #     "model": "qwen3.5-9b",
    #     "endpoint": "gpustack",
    # },
    {
        "label": "qwen3-vl-30b-a3b-instruct",
        "model": "qwen3-vl-30b-a3b-instruct",
        "endpoint": "gpustack",
    },
    # {
    #     "label": "huihui-thinkingcap-qwen3.6-27b-abliterated",
    #     "model": "huihui-thinkingcap-qwen3.6-27b-abliterated",
    #     "endpoint": "gpustack",
    # },
    # {
    #     "label": "gemma4-12b-it",
    #     "model": "gemma4-12b-it",
    #     "endpoint": "gpustack",
    # },
    # {
    #     "label": "claude-haiku-4-5",
    #     "model": "claude-haiku-4-5",
    #     "endpoint": "anthropic",
    # },
    # {
    #     "label": "claude-sonnet-4-6",
    #     "model": "claude-sonnet-4-6",
    #     "endpoint": "anthropic",
    # },
    # {
    #     "label": "claude-opus-4-6",
    #     "model": "claude-opus-4-6",
    #     "endpoint": "anthropic",
    # },
    # {
    #     "label": "claude-opus-5",
    #     "model": "claude-opus-5",
    #     "endpoint": "anthropic",
    # },
]

# RAG Configuration 
RAG_TOP_K                = 5
RAG_DB_PATH              = "rag_store/rag_db.json"
RAG_CAPACITY             = 100  # TOTAL rag capacity(50 for failed, 50 for success)
EMBEDDING_MODEL          = "all-MiniLM-L6-v2"
RAG_SIMILARITY_THRESHOLD        = 0.5   
RAG_FAILED_SIMILARITY_THRESHOLD = 0.80  #only trust very similar failures

# Pipeline Behavior 
MAX_RETRIES       = 3
REQUEST_TIMEOUT   = 60
LOG_INTERMEDIATES = True
ENABLE_THINKING   = False  

THINKING_MODELS = {
    "step3-vl-10b",
    "qwen3-vl-4b-instruct",
    "qwen3-vl-8b-instruct",
    "qwen3.5-9b",
    "qwen3.5-122b-a10b-copy3",
    "huihui-thinkingcap-qwen3.6-27b-abliterated",
    "huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated",
}

# Claude models reject the temperature parameter with a 400 error
NO_TEMPERATURE_MODELS = {
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-haiku-5",
}

def thinking_kwargs(model: str) -> dict:
    """Return chat_template_kwargs for thinking-capable models, empty dict otherwise."""
    if model in THINKING_MODELS:
        return {"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}}
    return {}

# Paths 
DATASETS_DIR = "datasets"
RESULTS_DIR  = "results"
LOGS_DIR     = "logs"
