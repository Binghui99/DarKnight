"""
tools/ocr_tool.py

Flow: image -> type-specific prompt -> output
"""
import re
import time
from typing import Optional
from utils.llm_client import get_client, chat_complete, extract_text, call_with_retry

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from utils.errors import ConnectionFailure, is_connection_error


OCR_ERROR_SENTINEL = "OCR_ERROR"


def _client(endpoint: str):
    return get_client(endpoint)


_PROMPTS = {
    "alphanumeric": (
        "Task: Recognize the CAPTCHA text in the image.\n"
        "1. If the image contains Chinese characters: each character is independent with no semantic connection between them.\n"
        "2. If the image contains English letters and/or digits: identify each character exactly as shown, preserving case (A-Z, a-z, 0-9).\n"
        "3. If the image shows a math equation (contains +, -, *, or =): compute and return ONLY the numeric answer.\n\n"
        "Rules:\n"
        "Output ONLY the final result. No explanation, no thinking, no spaces.\n"
        "No spaces in the result.\n"
        "Preserve the case exactly as shown.\n"
        "Do NOT add, repeat, or invent characters, output only what is visible.\n"
        "Do NOT duplicate any character unless it clearly appears twice in the image.\n"
        "Identify the text in the image:"
    ),
    "math": (
        "You are a CAPTCHA math extractor. Output ONLY the arithmetic expression.\n"
        "Rules:\n"
        "• Use standard operators: + - * / = ? and ONLY digits\n"
        "• No spaces, no explanation.\n"
        "• If this is not a math equation return exactly the text in the image.'\n"
        "What math expression is in this CAPTCHA? Output ONLY the expression."
    ),
    "chinese": (
        "You are a CAPTCHA recognition expert. Task: high-precision CAPTCHA recognition.\n"
        "Recognize each character in the image. The content consists of individual Chinese characters — "
        "each character is independent with no semantic connection between them (not words or phrases).\n"
        "Rules: Do NOT output any explanation, thinking process, or spaces. Output ONLY the final recognized characters.\n"
        "Identify the characters in the image:"
    ),
}




def run_ocr(image_b64: str,
            model: str,
            endpoint: str = "gpustack",
            captcha_type: str = "alphanumeric",
            char_count: int = None,
            hints: str = "",
            retries: int = 3,
            timeout: int = 100) -> str:
    """
    Call specific VLM to read CAPTCHA text.

    Returns the extracted text, or 'OCR_ERROR' on non-network failure.
    Raises ConnectionFailure if every retry failed with a network error.
    """
    if not image_b64.startswith("data:"):
        image_b64 = "data:image/png;base64," + image_b64

    t = (captcha_type or "alphanumeric").lower()
    prompt = _PROMPTS.get(t, _PROMPTS["alphanumeric"])
    if hints:
        prompt = hints + "\n" + prompt
    print(f"[OCR/{t}] prompt: {prompt!r}")
    client = _client(endpoint)

    _extra: dict = {}
    _max_tokens = 4096

    def _attempt(attempt: int) -> Optional[str]:
        extra_body = {"repetition_penalty": 1.2, **config.thinking_kwargs(model)} \
                     if endpoint != "anthropic" else None
        messages = [
            {"role": "system", "content": (
                "You are a precise, reliable, and concise assistant. Your primary goals are to provide information that is **strictly grounded** in your training data and the user-provided context, and to **avoid unnecessary repetition**.\n\n"
                "*   **Ground Your Answers:** Base your responses solely on:\n"
                "    *   Information from your pre-existing knowledge (up to your last training data).\n"
                "    *   **Explicit information provided by the user in this conversation.**\n"
                "    *   If specific sources or documents are provided, you must restrict your answers to those sources.\n"
                "*   **Acknowledge Limits:** If you do not know the answer or lack sufficient information, state this clearly. Do not invent facts, figures, quotes, citations, or details to fill gaps.\n"
                "*   **Avoid Speculation:** Do not speculate about events, data, or outcomes not supported by the available information.\n"
                "*   **Be Conceptually Concise:** Express each key point clearly once. Do not rephrase or restate the same idea in successive sentences unless specifically requested.\n"
                "*   **Monitor for Lexical Repetition:** Avoid using the same distinctive words or phrases repeatedly within a short span.\n"
                "*   **Stay On Topic:** Do not deviate from the user's core query with unsolicited background or tangents.\n\n"
                "Response Format Preference: Prioritize clarity and density of information. Favor straightforward language."
            )},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_b64}},
                {"type": "text", "text": prompt},
            ]},
        ]
        resp = chat_complete(
            client      = client,
            endpoint    = endpoint,
            model       = model,
            messages    = messages,
            max_tokens  = 4096,
            temperature = 0.1,
            extra_body  = extra_body,
        )
        print(resp)
        raw = extract_text(resp, endpoint, strip_spaces=True, single_line=True)
        print(f"[OCR/{t}] raw model output (attempt {attempt+1}): {raw!r}")
        text = raw
        if text:
            print(f"[OCR/{t}] cleaned → '{text}'")
            return text
        print(f"[OCR/{t}] returned empty — raw was: {raw!r}")
        return None

    def _on_exception(attempt: int, exc: Exception):
        print(f"[OCR] attempt {attempt + 1}/{retries} failed: {exc}")

    result = call_with_retry(
        _attempt,
        retries,
        on_exception=_on_exception,
        connection_failure_msg=lambda last_exc: (
            f"[OCR] all {retries} retries failed with connection errors: {last_exc}"
        ),
    )
    return result if result is not None else OCR_ERROR_SENTINEL
