"""
utils/llm_client.py
Abstraction over OpenAI-compatible and Anthropic APIs.

Planner and OCR both call chat_complete() endpoint routing happens here.
"""
import base64
import re
import sys
import os
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from utils.errors import ConnectionFailure, is_connection_error



def get_client(endpoint: str):
    if endpoint == "anthropic":
        from anthropic import Anthropic
        return Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            base_url=config.ANTHROPIC_BASE_URL,
        )
    from openai import OpenAI
    return OpenAI(base_url=config.GPUSTACK_BASE_URL, api_key=config.GPUSTACK_API_KEY)



def _detect_media_type(b64_data: str, declared: str) -> str:
    """Detect actual image format"""
    try:
        header = base64.b64decode(b64_data[:16])
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        if header[:3] == b'\xff\xd8\xff':
            return "image/jpeg"
        if header[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return "image/webp"
    except Exception:
        pass
    return declared


def _openai_image_to_anthropic(content_block: dict) -> dict:
    """Convert an OpenAI image_url block to an Anthropic image block."""
    url = content_block.get("image_url", {}).get("url", "")
    if url.startswith("data:"):
        header, data = url.split(",", 1)
        declared = header[5:].split(";")[0]
        media_type = _detect_media_type(data, declared)
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _convert_messages_for_anthropic(messages: list) -> tuple:
    """
    Split OpenAI-style messages into (system_str, user_messages).
    Converts image_url blocks to Anthropic image blocks along the way.
    """
    system = None
    converted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system = content if isinstance(content, str) else str(content)
            continue

        if isinstance(content, list):
            new_content = []
            for block in content:
                if block.get("type") == "image_url":
                    new_content.append(_openai_image_to_anthropic(block))
                else:
                    new_content.append(block)
            content = new_content

        converted.append({"role": role, "content": content})

    return system, converted


def chat_complete(
    client,
    endpoint: str,
    model: str,
    messages: list,
    max_tokens: int = None,
    temperature: float = None,
    extra_body: dict = None,
    response_format: dict = None,
) -> object:
    """
    Call the right API based on endpoint and return the raw response object.
    """
    if endpoint == "anthropic":
        system, converted = _convert_messages_for_anthropic(messages)
        kwargs = dict(
            model=model,
            messages=converted,
            max_tokens=max_tokens or 4096,
        )
        if system:
            kwargs["system"] = system
        if temperature is not None and model not in config.NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = temperature
        return client.messages.create(**kwargs)

    # OpenAI-compatible path
    kwargs = dict(model=model, messages=messages)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_body:
        kwargs["extra_body"] = extra_body
    if response_format is not None:
        kwargs["response_format"] = response_format
    return client.chat.completions.create(**kwargs)


def extract_text(resp, endpoint: str, strip_spaces: bool = False, single_line: bool = False) -> str:
    """
    Pull text from whichever response shape was returned.

    Handles:
      - Anthropic: resp.content list of TextBlock / ThinkingBlock
      - OpenAI-compatible: resp.choices[0].message.content (str or list)
      - step3-style models: content=None, answer in reasoning_content

    """
    if endpoint == "anthropic":
        # Anthropic returns a list of content blocks; skip ThinkingBlocks
        text = ""
        for block in (resp.content or []):
            block_type = getattr(block, "type", None)
            if block_type == "thinking":
                continue
            block_text = getattr(block, "text", None)
            if block_text:
                text = block_text
                break
    else:
        msg = resp.choices[0].message if (resp.choices) else None
        content = getattr(msg, "content", None) if msg else None

        if isinstance(content, list):
            # Some models return list of blocks (thinking + text)
            parts = []
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "thinking":
                        continue
                    parts.append(b.get("text", ""))
                else:
                    parts.append(str(b))
            text = "".join(parts).strip()
        elif isinstance(content, str):
            text = content
        else:
            # try reasoning_content if content=None (step3-vl-10b style)
            reasoning = getattr(msg, "reasoning_content", None) or "" if msg else ""
            text = reasoning

    # Strip <think>...</think> blocks and dangling </think>
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if single_line and "\n" in text:
        last = next((l.strip() for l in reversed(text.splitlines()) if l.strip()), text)
        text = last

    if strip_spaces:
        text = text.replace(" ", "")
    return text


# Retry loop

def call_with_retry(
    attempt_fn: Callable[[int], Optional[str]],
    retries: int,
    on_exception: Optional[Callable[[int, Exception], None]] = None,
    connection_failure_msg: Optional[Callable[[Exception], str]] = None,
) -> Optional[str]:
    """
    Shared retry/backoff/connection-classification shell 
    """
    last_exc: Optional[Exception] = None
    all_connection = True
    for attempt in range(retries):
        try:
            result = attempt_fn(attempt)
            if result is not None:
                return result
            # attempt_fn returned None without raising (e.g. empty OCR text)
            # original call sites fall through to the next attempt WITHOUT
            # sleeping in this case, so do not sleep here either.
        except Exception as exc:
            last_exc = exc
            if not is_connection_error(exc):
                all_connection = False
            if on_exception:
                on_exception(attempt, exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    if all_connection and last_exc is not None:
        msg = connection_failure_msg(last_exc) if connection_failure_msg else str(last_exc)
        raise ConnectionFailure(msg)
    return None
