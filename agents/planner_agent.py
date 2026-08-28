"""
agents/planner_agent.py
Planner Agent for Dark Knight.

Responsibilities:
  Part 1. Visual Analysis:
    Pass the raw CAPTCHA and extract structured visual features.
  Part 2. CoT Reasoning (with optional RAG hints):
    Reason independently from the visual features.
    RAG/MCP records are retrived only when similarity >= threshold, and are
    framed as soft hints.
  Part 3. Action Pipeline:
    Generate a structured JSON pipeline with ordered tool steps + parameters.
"""
import json
import re
from typing import Dict, Any, List, Optional, Tuple

from utils.llm_client import get_client, chat_complete, extract_text

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from tools.registry import list_tools, normalize_tool_name
from utils.logger import PipelineLogger
from utils.errors import ConnectionFailure, is_connection_error
from utils.llm_client import call_with_retry
from agents.planner_prompts import VISUAL_ANALYSIS_SYSTEM, VISUAL_ANALYSIS_USER, COT_SYSTEM


class PlannerAgent:
    """
    Wraps a VLM to perform visual analysis + independent CoT planning.
    RAG/MCP records are passed as scored hints; the model reasons for itself.
    """

    _IMAGE_TOOLS = {
        "color_extract",
        "sharpen",
        "remove_small_noise",
        "remove_straight_line",
        "binarize",
    }

    def __init__(self, model: str, endpoint: str = "ollama", label: str = "",
                 logger: Optional[PipelineLogger] = None):
        self.model    = model
        self.endpoint = endpoint
        self.label    = label or model
        self._client  = get_client(endpoint)
        self._logger  = logger

    # Internal helpers
    def _chat(self, messages: List[Dict], max_tokens: int = None,
              response_format: Optional[Dict] = None,
              retries: int = None) -> str:
        retries = retries or config.MAX_RETRIES
        extra_body = config.thinking_kwargs(self.model) if self.endpoint != "anthropic" else None

        def _attempt(attempt: int) -> str:
            resp = chat_complete(
                client         = self._client,
                endpoint       = self.endpoint,
                model          = self.model,
                messages       = messages,
                max_tokens     = max_tokens,
                extra_body     = extra_body or None,
                response_format= response_format,
            )
            print(resp)
            return extract_text(resp, self.endpoint)

        def _on_exception(attempt: int, exc: Exception):
            print(f"[Planner/{self.label}] attempt {attempt+1} failed: {exc}")

        result = call_with_retry(
            _attempt,
            retries,
            on_exception=_on_exception,
            connection_failure_msg=lambda last_exc: (
                f"[Planner/{self.label}] all {retries} retries failed "
                f"with connection errors: {last_exc}"
            ),
        )
        return result if result is not None else "{}"

    def _parse_json(self, text: str) -> Dict:
        cleaned = text.strip()
        for fence in ("```json", "```"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence):]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return {}

    def _normalize_plan(self, plan,
                        features: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(plan, list):
            plan = {"preprocess_pipeline": plan}
        preprocess_pipeline = plan.get("preprocess_pipeline")
        if preprocess_pipeline is None:
            preprocess_pipeline = plan.get("pipeline", [])
        if not isinstance(preprocess_pipeline, list):
            preprocess_pipeline = []

        normalized_steps = []
        for idx, step in enumerate(preprocess_pipeline, start=1):
            if not isinstance(step, dict):
                continue
            tool = normalize_tool_name(str(step.get("tool", "")))
            if not tool or tool not in self._IMAGE_TOOLS:
                continue
            params = step.get("params", {})
            if not isinstance(params, dict):
                params = {}
            normalized_steps.append({"step": idx, "tool": tool, "params": params})

        captcha_type = features.get("captcha_type", "alphanumeric")
        is_math      = features.get("is_math", False) or captcha_type == "math"
        is_chinese   = features.get("is_chinese", False) or captcha_type == "chinese"

        return {
            "reasoning":           plan.get("reasoning", ""),
            "preprocess_pipeline": normalized_steps,
            "ocr":    {"enabled": True},
            "post_ocr": {
                "detect_math_equation": is_math,
                "math_solver_tool":     "solve_math",
                "if_not_math":          "return_ocr_text_directly",
            },
            "tool_rank":   plan.get("tool_rank", []),
            "preprocess":  [s["tool"] for s in normalized_steps],
            "captcha_type": captcha_type,
            "is_math":     is_math,
            "is_chinese":  is_chinese,
            "pipeline":    normalized_steps,   
        }

    def _format_outcome_records(
        self,
        scored:  List[Tuple[Dict, float]],
        outcome: str = "correct",
        limit:   int = 3,
    ) -> str:
        """
        Format (record, similarity) pairs into a labelled prompt string.

        outcome : "correct" → frame as positive examples to replicate;
                  "failed"  → frame as negative examples to avoid.
        """
        if not scored:
            label = "No similar correct examples found." if outcome == "correct" \
                    else "No similar failed examples found."
            return label

        compacted = []
        for rec, score in scored[:limit]:
            if not isinstance(rec, dict):
                continue
            pipe = rec.get("pipeline", {})
            if isinstance(pipe, dict):
                steps = pipe.get("preprocess_pipeline") or pipe.get("pipeline") or []
            elif isinstance(pipe, list):
                steps = pipe
            else:
                steps = []

            tools = [s.get("tool") for s in steps[:3] if isinstance(s, dict)]
            entry = {
                "similarity":   round(score, 3),
                "description":  rec.get("description", ""),
                "tools_applied": tools,
                "slot":         rec.get("_slot", ""),
            }
            if outcome == "correct":
                entry["note"] = "✓ succeeded — replicate if features match"
            else:
                entry["ocr_result"] = rec.get("ocr_result", "")
                entry["note"] = "✗ failed — avoid the same tools on similar images"
            compacted.append(entry)

        return json.dumps(compacted, indent=2, ensure_ascii=False)


    @staticmethod
    def _upscale_if_needed(image_b64: str) -> str:
        """Upscaling is currently disabled — pass native image size to the model."""
        return image_b64

    # Part 1: Visual Analysis 
    def visual_analysis(self, image_b64: str,
                        sample_id: str = "sample") -> Dict[str, Any]:
        image_b64 = self._upscale_if_needed(image_b64)
        messages = [
            {"role": "system", "content": VISUAL_ANALYSIS_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text",      "text": VISUAL_ANALYSIS_USER},
                ],
            },
        ]
        print(f"[Planner] visual_analysis messages: {messages}")
        raw      = self._chat(messages)
        print(f"[Planner] visual_analysis raw: {raw!r}")
        features = self._parse_json(raw)

        features.setdefault("noise",        "none")
        features.setdefault("background",   "clean")
        features.setdefault("distortion",   "none")
        features.setdefault("overlap",      "none")
        features.setdefault("char_quality", "normal")
        features.setdefault("captcha_type", "alphanumeric")
        features.setdefault("char_count",   5)
        features.setdefault("is_math",      False)
        features.setdefault("is_chinese",   False)
        features.setdefault("description",  "CAPTCHA image")

        if self._logger:
            self._logger.log_visual_analysis(sample_id, features)
        return features

    #  Part 2 + 3: Independent CoT + Action Pipeline
    def plan(
        self,
        features:        Dict[str, Any],
        correct_records: List[Tuple[Dict, float]] = None,  # (record, score) succeeded
        failed_records:  List[Tuple[Dict, float]] = None,  # (record, score) failed
        sample_id:       str = "sample",
    ) -> Dict[str, Any]:
        """
        Generate a preprocessing plan by reasoning from visual features.

        Parameters
        ----------
        correct_records : (record, score) pairs for successful past runs.
        failed_records  : (record, score) pairs for failed past runs.
        """
        tools_available = list_tools()
        correct_str = self._format_outcome_records(correct_records or [], outcome="correct", limit=3)
        failed_str  = self._format_outcome_records(failed_records  or [], outcome="failed",  limit=3)

        has_records = bool(correct_records) or bool(failed_records)
        no_records_note = "No past records found.. reason entirely from visual features.\n" if not has_records else ""

        user_content = (
            f"═══ CAPTCHA VISUAL FEATURES ═══\n"
            f"{json.dumps(features, indent=2)}\n\n"
            f"{no_records_note}"
            f"═══ CORRECT EXAMPLES — reference these for guidance ═══\n"
            f"{correct_str}\n\n"
            f"═══ FAILED EXAMPLES — learn from these mistakes ═══\n"
            f"{failed_str}\n\n"
            f"Now produce the preprocessing plan JSON:"
        )
        system_content = COT_SYSTEM.format(tools=", ".join(tools_available))

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]
        print(f"[Planner] plan messages: {messages}")
        raw    = self._chat(messages)
        print(f"[Planner] plan raw: {raw!r}")
        parsed = self._parse_json(raw)

        normalized = self._normalize_plan(parsed, features)

        if self._logger:
            self._logger.log_plan(sample_id, normalized)
        return normalized