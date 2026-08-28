"""
agents/executor_agent.py
Executor Agent for the Dark knight.

Execution flow:
  1. Run each image preprocessing tool according to the planner's plan.
  2. Route OCR to a type-specific prompt:
       is_math    → math extraction prompt  → solve via solve_equation
       is_chinese → Chinese OCR prompt
       else       → English/alphanumeric OCR prompt
  3. Post-OCR: if captcha is math, solve the extracted expression.

Returns a structured execution trace:
{
  "steps":       [{step, tool, params, success, error}, ...],
  "ocr_raw":     str,
  "final_text":  str,
  "final_image": str,   # b64 of the last processed image if multiple tools applied
  "is_math":     bool,
  "captcha_type": str,
}
"""
import os
import re
import base64
import time
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from tools.registry   import get_tool
from tools.ocr_tool   import run_ocr, OCR_ERROR_SENTINEL
from utils.solve_math import is_valid_math_ocr, solve_equation, ERROR_EMPTY, DIV_ZERO, MATH_ERR_PREFIX
from utils.logger     import PipelineLogger


# Handle Cyrillic characters (visually equivalent to Latin)
_CYRILLIC_TO_LATIN = str.maketrans(
    "аеорсухіАВЕКМНОРСТХ",
    "aeopcyxiABEKMHOPCTX",
)

# Math-expression detection
_MATH_CHARS   = re.compile(r'^[0-9+\-*/=?×÷]+$')
_HAS_OPERATOR = re.compile(r'[+\-*/=?×÷]')

def _normalize_homoglyphs(text: str) -> str:
    return text.translate(_CYRILLIC_TO_LATIN)


class ExecutorAgent:
    """
    executor: runs preprocessing tool steps then calls OCR.
    """

    def __init__(self, ocr_model: str, ocr_endpoint: str = "ollama",
                 ocr_label: str = "", log_dir: Optional[str] = None,
                 logger: Optional[PipelineLogger] = None):
        self.ocr_model    = ocr_model
        self.ocr_endpoint = ocr_endpoint
        self.ocr_label    = ocr_label or ocr_model
        self.log_dir      = log_dir
        self._logger      = logger

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Logging helpers 
    def _save_intermediate(self, image_b64: str, name: str):
        if not self.log_dir or not config.LOG_INTERMEDIATES:
            return
        try:
            enc = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
            raw = base64.b64decode(enc)
            with open(os.path.join(self.log_dir, f"{name}.png"), "wb") as fh:
                fh.write(raw)
        except Exception as exc:
            print(f"[Executor] Could not save intermediate '{name}': {exc}")

    #  Image upscaling helper
    @staticmethod
    def _upscale_if_needed(image_b64: str) -> str:
        return image_b64

    #  Execution sub-steps 
    @staticmethod
    def _classify_captcha(pipeline_json: Dict[str, Any],
                           features: Dict[str, Any]) -> Dict[str, Any]:
        """Derive captcha_type / is_math / is_chinese / ocr_type from planner output."""
        captcha_type = (
            pipeline_json.get("captcha_type")
            or features.get("captcha_type", "alphanumeric")
        )
        is_math_captcha = (
            pipeline_json.get("is_math")
            or features.get("is_math", False)
            or captcha_type == "math"
        )
        is_chinese_captcha = (
            pipeline_json.get("is_chinese")
            or features.get("is_chinese", False)
            or captcha_type == "chinese"
        )

        if is_chinese_captcha:
            ocr_type = "chinese"
        elif is_math_captcha:
            ocr_type = "math"
        else:
            ocr_type = "alphanumeric"

        return {
            "captcha_type":       captcha_type,
            "is_math_captcha":    is_math_captcha,
            "is_chinese_captcha": is_chinese_captcha,
            "ocr_type":           ocr_type,
        }

    @staticmethod
    def _extract_pipeline_steps(pipeline_json: Dict[str, Any]) -> List[Dict]:
        pipeline_steps = pipeline_json.get("preprocess_pipeline")
        if pipeline_steps is None:
            pipeline_steps = pipeline_json.get("pipeline", [])
        if not isinstance(pipeline_steps, list):
            pipeline_steps = []
        return pipeline_steps

    def _run_preprocessing(self, current_image: str, pipeline_steps: List[Dict],
                            sample_id: str) -> "tuple[str, List[Dict]]":
        """Run each preprocessing tool step in order; returns (image, steps_trace)."""
        steps_trace: List[Dict] = []

        for i, step in enumerate(pipeline_steps):
            tool_name = step.get("tool", "").strip().lower()
            params    = step.get("params", {})

            if tool_name == "solve_math":
                # solve_math is a post-OCR tool, defer it
                steps_trace.append({
                    "step":    i + 1,
                    "tool":    "solve_math",
                    "params":  params,
                    "success": True,
                    "note":    "deferred to post-OCR",
                })
                continue

            try:
                tool_fn       = get_tool(tool_name)
                current_image = tool_fn(current_image, **params)
                self._save_intermediate(current_image, f"{sample_id}_{i+1:02d}_{tool_name}")
                steps_trace.append({
                    "step":    i + 1,
                    "tool":    tool_name,
                    "params":  params,
                    "success": True,
                    "error":   None,
                })
                if self._logger:
                    self._logger.log_preprocess_step(
                        sample_id, i + 1, tool_name, params, success=True)
                else:
                    print(f"[Executor] step {i+1} '{tool_name}' ✓")

            except KeyError as exc:
                err_msg = f"UnknownTool: {exc}"
                steps_trace.append({
                    "step":    i + 1,
                    "tool":    tool_name,
                    "params":  params,
                    "success": False,
                    "error":   err_msg,
                })
                if self._logger:
                    self._logger.log_preprocess_step(
                        sample_id, i + 1, tool_name, params, success=False, error=err_msg)
                else:
                    print(f"[Executor] step {i+1} '{tool_name}' ✗ UnknownTool")

            except Exception as exc:
                err_msg = str(exc)
                steps_trace.append({
                    "step":    i + 1,
                    "tool":    tool_name,
                    "params":  params,
                    "success": False,
                    "error":   err_msg,
                })
                if self._logger:
                    self._logger.log_preprocess_step(
                        sample_id, i + 1, tool_name, params, success=False, error=err_msg)
                else:
                    print(f"[Executor] step {i+1} '{tool_name}' ✗ {exc}")

        return current_image, steps_trace

    @staticmethod
    def _ocr_looks_bad(text: str, char_count: Optional[int]) -> bool:
        """
        Triggers when: result contains newlines/ too long for a CAPTCHA,
        """
        if not text or text == OCR_ERROR_SENTINEL:
            return False
        if "\n" in text:
            return True
        if len(text) > 20:
            return True
        if char_count and len(text) > char_count * 2:
            return True
        return False

    def _run_ocr_with_fallback(self, current_image: str, image_b64: str, ocr_type: str,
                                char_count: Optional[int], ocr_hints: str,
                                is_math_captcha: bool, is_chinese_captcha: bool) -> str:
        """Retry on the original image if the result looks bad."""
        print(f"[Executor] captcha_type={ocr_type}  char_count={char_count}  "
              f"is_math={bool(is_math_captcha)}  is_chinese={bool(is_chinese_captcha)}")
        print(f"[Executor] Running OCR [{ocr_type}] with model '{self.ocr_label}'…")
        ocr_raw = run_ocr(
            current_image,
            model        = self.ocr_model,
            endpoint     = self.ocr_endpoint,
            captcha_type = ocr_type,
            char_count   = char_count,
            hints        = ocr_hints,
            retries      = config.MAX_RETRIES,
            timeout      = config.REQUEST_TIMEOUT,
        )
        print(f"[Executor] OCR raw: '{ocr_raw}'")

        if self._ocr_looks_bad(ocr_raw, char_count):
            print(f"[Executor] OCR result looks bad — retrying on original image")
            ocr_fallback = run_ocr(
                image_b64,
                model        = self.ocr_model,
                endpoint     = self.ocr_endpoint,
                captcha_type = ocr_type,
                char_count   = char_count,
                retries      = config.MAX_RETRIES,
                timeout      = config.REQUEST_TIMEOUT,
            )
            print(f"[Executor] Fallback OCR raw: '{ocr_fallback}'")
            if ocr_fallback and ocr_fallback != OCR_ERROR_SENTINEL and not self._ocr_looks_bad(ocr_fallback, char_count):
                print(f"[Executor] Using fallback result over bad result")
                ocr_raw = ocr_fallback

        return ocr_raw

    def _try_math_reclassification(self, current_image: str, ocr_raw: str,
                                    is_math_captcha: bool,
                                    steps_trace: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Handle misclassification for math captcha.
        Returns a completed result dict if reclassification succeeded, else None.
        """
        if not (not is_math_captcha
                and ocr_raw not in (OCR_ERROR_SENTINEL, "")
                and _MATH_CHARS.fullmatch(ocr_raw)
                and _HAS_OPERATOR.search(ocr_raw)):
            return None

        print(f"[Executor] alphanumeric OCR looks like math ({ocr_raw!r}) ,retrying with math prompt")
        math_ocr = run_ocr(
            current_image,
            model        = self.ocr_model,
            endpoint     = self.ocr_endpoint,
            captcha_type = "math",
            retries      = 1,
            timeout      = config.REQUEST_TIMEOUT,
        )
        if math_ocr not in (OCR_ERROR_SENTINEL, ""):
            solved = solve_equation(math_ocr)
            if not solved.startswith((ERROR_EMPTY, DIV_ZERO, MATH_ERR_PREFIX)):
                print(f"[Executor] Math reclassified: '{ocr_raw}' → solved '{solved}'")
                ocr_raw    = math_ocr
                final_text = solved
                final_text = _normalize_homoglyphs(final_text)
                return {
                    "steps":        steps_trace,
                    "ocr_raw":      ocr_raw,
                    "final_text":   final_text,
                    "final_image":  current_image,
                    "is_math":      True,
                    "captcha_type": "math",
                }
        return None

    @staticmethod
    def _solve_if_math(ocr_raw: str, is_math_captcha: bool) -> str:
        """
        Solve math if OCR output is a valid expression.
        Validation: must contain (= or ?) AND at least one operator (+ - * /).
        If the format is not valid math, return ocr_raw.
        solve_equation handles all three forms internally:
          "17+16=33" → extracts "33" directly (answer already embedded)
          "4+5=?"    → strips marker, evaluates
          "34-66="   → strips "=", evaluates
        """
        final_text = ocr_raw
        if ocr_raw not in (OCR_ERROR_SENTINEL, "") and (is_valid_math_ocr(ocr_raw) or is_math_captcha):
            solved = solve_equation(ocr_raw)
            if not solved.startswith((ERROR_EMPTY, DIV_ZERO, MATH_ERR_PREFIX)):
                print(f"[Executor] Math solved: '{ocr_raw}' → '{solved}'")
                final_text = solved
        return final_text

    # Execution
    def execute(self,
                image_b64:     str,
                pipeline_json: Dict[str, Any],
                features:      Dict[str, Any],
                sample_id:     str = "sample",
                ocr_hints:     str = "") -> Dict[str, Any]:
        """
        Execute the tool pipeline and return an execution trace.

        Parameters
        ----------
        image_b64 : str
            Input CAPTCHA image (data-URI base64).
        pipeline_json : dict
            Output of PlannerAgent.plan().
        features : dict
            Output of PlannerAgent.visual_analysis().
        sample_id : str
            Used for intermediate file naming.
        """
        current_image = self._upscale_if_needed(image_b64)

        # Determine captcha type from planner 
        classification      = self._classify_captcha(pipeline_json, features)
        captcha_type        = classification["captcha_type"]
        is_math_captcha     = classification["is_math_captcha"]
        is_chinese_captcha  = classification["is_chinese_captcha"]
        ocr_type            = classification["ocr_type"]

        pipeline_steps = self._extract_pipeline_steps(pipeline_json)

        self._save_intermediate(current_image, f"{sample_id}_00_input")

        # 1. Preprocessing
        current_image, steps_trace = self._run_preprocessing(current_image, pipeline_steps, sample_id)

        self._save_intermediate(current_image, f"{sample_id}_final_processed")

        # 2. Type-routed OCR 
        char_count = features.get("char_count") or pipeline_json.get("char_count")
        ocr_raw = self._run_ocr_with_fallback(
            current_image, image_b64, ocr_type, char_count, ocr_hints,
            is_math_captcha, is_chinese_captcha,
        )

        # 2b. Math reclassification fallback 
        reclassified = self._try_math_reclassification(
            current_image, ocr_raw, is_math_captcha, steps_trace)
        if reclassified is not None:
            return reclassified

        # 3. Solve math if OCR output is a valid expression 
        final_text = self._solve_if_math(ocr_raw, is_math_captcha)

        final_text = _normalize_homoglyphs(final_text)

        return {
            "steps":        steps_trace,
            "ocr_raw":      ocr_raw,
            "final_text":   final_text,
            "final_image":  current_image,
            "is_math":      is_math_captcha,
            "captcha_type": captcha_type,
        }