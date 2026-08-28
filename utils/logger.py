"""
utils/logger.py
Structured logger for Dark Knight.

Each run writes two files inside logs/<run_id>/:
  events.jsonl: one JSON object per event (machine-readable)
  run.log: human-readable formatted transcript

Usage
-----
logger = PipelineLogger(run_id="freecity__step3__qwen9b__20260422_153000",
                        logs_dir="logs")

logger.log_visual_analysis(sample_id, features)
logger.log_rag_retrieval(sample_id, num_records, stage)
logger.log_plan(sample_id, plan)
logger.log_preprocess_step(sample_id, step_num, tool, params, success, error)
logger.log_ocr(sample_id, captcha_type, ocr_raw)
logger.log_math_solve(sample_id, expression, result)
logger.log_result(sample_id, ground_truth, prediction, correct, duration_sec)
logger.log_error(sample_id, message)
"""
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional


# ANSI colour helpers (terminal only) 
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _RESET


class PipelineLogger:
    """
    Structured logger per run.

    Parameters
    ----------
    run_id : str
        Unique identifier for this run (dataset + model combo + timestamp).
    logs_dir : str
        Root directory for all log files.
    """

    def __init__(self, run_id: str, logs_dir: str = "logs"):
        self.run_id   = run_id
        self.run_dir  = os.path.join(logs_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self._jsonl_path = os.path.join(self.run_dir, "events.jsonl")
        self._log_path   = os.path.join(self.run_dir, "run.log")

        # Open both files in append mode so multiple pipeline runs don't clobber
        self._jsonl = open(self._jsonl_path, "a", encoding="utf-8")
        self._log   = open(self._log_path,   "a", encoding="utf-8")

        self._write_header()

    # Internal writers 
    def _write_header(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._txt(f"\n{'='*70}")
        self._txt(f"Run: {self.run_id}")
        self._txt(f"Started: {ts}")
        self._txt(f"{'='*70}\n")

    def _emit(self, event: Dict[str, Any]):
        """Write one event to the JSONL file."""
        event.setdefault("ts", time.time())
        self._jsonl.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def _txt(self, line: str):
        """Write one line to the human-readable log (and flush)."""
        self._log.write(line + "\n")
        self._log.flush()

    def _separator(self, sample_id: str):
        self._txt(f"\n{'─'*60}")
        self._txt(f"Sample: {sample_id}")
        self._txt(f"{'─'*60}")

    # Public logging API

    def log_visual_analysis(self, sample_id: str, features: Dict[str, Any]):
        """Log the planner's visual analysis output (extracted CAPTCHA features)."""
        self._emit({
            "event":     "visual_analysis",
            "sample_id": sample_id,
            "features":  features,
        })
        self._separator(sample_id)
        self._txt("[Planner] Visual Analysis")
        for key, val in features.items():
            self._txt(f"  {key:15s}: {val}")
        print(
            _c("[Planner]", _CYAN, _BOLD)
            + f" {sample_id} | "
            + f"type={features.get('captcha_type','?')}  "
            + f"is_math={features.get('is_math','?')}  "
            + f"is_chinese={features.get('is_chinese','?')}  "
            + f"noise={features.get('noise','?')}"
        )

    def log_rag_retrieval(self, sample_id: str, num_records: int, stage: str = ""):
        """Log how many RAG records were retrieved and at which stage."""
        self._emit({
            "event":       "rag_retrieval",
            "sample_id":   sample_id,
            "num_records": num_records,
            "stage":       stage,
        })
        tag = f" ({stage})" if stage else ""
        self._txt(f"[RAG{tag}] Retrieved {num_records} record(s)")
        print(_c("[RAG]", _CYAN) + f" {sample_id} | {num_records} record(s){tag}")

    def log_plan(self, sample_id: str, plan: Dict[str, Any]):
        """Log the planner's generated execution plan."""
        steps = plan.get("preprocess_pipeline", plan.get("pipeline", []))
        self._emit({
            "event":     "plan",
            "sample_id": sample_id,
            "plan":      plan,
        })
        self._txt("[Planner] Execution Plan")
        self._txt(f"  reasoning : {plan.get('reasoning', '')}")
        self._txt(f"  steps     : {[s.get('tool') for s in steps]}")
        self._txt(f"  captcha   : {plan.get('captcha_type', '?')}")
        print(
            _c("[Planner]", _CYAN, _BOLD)
            + f" {sample_id} | plan → "
            + str([s.get("tool") for s in steps])
        )

    def log_preprocess_step(self, sample_id: str, step_num: int,
                            tool: str, params: Dict,
                            success: bool, error: Optional[str] = None):
        """Log one preprocessing tool execution."""
        self._emit({
            "event":     "preprocess_step",
            "sample_id": sample_id,
            "step":      step_num,
            "tool":      tool,
            "params":    params,
            "success":   success,
            "error":     error,
        })
        if success:
            status_txt = "✓"
            status_col = _c(f"step {step_num} '{tool}' ✓", _GREEN)
        else:
            status_txt = f"✗ {error}"
            status_col = _c(f"step {step_num} '{tool}' ✗  {error}", _RED)
        self._txt(f"[Executor] {status_txt}")
        print(_c("[Executor]", _YELLOW) + f" {sample_id} | " + status_col)

    def log_ocr(self, sample_id: str, captcha_type: str, ocr_raw: str):
        """Log the raw OCR output."""
        self._emit({
            "event":        "ocr",
            "sample_id":    sample_id,
            "captcha_type": captcha_type,
            "ocr_raw":      ocr_raw,
        })
        self._txt(f"[OCR/{captcha_type}] raw → '{ocr_raw}'")
        print(
            _c("[OCR]", _CYAN)
            + f" {sample_id} | [{captcha_type}] → "
            + _c(f"'{ocr_raw}'", _BOLD)
        )

    def log_math_solve(self, sample_id: str, expression: str, result: str):
        """Log a math equation being solved post-OCR."""
        self._emit({
            "event":      "math_solve",
            "sample_id":  sample_id,
            "expression": expression,
            "result":     result,
        })
        self._txt(f"[Math] '{expression}' → '{result}'")
        print(
            _c("[Math]", _CYAN)
            + f" {sample_id} | '{expression}' → "
            + _c(f"'{result}'", _GREEN, _BOLD)
        )

    def log_result(self, sample_id: str, ground_truth: Optional[str],
                   prediction: str, correct: Optional[bool],
                   duration_sec: float):
        """Log the final prediction result for a sample."""
        self._emit({
            "event":        "result",
            "sample_id":    sample_id,
            "ground_truth": ground_truth,
            "prediction":   prediction,
            "correct":      correct,
            "duration_sec": duration_sec,
        })
        if correct is True:
            symbol, col = "✓ CORRECT", _GREEN
        elif correct is False:
            symbol, col = "✗ WRONG  ", _RED
        else:
            symbol, col = "? NO_GT  ", _YELLOW

        self._txt(
            f"[Result] {symbol}  gt='{ground_truth}'  "
            f"pred='{prediction}'  ({duration_sec:.2f}s)"
        )
        print(
            _c("[Result]", _BOLD)
            + f" {sample_id} | "
            + _c(symbol, col, _BOLD)
            + f"  gt='{ground_truth}'  pred='{prediction}'"
            + f"  ({duration_sec:.2f}s)"
        )

    def log_error(self, sample_id: str, message: str):
        """Log an unhandled error during processing."""
        self._emit({
            "event":     "error",
            "sample_id": sample_id,
            "message":   message,
        })
        self._txt(f"[ERROR] {message}")
        print(_c(f"[ERROR] {sample_id} | {message}", _RED, _BOLD))

    # Lifecycle
    def close(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._txt(f"\n{'='*70}")
        self._txt(f"Run finished: {ts}")
        self._txt(f"{'='*70}\n")
        self._jsonl.close()
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()