"""
pipeline.py  - Orchestrates one CAPTCHA sample end-to-end.

Flow per sample
  1. PlannerAgent.visual_analysis  ->  image → features
  2. RAGStore.query_split          ->  similar past records, by description
                                        text + image (pHash) together
  3. PlannerAgent.plan()           ->  preprocessing pipeline
  4. ExecutorAgent.execute()       ->  preprocess + OCR + math solve
  5. Write result back to RAG store
"""
import os
import time
import traceback
from datetime import datetime
from typing import Dict, Any

import sys
sys.path.insert(0, os.path.dirname(__file__))

import config
from agents.planner_agent  import PlannerAgent
from agents.executor_agent import ExecutorAgent
from utils.rag             import RAGStore
from utils.result_tracker  import ResultTracker
from utils.logger          import PipelineLogger
from utils.errors          import ConnectionFailure, CONNECTION_ERROR_KEY
from utils.naming          import sanitize_tag


class CaptchaPipeline:
    """
    One pipeline instance = one (dataset, planner-model, ocr-model) combination.

    Parameters
    ----------
    planner_cfg  : {"label": str, "model": str, "endpoint": str}
    ocr_cfg      : {"label": str, "model": str, "endpoint": str}
    rag_store    : per-combo RAGStore (isolated per dataset + planner + ocr)
    tracker      : per-combo ResultTracker
    dataset_name : name of the dataset being evaluated
    """

    def __init__(self, planner_cfg: Dict, ocr_cfg: Dict,
                 rag_store: RAGStore, tracker: ResultTracker,
                 dataset_name: str = "unknown",
                 use_rag: bool = True):
        self.planner_cfg  = planner_cfg
        self.ocr_cfg      = ocr_cfg
        self.dataset_name = dataset_name
        self.rag_store    = rag_store
        self.tracker      = tracker
        self.use_rag      = use_rag

        run_id = sanitize_tag(
            f"{dataset_name}__{planner_cfg['label']}__{ocr_cfg['label']}"
            f"__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self.logger = PipelineLogger(run_id=run_id, logs_dir=config.LOGS_DIR)

        self.planner = PlannerAgent(
            model    = planner_cfg["model"],
            endpoint = planner_cfg["endpoint"],
            label    = planner_cfg["label"],
            logger   = self.logger,
        )
        self.executor = ExecutorAgent(
            ocr_model    = ocr_cfg["model"],
            ocr_endpoint = ocr_cfg["endpoint"],
            ocr_label    = ocr_cfg["label"],
            log_dir      = os.path.join(config.LOGS_DIR, "processed_images"),
            logger       = self.logger,
        )

    
    def _query_rag(self, sample_id: str, desc: str, image_b64: str):
        """RAG dual query: text (description) + image (pHash). Returns
        (correct_scored, failed_scored, rag_retrieval_sec)."""
        t_rag = time.time()
        if self.use_rag:
            correct_scored, failed_scored = self.rag_store.query_split(
                description = desc,
                image_b64   = image_b64,
                top_k       = config.RAG_TOP_K,
                threshold   = config.RAG_SIMILARITY_THRESHOLD,
            )
            self.logger.log_rag_retrieval(sample_id, len(correct_scored) + len(failed_scored))
        else:
            correct_scored, failed_scored = [], []
            self.logger.log_rag_retrieval(sample_id, 0)
        rag_retrieval_sec = round(time.time() - t_rag, 3)
        return correct_scored, failed_scored, rag_retrieval_sec

    def _write_back_to_rag(self, sample_id: str, desc: str, features: Dict,
                            plan: Dict, exec_result: Dict, ground_truth,
                            prediction: str, correct: bool, image_b64: str):
        """Write the executed sample back to the RAG store (if enabled)."""
        tools_applied = [s["tool"] for s in plan.get("preprocess_pipeline", [])]
        if self.use_rag:
            self.rag_store.add(
                {
                    "id":           sample_id,
                    "description":  desc,
                    "features":     features,
                    "pipeline":     plan,
                    "ocr_result":   exec_result["ocr_raw"],
                    "final_text":   prediction,
                    "ground_truth": ground_truth,
                    "success":      bool(correct),
                },
                tools_used = tools_applied,
                image_b64  = image_b64,
            )

    def _build_result(self, sample_id, ground_truth, duration, *,
                       prediction="", correct=False, captcha_type="unknown",
                       tools_used=None, planned_tools=None, executor_steps=None,
                       planner_reasoning="", ocr_raw="", rag_retrieval_sec=None,
                       error=None) -> Dict:
        """
        Generate the common result-dict schema.
        """
        result = {
            "dataset":           self.dataset_name,
            "sample_id":         sample_id,
            "ground_truth":      ground_truth,
            "prediction":        prediction,
            "correct":           correct,
            "planner_model":     self.planner_cfg["label"],
            "ocr_model":         self.ocr_cfg["label"],
            "captcha_type":      captcha_type,
            "tools_used":        tools_used if tools_used is not None else [],
            "planned_tools":     planned_tools if planned_tools is not None else [],
            "executor_steps":    executor_steps if executor_steps is not None else [],
            "planner_reasoning": planner_reasoning,
            "ocr_raw":           ocr_raw,
        }
        if rag_retrieval_sec is not None:
            result["rag_retrieval_sec"] = rag_retrieval_sec
        result["duration_sec"] = duration
        result["error"]        = error
        return result

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full pipeline for ONE single CAPTCHA. """
        t0           = time.time()
        sample_id    = str(sample.get("id", "unknown"))
        ground_truth = sample.get("label")
        image_b64    = sample.get("b64_string", "")

        if not image_b64:
            return self._error_result(sample_id, ground_truth, "No image data", t0)

        try:
            # 1. Visual analysis
            features = self.planner.visual_analysis(
                image_b64 = image_b64,
                sample_id = sample_id,
            )
            desc = features.get("description", "")

            # 2. RAG dual query: text (description) + image (pHash)
            correct_scored, failed_scored, rag_retrieval_sec = self._query_rag(
                sample_id, desc, image_b64)

            # 3. Planning: features + RAG hints → tool pipeline
            plan = self.planner.plan(
                features        = features,
                correct_records = correct_scored,
                failed_records  = failed_scored,
                sample_id       = sample_id,
            )

            ocr_hints = RAGStore.build_ocr_hints(failed_scored) if self.use_rag else ""

            # 4. Execute: preprocess → OCR → math solve
            exec_result = self.executor.execute(
                image_b64     = image_b64,
                pipeline_json = plan,
                features      = features,
                sample_id     = f"{self.dataset_name}_{sample_id}",
                ocr_hints     = ocr_hints,
            )

            prediction = exec_result["final_text"]
            correct    = (
                ground_truth is not None
                and prediction.strip() == str(ground_truth).strip()
            )

            # 5. Write back to RAG
            self._write_back_to_rag(
                sample_id, desc, features, plan, exec_result,
                ground_truth, prediction, correct, image_b64,
            )

            duration = round(time.time() - t0, 2)
            result = self._build_result(
                sample_id, ground_truth, duration,
                prediction        = prediction,
                correct           = correct,
                captcha_type      = features.get("captcha_type", "unknown"),
                tools_used        = [s["tool"] for s in exec_result["steps"]],
                planned_tools     = [s["tool"] for s in plan.get("preprocess_pipeline", [])],
                executor_steps    = [
                    {"tool": s["tool"], "success": s.get("success"),
                     "error": s.get("error"), "note": s.get("note")}
                    for s in exec_result["steps"]
                ],
                planner_reasoning = plan.get("reasoning", ""),
                ocr_raw           = exec_result.get("ocr_raw", ""),
                rag_retrieval_sec = rag_retrieval_sec,
                error             = None,
            )
            self.logger.log_result(sample_id, ground_truth, prediction, correct, duration)

        except ConnectionFailure as exc:
            msg = str(exc)
            self.logger.log_error(sample_id, "[connection_error] " + msg)
            result = self._error_result(sample_id, ground_truth, "connection_error", t0)
            result[CONNECTION_ERROR_KEY] = True
            return result

        except Exception as exc:
            msg = str(exc)
            self.logger.log_error(sample_id, msg + "\n" + traceback.format_exc())
            result = self._error_result(sample_id, ground_truth, msg, t0)

        self.tracker.record(result)
        return result

    def _error_result(self, sample_id, ground_truth, error_msg, t0) -> Dict:
        duration = round(time.time() - t0, 2)
        result = self._build_result(
            sample_id, ground_truth, duration,
            error = error_msg,
        )
        self.logger.log_result(sample_id, ground_truth, "", False, duration)
        return result

    def close(self):
        self.logger.close()
