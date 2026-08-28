"""
utils/rag.py
Dual-retrieval RAG store.

Each RAGStore instance is one JSON-backed database, isolated per
(dataset, planner_model, ocr_model) combo by the caller (see run_pipeline.py)

Architecture:
  - Text query  → sentence-transformer embedding (384-dim) → cosine similarity
  - Image query → per-channel RGB pHash (192-dim)          → cosine similarity
  Both scores are weighted equally (0.5 / 0.5); records below the similarity
  threshold are dropped so the planner never receives irrelevant hints.

Records are bucketed into two slots per store, each capped at slot_capacity
and evicting the least-retrieved record when full:
  Slot correct: past successes.
  Slot failed:  past failures.

query_split() is the primary retrieval path: top k/2 from correct + top k/2
from failed, each ranked independently by weighted similarity, so the planner
sees both "what worked" and "what to avoid" examples.

Records are persisted to a JSON file for human-readable inspection.
"""
import json
import os
import re
import hashlib
import time
from typing import List, Dict, Any, Tuple

import numpy as np
from scipy.fft import dct as _scipy_dct

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    print("[RAG] Warning: sentence-transformers not installed; NLP retrieval disabled.")


# Slot constants
_SLOT_CORRECT    = "correct"   # all successes
_SLOT_FAILED     = "failed"    # all failures
_SLOT_CAPACITY   = 50          # PER-SLOT fallback default
_DEDUP_THRESHOLD = 0.99        # skip add only if record is near-identical
_TEXT_WEIGHT     = 0.5         # description text similarity weight
_FEAT_WEIGHT     = 0.5         # pHash image similarity weight
_FEAT_DIM        = 192         # pHash output dimension: 64-dim × 3 RGB channels


def _classify_slot(success: bool) -> str:
    return _SLOT_CORRECT if success else _SLOT_FAILED


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# pHash-RGB (perceptual hash, per-channel) 
# Standard pHash algorithm (Krawetz 2011; Zauner 2010) extended to RGB:
#   1. Resize to 32×32 (hash_size=8, highfreq_factor=4 per the standard)
#   2. Apply 2D DCT (row-wise then column-wise, ortho normalisation)
#   3. Retain the top-left 8×8 block of low-frequency DCT coefficients
#   4. Threshold each of the 64 values against their median → 64 binary bits
#   5. Repeat independently for R, G, B channels
#   6. Concatenate R‖G‖B → 192-dim binary vector
# Steps 1–4 match the canonical single-channel pHash exactly (Krawetz 2011).
# The per-channel extension (steps 5–6) preserves colour gradient structure
# so that, e.g., red interference lines hash differently from black ones.

def _phash_vector(image_b64: str) -> np.ndarray:
    """
    192-dim pHash-RGB: standard pHash (Krawetz 2011) applied per RGB channel.
    Returns a zero vector on decode/resize failure.
    """
    if not image_b64:
        return np.zeros(_FEAT_DIM, dtype=np.float32)
    try:
        import base64 as _b64
        from PIL import Image
        import io
        payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        raw = _b64.b64decode(payload)
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((32, 32), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)  # shape: (32, 32, 3)
        bits = []
        for c in range(3):  # R, G, B
            dct2d = _scipy_dct(_scipy_dct(arr[:, :, c], axis=0, norm="ortho"),
                               axis=1, norm="ortho")
            top = dct2d[:8, :8].flatten()                        
            bits.append((top > np.median(top)).astype(np.float32))  # median threshold (Krawetz 2011)
        return np.concatenate(bits)
    except Exception:
        return np.zeros(_FEAT_DIM, dtype=np.float32)


# ───────────────────────────────────────────────────────────────────────────
class RAGStore:
    """
    In-memory + JSON-persisted RAG database with dual retrieval.
    """

    def __init__(self, db_path: str = "rag_store/rag_db.json",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.5,
                 failed_similarity_threshold: float = None,
                 slot_capacity: int = _SLOT_CAPACITY):
        self.db_path                     = db_path
        self.similarity_threshold        = similarity_threshold
        self.failed_similarity_threshold = failed_similarity_threshold if failed_similarity_threshold is not None else similarity_threshold
        self._slot_capacity              = slot_capacity
        self.records: List[Dict[str, Any]] = []
        self._text_embeddings: List[np.ndarray] = []
        self._feat_embeddings: List[np.ndarray] = []

        self._st_model = None
        if _ST_AVAILABLE:
            try:
                self._st_model = SentenceTransformer(embedding_model)
            except Exception as exc:
                print(f"[RAG] Could not load embedding model: {exc}")

        self.load()

    # Persistence (JSON)
    def save(self, silent: bool = False):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        payload = {
            "records":          self.records,
            "text_embeddings":  [e.tolist() for e in self._text_embeddings],
            "feat_embeddings":  [e.tolist() for e in self._feat_embeddings],
        }
        with open(self.db_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        if not silent:
            print(f"[RAG] Saved {len(self.records)} records → {self.db_path}")

    def load(self):
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.records          = data.get("records", [])
            self._text_embeddings = [
                np.array(e, dtype=np.float32)
                for e in data.get("text_embeddings", [])
            ]
            self._feat_embeddings = [
                np.array(e, dtype=np.float32)
                for e in data.get("feat_embeddings", [])
            ]

            if (len(self._feat_embeddings) != len(self.records)
                    or (self._feat_embeddings
                        and self._feat_embeddings[0].shape[0] != _FEAT_DIM)):
                print(f"[RAG] feat_embeddings dimension mismatch — zeroing "
                      f"{len(self.records)} vectors (images not stored)")
                self._feat_embeddings = [
                    np.zeros(_FEAT_DIM, dtype=np.float32) for _ in self.records
                ]
            print(f"[RAG] Loaded {len(self.records)} records from {self.db_path}")
        except Exception as exc:
            print(f"[RAG] Could not load DB: {exc}")

    # Indexing
    def _embed_text(self, text: str) -> np.ndarray:
        if self._st_model:
            return self._st_model.encode(text, convert_to_numpy=True).astype(np.float32)
        raw = text.encode("utf-8")
        h   = hashlib.sha256(raw).hexdigest()
        return np.array([int(c, 16) / 15.0 for c in h[:64]], dtype=np.float32)

    # Slot helpers
    def _count_slot(self, slot: str) -> int:
        return sum(1 for r in self.records if r.get("_slot") == slot)

    def _evict_least_retrieved(self, slot: str):
        """Remove the least-retrieved record in the given slot."""
        indices = [
            i for i, r in enumerate(self.records)
            if r.get("_slot") == slot
        ]
        if not indices:
            return
        least_i = min(indices, key=lambda i: self.records[i].get("_retrieval_count", 0))
        self.records.pop(least_i)
        self._text_embeddings.pop(least_i)
        self._feat_embeddings.pop(least_i)

    def _is_duplicate(self, slot: str,
                      t_emb: np.ndarray, f_emb: np.ndarray) -> bool:
        """Return True if an existing record in the slot is near-identical."""
        for i, r in enumerate(self.records):
            if r.get("_slot") != slot:
                continue
            t_sim = _cosine_sim(t_emb, self._text_embeddings[i])
            f_sim = _cosine_sim(f_emb, self._feat_embeddings[i])
            if _TEXT_WEIGHT * t_sim + _FEAT_WEIGHT * f_sim >= _DEDUP_THRESHOLD:
                return True
        return False

    def _classify_ocr_failure(self, record: Dict) -> str:
        """Tag each record with the type of OCR failure so hints can be surfaced later."""
        if record.get("ocr_result") == "OCR_ERROR":
            return "empty_output"
        if record.get("success"):
            return "correct"
        final = str(record.get("final_text", ""))
        gt    = str(record.get("ground_truth", "") or "")
        if re.search(r"(.)\1", final):
            return "duplicate_chars"
        if len(final) > len(gt) + 2:
            return "over_generation"
        if gt and len(final) < len(gt) - 1:
            return "under_generation"
        return "wrong_text"

    def add(self, record: Dict[str, Any],
            tools_used: List[str] = None,
            image_b64: str = ""):
        """
        Add a record into the correct or failed slot based on outcome.

        The image is used to compute a pHash embedding for similarity search
        but is NOT stored in the record (keeps storage compact).

        Parameters
        ----------
        tools_used : list of tool names applied in this run (unused, reserved)
        image_b64  : base64 CAPTCHA image for pHash computation
        """
        record.setdefault("timestamp", time.time())
        record.setdefault("record_type", "execution")
        record.setdefault("_retrieval_count", 0)
        record.setdefault("ocr_failure_type", self._classify_ocr_failure(record))

        slot  = _classify_slot(bool(record.get("success", False)))

        t_emb = self._embed_text(record.get("description", ""))
        f_emb = _phash_vector(image_b64)

        if self._is_duplicate(slot, t_emb, f_emb):
            return

        if self._count_slot(slot) >= self._slot_capacity:
            self._evict_least_retrieved(slot)

        record["_slot"] = slot

        self.records.append(record)
        self._text_embeddings.append(t_emb)
        self._feat_embeddings.append(f_emb)

    #  Retrieval from each slot
    def _query_slot(
        self,
        q_text: np.ndarray,
        q_feat: np.ndarray,
        slot:   str,
        top_k:  int,
        threshold: float,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return top_k records from a specific slot, ranked by weighted similarity."""
        scored = []
        for i, r in enumerate(self.records):
            if r.get("_slot") != slot:
                continue
            t_sim = _cosine_sim(q_text, self._text_embeddings[i])
            f_sim = _cosine_sim(q_feat, self._feat_embeddings[i])
            avg   = _TEXT_WEIGHT * t_sim + _FEAT_WEIGHT * f_sim
            if avg >= threshold:
                scored.append((i, avg))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.records[i], round(score, 4)) for i, score in scored[:top_k]]

    def query_with_scores(
        self,
        description: str = "",
        image_b64:   str = "",
        top_k:       int = 5,
        threshold:   float = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Dual-query retrieval. Returns (record, score) pairs, sorted by score
        descending, filtered to score >= threshold.

        Similarity:
          - text cosine similarity  (sentence-transformer on description)
          - pHash cosine similarity (perceptual image hash)

        Parameters
        ----------
        image_b64 : base64-encoded CAPTCHA image for dHash similarity.
        threshold : override the store's default similarity_threshold.
        """
        if not self.records:
            return []

        effective_threshold = threshold if threshold is not None else self.similarity_threshold

        q_text = self._embed_text(description)
        q_feat = _phash_vector(image_b64)

        scored = []
        for i, (te, fe) in enumerate(zip(self._text_embeddings, self._feat_embeddings)):
            t_sim = _cosine_sim(q_text, te)
            f_sim = _cosine_sim(q_feat, fe)
            avg   = _TEXT_WEIGHT * t_sim + _FEAT_WEIGHT * f_sim
            if avg >= effective_threshold:
                scored.append((i, avg))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.records[i], round(score, 4)) for i, score in scored[:top_k]]

    def query(
        self,
        description: str = "",
        image_b64:   str = "",
        top_k:       int = 5,
        threshold:   float = None,
    ) -> List[Dict[str, Any]]:
        return [r for r, _ in self.query_with_scores(
            description=description,
            image_b64=image_b64,
            top_k=top_k,
            threshold=threshold,
        )]

    def query_split(
        self,
        description: str = "",
        image_b64:   str = "",
        top_k:       int = 5,
        threshold:   float = None,
    ) -> Tuple[List[Tuple[Dict, float]], List[Tuple[Dict, float]]]:
        """
        Retrieve top k/2 from the correct slot and top k/2 from the failed slot.

        Returns
        correct_hits : list of (record, score), sorted by score desc, len ≤ k/2
        failed_hits  : list of (record, score), sorted by score desc, len ≤ k/2
        """
        if not self.records:
            return [], []

        half_k    = max(1, top_k // 2)
        threshold = threshold if threshold is not None else self.similarity_threshold
        q_text    = self._embed_text(description)
        q_feat    = _phash_vector(image_b64)

        correct_hits = self._query_slot(q_text, q_feat, _SLOT_CORRECT, half_k, threshold)
        failed_hits  = self._query_slot(q_text, q_feat, _SLOT_FAILED,  half_k, self.failed_similarity_threshold)

        for r, _ in correct_hits + failed_hits:
            r["_retrieval_count"] = r.get("_retrieval_count", 0) + 1

        return correct_hits, failed_hits

    def query_split_image_only(
        self,
        image_b64: str,
        top_k:     int = 5,
        threshold: float = None,
    ) -> Tuple[List[Tuple[Dict, float]], List[Tuple[Dict, float]]]:
        """
        Query by pHash ONLY if the description is not yet available. 
        """
        if not self.records:
            return [], []

        half_k    = max(1, top_k // 2)
        threshold = (threshold if threshold is not None else self.similarity_threshold)
        q_feat    = _phash_vector(image_b64)

        def _slot_hits(slot):
            scored = [
                (i, _cosine_sim(q_feat, self._feat_embeddings[i]))
                for i, r in enumerate(self.records)
                if r.get("_slot") == slot
            ]
            scored = [(i, s) for i, s in scored if s >= threshold]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [(self.records[i], round(s, 4)) for i, s in scored[:half_k]]

        correct_hits = _slot_hits(_SLOT_CORRECT)
        failed_hits  = _slot_hits(_SLOT_FAILED)

        for r, _ in correct_hits + failed_hits:
            r["_retrieval_count"] = r.get("_retrieval_count", 0) + 1

        return correct_hits, failed_hits

    def __len__(self):
        return len(self.records)

    @staticmethod
    def build_ocr_hints(failed_records: List[Tuple[Dict, float]]) -> str:
        """Build a short warning string from OCR failure patterns in failed RAG records."""
        failure_types = {
            r.get("ocr_failure_type", "")
            for r, _ in failed_records
            if isinstance(r, dict)
        }
        hints = []
        if "duplicate_chars" in failure_types:
            hints.append("⚠ Similar images previously caused duplicate characters — output each character exactly once.")
        if "empty_output" in failure_types:
            hints.append("⚠ Similar images previously returned empty — ensure you output the visible characters.")
        if "over_generation" in failure_types:
            hints.append("⚠ Similar images previously caused too many characters — do not add extra characters.")
        return " ".join(hints)
