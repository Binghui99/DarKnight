# DarKnight

## Overview

A pipeline for solving CAPTCHAs using vision-language models (VLMs), a RAG-based experience memory, and a small structured image preprocessing toolbox.

- **PlannerAgent** : analyses the CAPTCHA image, retrieves similar past examples from the RAG store, and produces a preprocessing plan via chain-of-thought reasoning.
- **ExecutorAgent** : deterministically runs the plan's tools, calls the OCR model with a type-specific prompt, and solves math expressions post-OCR when needed.
- **RAGStore** : persists execution history (per dataset/planner/OCR combo) and retrieve similar past records to the planner so it can learn from prior successes and failures.

---

## Project Structure

```
DarkKnight/
├── agents/
│   ├── planner_agent.py       # visual analysis & generate CoT plan 
│   ├── planner_prompts.py     # Prompt templates used by PlannerAgent
│   └── executor_agent.py      # Executes tools, ocr and math solving
├── tools/
│   ├── registry.py            # Tool registry
│   ├── toolbox.py             # Preprocessing implementations (numpy arrays)
│   ├── color_lines_removal.py # color_lines_removal tool (base64-native)
│   └── ocr_tool.py            # OCR via VLM 
├── utils/
│   ├── rag.py                 # Dual retrieval (text + image) memory system
│   ├── solve_math.py          # Post-OCR math equation solver
│   ├── dataset_loader.py      # Dataset loading (JSON / PDF / image files)
│   ├── result_tracker.py      # Result tracking, CSV/JSON export, summary printing
│   ├── logger.py              # Per-run structured logging
│   ├── llm_client.py          # Abstraction over OpenAI-compatible + Anthropic APIs
│   ├── naming.py              # Filesystem-safe tag sanitisation
│   └── errors.py              # ConnectionFailure and network-error classification
├── config/
│   └── config.py              # Model endpoints, MODELS list, RAG/pipeline settings
├── dataset/                   # Dataset JSON files and png
├── rag_store/                 # Stored RAG databases
├── pipeline.py                # Per-sample run orchestrator
└── run_pipeline.py            # Entry point: dataset × model evaluation loop
```

---

## Setup

1. install requirements
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your  endpoint URLs and API keys:

```
GPUSTACK_BASE_URL=...        # your OpenAI-compatible endpoint, e.g. https://<your-gpustack-host>/v1
ZIYI_CAPTCHA_API_KEY=...     # API key for that endpoint

ANTHROPIC_BASE_URL=...       # your Anthropic endpoint
AZURE_OPENAI_API_KEY=...     # API key for that endpoint
```

3. Setup models in `config.MODELS`: Remember to add endpoint(s) for each model `"endpoint": "gpustack"` or `"endpoint": "anthropic"`.
   
   See [Model entries](#model-entries) below for the expected structure.

4. Create a `datasets/` directory in the project root and place your dataset files there before running the pipeline.

    See [Dataset Format](#dataset-format) below for the expected structure.

---

## Configuration (`config/config.py`)

| Setting | Default | Description |
|---|---|---|
| `GPUSTACK_BASE_URL` / `GPUSTACK_API_KEY` | `env: GPUSTACK_BASE_URL` / `env: ZIYI_CAPTCHA_API_KEY` | OpenAI-compatible endpoint used for `endpoint="gpustack"`  |
| `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` | `env: ANTHROPIC_BASE_URL` / `env: AZURE_OPENAI_API_KEY` | Anthropic endpoint used for `endpoint="anthropic"`  |
| `MODELS` | see below | List of `{label, model, endpoint}` dicts: each entry is used as **both** the planner and OCR model |
| `RAG_TOP_K` | `5` | Records retrieved per sample (split k/2 correct + k/2 failed)|
| `RAG_CAPACITY` | `100` | Total records per RAG store combo (correct + failed slots combined); halved into each slot's `slot_capacity` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for text embeddings |
| `RAG_SIMILARITY_THRESHOLD` | `0.5` | Correct-slot records below this weighted score are dropped |
| `RAG_FAILED_SIMILARITY_THRESHOLD` | `0.80` | Failed-slot records need a much higher score before being retireved |
| `REQUEST_TIMEOUT` | `60` | Per-request timeout in seconds |
| `LOG_INTERMEDIATES` | `True` | Save each preprocessing step's output image to `processed_images/` |
| `ENABLE_THINKING` | `False` | Passed to thinking-capable models via `thinking_kwargs()` |
| `THINKING_MODELS` | set of model names | Models that accept a `chat_template_kwargs.enable_thinking` flag |
| `NO_TEMPERATURE_MODELS` | set of model names | Models (Claude family) that reject the `temperature` parameter |

### Model entries

Each entry in `MODELS` is used as **both** the planner and OCR model for a given run:

```python
MODELS = [
    {
        "label":    "qwen3-vl-30b-a3b-instruct",
        "model":    "qwen3-vl-30b-a3b-instruct",
        "endpoint": "gpustack",   # "gpustack" (OpenAI-compatible) or "anthropic"
    },
]
```

Add more entries to evaluate multiple models in one run.

---

## Running the Pipeline

```bash
# Run all datasets in datasets/ with all configured models
python run_pipeline.py

# Specific dataset files
python run_pipeline.py --datasets datasets/freecity.json

# Limit samples per dataset (quick testing)
python run_pipeline.py --max-samples 10

# Choose specific models by index into config.MODELS
python run_pipeline.py --model-idx 0
python run_pipeline.py --model-idx 0 1

# Disable RAG entirely (fresh run, no memory read/write)
python run_pipeline.py --no-rag

# Resume an interrupted run (skip samples already recorded)
python run_pipeline.py --resume

# Only rerun samples that were wrong or errored
python run_pipeline.py --rerun-failed

# Load datasets and print the evaluation plan without running any model
python run_pipeline.py --dry-run
```

Full flag list: `python run_pipeline.py --help` (see `parse_args()` in `run_pipeline.py`).

---

## Execution Flow (per sample)

```
Image
  │
  ├─ 1. PlannerAgent.visual_analysis()
  │       → visual features dict + text description
  │         (noise, background, distortion, char_quality,
  │          captcha_type, is_math, is_chinese, char_count)
  │
  ├─ 2. RAGStore.query_split()
  │       → top k/2 correct examples + top k/2 failed examples
  │         (ranked by 0.5×text_similarity + 0.5×pHash_similarity)
  │
  ├─ 3. PlannerAgent.plan()
  │       → preprocessing pipeline JSON + reasoning
  │         (CoT guided by retrieved examples)
  │
  ├─ 4. ExecutorAgent.execute()
  │       → run preprocessing tools sequentially
  │       → OCR with type-specific prompt (alphanumeric / math / chinese)
  │       → math solving post-OCR, with a math-reclassification fallback
  │         if an alphanumeric-classified image OCRs as an equation
  │
  └─ 5. RAGStore.add()
          → store the result (success or failure) for future retrieval
```

This is orchestrated by `CaptchaPipeline.process_sample()` in `pipeline.py`.

---

## Planner Agent (`agents/planner_agent.py`)

Wraps a VLM to perform two LLM calls per sample:

1. **`visual_analysis(image_b64)`** : examines the raw CAPTCHA and extracts a structured features dict (noise, background, distortion, overlap, char_quality, captcha_type, char_count, is_math, is_chinese, description). Prompted by `VISUAL_ANALYSIS_SYSTEM` in `agents/planner_prompts.py`.
2. **`plan(features, correct_records, failed_records)`** : reasons independently from the visual features plus RAG-retrieved examples (framed as soft hints, not hard rules) and emits a structured JSON preprocessing pipeline. Prompted by `COT_SYSTEM` in `agents/planner_prompts.py`.

Both calls go through `_chat()`, which uses the shared `call_with_retry` helper (`utils/llm_client.py`) for retry/back-off and connection-error classification. Raw model output is parsed with a fence-stripping + regex-fallback JSON parser (`_parse_json`), then reconciled against the known tool whitelist (`_normalize_plan`).

---

## RAG System (`utils/rag.py`)

### Architecture

- **Text query** → sentence-transformer embedding (`all-MiniLM-L6-v2`, 384-dim) → cosine similarity, computed on the planner's image `description`.
- **Image query** → per-channel RGB perceptual hash (pHash, 192-dim: 64 bits × R/G/B) → cosine similarity.
- Both scores are weighted equally (0.5 / 0.5) into one similarity score.

### Storage

Each `RAGStore` instance is one JSON-backed database, isolated per `(dataset, planner_model, ocr_model)` combo by `run_pipeline.py`. Records are bucketed into two slots, each capped at `slot_capacity` (evicting the least-retrieved record when full):

- **`correct`** : past successes.
- **`failed`** : past failures, additionally tagged with an `ocr_failure_type` (`duplicate_chars`, `empty_output`, `over_generation`, `under_generation`, `wrong_text`).


### Retrieval

`query_split()` is the primary retrieval path used by the pipeline: top `k/2` from the `correct` slot and top `k/2` from the `failed` slot, each ranked independently by weighted similarity, so the planner sees both "what worked" and "what to avoid." Failed-slot records use a separate, higher `RAG_FAILED_SIMILARITY_THRESHOLD` so only very similar failures are surfaced. `RAGStore.build_ocr_hints()` turns retrieved failure tags into short warning strings passed to the OCR step.

---

## Preprocessing Tools (`tools/`)

All tools accept and return base64-encoded images. `tools/registry.py` wraps the numpy-based `toolbox.py` functions automatically so the executor never handles raw pixel data directly.

| Tool | Trigger condition | Implementation |
|---|---|---|
| `color_extract` | Coloured (non-black) wavy or diagonal lines cross the text | Per-colour pixel masking (`tools/color_lines_removal.py`) : discards wide-spanning colour bands |
| `sharpen` | `distortion = "wave"` or `"blur"`  edges softened by distortion | Unsharp masking |
| `remove_small_noise` | Residual dots after `color_extract`, or patterned/textured backgrounds | Connected-component area filter |
| `remove_straight_line` | `noise = "strikethrough"` one thin dark line crosses all characters | Line-path fitting + TELEA inpainting + background-subtraction contrast enhancement |
| `binarize` | `char_quality = "faded"` gray (not pure black) ink on a clean background | Otsu's global threshold |

### Tool Ordering Rules

The canonical order enforced in the planner prompt:

```
color_extract → remove_small_noise → remove_straight_line → sharpen
```

- `color_extract` must come **before** `sharpen` or `binarize` 
- `remove_small_noise` follows `color_extract` to clean residual dot artefacts from colour separation.
- `sharpen` is never applied to `char_quality = "hollow"` 
- The planner defaults to an empty pipeline (`preprocess_pipeline = []`) whenever the image looks clean

To add a new tool: implement it in `toolbox.py` (or as a standalone base64-native module like `color_lines_removal.py`), then register it in `tools/registry.py`'s `TOOL_REGISTRY` and add it to `PlannerAgent._IMAGE_TOOLS` so the planner is allowed to select it.

---

## OCR (`tools/ocr_tool.py`)

`run_ocr()` calls a VLM with a **type-specific prompt** selected by `captcha_type` (`alphanumeric`, `math`, or `chinese`) from the planner's visual analysis. Uses `max_tokens=4096` and `temperature=0.1`)

Endpoint routing (OpenAI-compatible vs. Anthropic message/response format) is handled centrally in `utils/llm_client.py`.

---

## Math Handling (`utils/solve_math.py`)

Post-OCR math solving handles three common output forms:

| Form | Example | Handling |
|---|---|---|
| Answer embedded | `17+16=33` | Extract digit(s) after `=` directly |
| Unknown marker | `4+5=?` | Strip `?`, evaluate expression |
| Empty after `=` | `34-66=` | Strip `=`, evaluate expression |

**Misclassification recovery**: if the planner classifies a CAPTCHA as `alphanumeric` but the OCR output is purely digits + operators with at least one operator, the executor automatically retries OCR with the math prompt and solves the result.

---

## Dataset Format

```json
[
  {
    "id": "001",
    "label": "4vNf",
    "b64_string": "<base64-encoded PNG, with or without data-URI prefix>"
  }
]
```

---

## Output

Results are saved per `(dataset, planner, ocr)` combo and as a global summary, under `results/` (or `results_no_rag/` when run with `--no-rag`):

```
results/
├── summary.json                          # All combos combined
├── summary.csv
└── {dataset}__{planner}__{ocr}.json      # Per-combo results (+ matching .csv)
```

Each result record contains: `dataset`, `sample_id`, `ground_truth`, `prediction`, `correct`, `planner_model`, `ocr_model`, `captcha_type`, `tools_used`, `planned_tools`, `executor_steps`, `planner_reasoning`, `ocr_raw`, `rag_retrieval_sec`, `duration_sec`, `error`.

RAG databases are saved per combo under `rag_store/{dataset}__{planner}__{ocr}.json`. 

Per-run structured logs (JSONL + human-readable transcript) are saved under `logs/{run_id}/`.
