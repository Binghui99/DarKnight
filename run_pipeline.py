"""
run_pipeline.py
Entry point for Dark Knight
Loop structure:
  for each dataset:
    for each model (used as both planner and executor):
      run all samples → record results

At the end, save results to CSV / JSON and print a summary.

Usage
-----
# Run with all auto-discovered datasets in datasets/ directory:
python run_pipeline.py

# Specify datasets explicitly:
python run_pipeline.py --datasets datasets/freecity.json datasets/other.json

# Limit samples per dataset:
python run_pipeline.py --max-samples 10

# Choose specific models by index into config.MODELS:
python run_pipeline.py --model-idx 0
python run_pipeline.py --model-idx 0 1
"""
import argparse
import os
import sys
import time
import uuid

# Make sure local modules resolve
sys.path.insert(0, os.path.dirname(__file__))

import config
from utils.dataset_loader  import load_json_dataset, discover_datasets
from utils.rag             import RAGStore
from utils.result_tracker  import ResultTracker
from utils.errors          import CONNECTION_ERROR_KEY
from utils.naming          import sanitize_tag
from pipeline              import CaptchaPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Captcha Agentic Pipeline – Evaluation")
    p.add_argument("--datasets",    nargs="*",   default=None,
                   help="Dataset file paths. If omitted, auto-discovers in datasets/")
    p.add_argument("--datasets-dir",             default=config.DATASETS_DIR,
                   help="Directory to scan for datasets (default: datasets/)")
    p.add_argument("--results-dir",              default=None,
                   help="Where to save results (default: results/ or results_no_rag/ with --no-rag)")
    p.add_argument("--max-samples", type=int,    default=None,
                   help="Max samples per dataset (for quick testing)")
    p.add_argument("--model-idx",   nargs="*",   type=int, default=None,
                   help="Indices into config.MODELS to use (e.g. --model-idx 0 1)")
    p.add_argument("--rag-db",                   default=config.RAG_DB_PATH,
                   help="Path to RAG persistence file")
    p.add_argument("--rag-capacity",  type=int,  default=None,
                   help="Total RAG capacity across both slots (default: 200). "
                        "Must be even; each slot gets half.")
    p.add_argument("--no-rag",        action="store_true",
                   help="Disable RAG (start fresh each run)")
    p.add_argument("--dry-run",       action="store_true",
                   help="Load datasets and print plan only; do not run models")
    p.add_argument("--resume",        action="store_true",
                   help="Skip samples already present in results.json (resume interrupted run)")
    p.add_argument("--rerun-failed",  action="store_true",
                   help="Only rerun samples that were wrong or errored; skip already-correct ones")
    return p.parse_args()


def _discover_dataset_paths(args):
    dataset_paths = []
    if args.datasets:
        # If specific files passed to run
        for d in args.datasets:
            if os.path.isdir(d):
                dataset_paths.extend(discover_datasets(d, extensions=[".json"]))
            else:
                dataset_paths.append(d)
    else:
        # Default: auto-discover all .json files in the datasets directory
        dataset_paths = discover_datasets(args.datasets_dir, extensions=[".json"])
    return dataset_paths


def _select_models(args):
    all_models = config.MODELS
    return (
        [all_models[i] for i in args.model_idx]
        if args.model_idx else all_models
    )


def _load_global_tracker(args):
    global_tracker = ResultTracker(results_dir=args.results_dir,
                                   filename="summary.json")
    if args.resume or args.rerun_failed:
        loaded = global_tracker.load()
        if loaded:
            mode = "resume (skip all done)" if args.resume else "rerun-failed (skip correct)"
            print(f"[run_pipeline] Mode: {mode} — {loaded} previous result(s) loaded")
    return global_tracker


def _make_rag_store(args, rag_dir, safe_tag):
    # Per-combo RAG store, isolated per (dataset, planner, ocr)
    # so different model combinations never share retrieval memory.
    if args.no_rag:
        rag_path = os.path.join(
            args.results_dir, f".rag_tmp_{uuid.uuid4().hex}.json"
        )
    else:
        rag_path = os.path.join(rag_dir, f"{safe_tag}.json")
    _default_cap = config.RAG_CAPACITY
    _cap = args.rag_capacity if args.rag_capacity else _default_cap
    slot_cap = _cap // 2
    rag_store = RAGStore(
        db_path                     = rag_path,
        embedding_model             = config.EMBEDDING_MODEL,
        similarity_threshold        = config.RAG_SIMILARITY_THRESHOLD,
        failed_similarity_threshold = getattr(config, "RAG_FAILED_SIMILARITY_THRESHOLD", None),
        slot_capacity               = slot_cap,
    )
    print(f"  RAG store : {rag_path}")
    return rag_store


def _run_combo_samples(pipeline, samples, args, global_tracker, combo_tracker,
                        rag_store, dataset_name, planner_cfg, ocr_cfg):
    skipped   = 0
    processed = 0
    for sample in samples:
        sid = str(sample.get("id", ""))

        if args.resume and combo_tracker.is_done(
            dataset_name, sid, planner_cfg["label"], ocr_cfg["label"]
        ):
            skipped += 1
            continue

        if args.rerun_failed and (
            global_tracker.was_correct(dataset_name, sid, planner_cfg["label"], ocr_cfg["label"])
            or combo_tracker.was_correct(dataset_name, sid, planner_cfg["label"], ocr_cfg["label"])
        ):
            skipped += 1
            continue

        result = pipeline.process_sample(sample)

        if result.get(CONNECTION_ERROR_KEY):
            print(f"  [connection_error] sample {sid} skipped "
                  f"(will rerun on --resume)")
            continue

        global_tracker.record(result)
        processed += 1

        #  save after every sample
        combo_tracker.save_json(silent=True)
        combo_tracker.save_csv(silent=True)
        global_tracker.save_json(silent=True)
        rag_store.save(silent=True)

    if skipped:
        print(f"  [skip] {skipped} sample(s) skipped")


def _run_combo(dataset_path, dataset_name, samples, model_cfg, args,
                rag_dir, global_tracker, combo_count, total_combos):
    planner_cfg = model_cfg
    ocr_cfg     = model_cfg
    combo_label = model_cfg["label"]
    print(f"\n[Run {combo_count}/{total_combos}]  "
          f"dataset={dataset_name}  model={combo_label}")

    # Per-combo tag used for both result and RAG filenames, e.g.:
    #   results/freecity__qwen3.5-9b__step3-vl-10b.json
    #   rag_store/freecity__qwen3.5-9b__step3-vl-10b.json
    safe_tag = sanitize_tag(
        f"{dataset_name}__{planner_cfg['label']}__{ocr_cfg['label']}"
    )

    rag_store = _make_rag_store(args, rag_dir, safe_tag)

    combo_tracker = ResultTracker(
        results_dir = args.results_dir,
        filename    = f"{safe_tag}.json",
    )
    if args.resume or args.rerun_failed:
        combo_tracker.load()

    pipeline = CaptchaPipeline(
        planner_cfg  = planner_cfg,
        ocr_cfg      = ocr_cfg,
        rag_store    = rag_store,
        tracker      = combo_tracker,
        dataset_name = dataset_name,
        use_rag      = not args.no_rag,
    )

    _run_combo_samples(pipeline, samples, args, global_tracker, combo_tracker,
                        rag_store, dataset_name, planner_cfg, ocr_cfg)

    pipeline.close()


def main():
    args = parse_args()
    if args.results_dir is None:
        args.results_dir = "results_no_rag" if args.no_rag else config.RESULTS_DIR
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(config.LOGS_DIR,  exist_ok=True)
    datasets_dir = config.DATASETS_DIR

    # Load datasets
    dataset_paths = _discover_dataset_paths(args)

    if not dataset_paths:
        print(f"Error: No .json datasets found in {args.datasets_dir}")
        return

    print(f"\n[run_pipeline] Found {len(dataset_paths)} dataset(s):")
    for p in dataset_paths:
        print(f"  • {p}")

    # Load models
    models = _select_models(args)

    total_combos = len(dataset_paths) * len(models)
    print(f"\n[run_pipeline] Evaluation plan:")
    print(f"  Datasets : {len(dataset_paths)}")
    print(f"  Models   : {[m['label'] for m in models]}")
    print(f"  Total    : {total_combos} run(s)")

    if args.dry_run:
        print("\n[run_pipeline] DRY RUN – exiting.")
        return

    # RAG stores directory
    rag_dir = os.path.dirname(args.rag_db) or "rag_store"
    os.makedirs(rag_dir, exist_ok=True)

    global_tracker = _load_global_tracker(args)

    # Loop start
    combo_count = 0
    t_total     = time.time()

    for dataset_path in dataset_paths:
        print(f"\n{'═'*70}")
        print(f"Loading dataset: {dataset_path}")
        try:
            samples = load_json_dataset(dataset_path)
        except Exception as exc:
            print(f"  ✗ Could not load: {exc}")
            continue

        if args.max_samples:
            samples = samples[: args.max_samples]

        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        print(f"  {len(samples)} sample(s) from '{dataset_name}'")

        for model_cfg in models:
            combo_count += 1
            _run_combo(dataset_path, dataset_name, samples, model_cfg, args,
                       rag_dir, global_tracker, combo_count, total_combos)

    # Final save + summary
    global_tracker.save_json()
    global_tracker.save_csv()
    global_tracker.print_summary()

    print(f"\n[run_pipeline] Done in {time.time() - t_total:.1f}s  "
          f"Results → {args.results_dir}/")


if __name__ == "__main__":
    main()