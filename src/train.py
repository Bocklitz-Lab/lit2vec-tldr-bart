# src/train.py
"""Fine-tune DistilBART on the Lit2Vec TL;DR dataset.

Run from the repository root:

    python src/train.py --config configs/train_distilbart.yaml

Settings come from the YAML file; command-line options override it. Anything that is
missing falls back to the defaults in DEFAULTS below, which reproduce the training run
of the released model. The earlier single-file version of this script is
scripts/train.py.
"""

import os
import json
import argparse

import torch
import yaml
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from preprocess import tokenize_dataset
from metrics import compute_metrics
from utils import set_seed, setup_logger, ensure_dir, get_device, get_device_name, save_json


# -----------------
# Defaults
# -----------------

DATASET_ID = "Bocklitz-Lab/lit2vec-tldr-bart-dataset"
MODEL_CHECKPOINT = "sshleifer/distilbart-cnn-12-6"

DEFAULTS = {
    # run / tracking
    "project_name": "lit2vec_tldr_bart",
    "wandb_entity": None,
    "run_name": "lit2vec_distilbart_run",
    "use_wandb": True,
    "seed": 42,
    # data / model
    "dataset_id": DATASET_ID,
    "model_checkpoint": MODEL_CHECKPOINT,
    "max_input_length": 1024,
    "max_target_length": 128,
    # paths
    "tokenized_cache_path": "./saved_data/cached_tokenized_data",
    "checkpoint_dir": "./saved_data/checkpoints",
    "final_model_dir": "./saved_data/final_model",
    "evaluation_dir": "./evaluation",
    # training
    "train_batch_size": 4,
    "eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "num_train_epochs": 5,
    "fp16": "auto",
    # scheduling / logging
    "logging_steps": 500,
    "eval_steps": 1000,
    "save_steps": 1000,
    "save_total_limit": 2,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "logging_strategy": "steps",
    # Options that change the training data or objective. The defaults match the
    # released model; see the README ("Notes on the released model").
    "mask_label_padding": False,   # ignore padded label positions in the loss
    "drop_empty_pairs": False,     # drop records without an abstract or a summary
    # Debugging: use only the first N records of each split
    "max_train_samples": None,
    "max_eval_samples": None,
}

INT_KEYS = [
    "train_batch_size", "eval_batch_size", "gradient_accumulation_steps",
    "num_train_epochs", "max_input_length", "max_target_length", "seed",
    "logging_steps", "eval_steps", "save_steps", "save_total_limit",
]
TEXT_COLUMNS = [
    "corpus_id", "doi", "title", "authors", "venue", "year",
    "fields_of_study", "publication_date", "abstract", "summary",
    "license_type", "license_publisher", "license_url", "source_url",
]


# -----------------
# Helpers
# -----------------

def _cache_is_current(cache_path: str, signature: dict) -> bool:
    """The tokenized cache is only reused if it was built with the same settings."""
    meta_path = os.path.join(cache_path, "cache_signature.json")
    if not (os.path.isdir(cache_path) and os.path.exists(meta_path)):
        return False
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f) == signature


def _limit(split, n):
    return split.select(range(min(n, len(split)))) if n else split


# -----------------
# Training
# -----------------

def main(args):
    logger = setup_logger("train")
    set_seed(args.seed)

    device = get_device()
    logger.info(f"Using device: {get_device_name(device)}")

    use_wandb = bool(args.use_wandb)
    wandb = None
    if use_wandb:
        import wandb
        wandb.init(
            project=args.project_name,
            entity=args.wandb_entity,
            config={
                "model_checkpoint": args.model_checkpoint,
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "seed": args.seed,
                "mask_label_padding": args.mask_label_padding,
                "drop_empty_pairs": args.drop_empty_pairs,
                "device": str(device),
            },
            notes="Fine-tuning DistilBART on the Lit2Vec TL;DR dataset",
            tags=["summarization", "chemistry", "distilbart"],
        )

    # ==== STEP 1: LOAD DATASET ====
    logger.info("Loading dataset...")
    base_url = f"https://huggingface.co/datasets/{args.dataset_id}/resolve/main/data"
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{base_url}/train.jsonl",
            "validation": f"{base_url}/validation.jsonl",
            "test": f"{base_url}/test.jsonl",
        },
    )
    if args.drop_empty_pairs:
        before = {k: len(v) for k, v in dataset.items()}
        dataset = dataset.filter(lambda r: bool((r["abstract"] or "").strip()) and bool((r["summary"] or "").strip()))
        after = {k: len(v) for k, v in dataset.items()}
        logger.info(f"Dropped records without abstract or summary: {before} -> {after}")
    for split, n in (("train", args.max_train_samples), ("validation", args.max_eval_samples), ("test", args.max_eval_samples)):
        dataset[split] = _limit(dataset[split], n)

    # ==== STEP 2: TOKENIZER & MODEL ====
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    # ==== STEP 3: TOKENIZE OR LOAD CACHE ====
    signature = {
        "dataset_id": args.dataset_id,
        "model_checkpoint": args.model_checkpoint,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "mask_label_padding": bool(args.mask_label_padding),
        "drop_empty_pairs": bool(args.drop_empty_pairs),
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
    }
    if _cache_is_current(args.tokenized_cache_path, signature):
        logger.info("Loading tokenized dataset from disk...")
        tokenized_datasets = load_from_disk(args.tokenized_cache_path)
    else:
        logger.info("Tokenizing dataset...")
        tokenized_datasets = tokenize_dataset(
            dataset,
            tokenizer,
            input_col="abstract",
            target_col="summary",
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            remove_columns=TEXT_COLUMNS,
            mask_label_padding=bool(args.mask_label_padding),
        )
        ensure_dir(args.tokenized_cache_path)
        tokenized_datasets.save_to_disk(args.tokenized_cache_path)
        save_json(signature, os.path.join(args.tokenized_cache_path, "cache_signature.json"))
        logger.info(f"Tokenized dataset saved to {args.tokenized_cache_path}")

    # ==== STEP 4: DATA COLLATOR ====
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ==== STEP 5: TRAINING ARGS ====
    fp16 = torch.cuda.is_available() if args.fp16 == "auto" else bool(args.fp16)
    ensure_dir(args.checkpoint_dir)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.checkpoint_dir,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=fp16,
        seed=args.seed,
        report_to="wandb" if use_wandb else "none",
        run_name=args.run_name,
    )

    # ==== STEP 6: TRAINER ====
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, tokenizer, log_to_wandb=use_wandb, wandb=wandb
        ),
    )

    # ==== STEP 7: TRAIN OR RESUME ====
    last_checkpoint = None
    if os.path.isdir(args.checkpoint_dir):
        checkpoints = [
            os.path.join(args.checkpoint_dir, d)
            for d in os.listdir(args.checkpoint_dir)
            if d.startswith("checkpoint")
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getctime)

    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("Starting training from scratch...")
        trainer.train()

    # ==== STEP 8: SAVE FINAL MODEL ====
    logger.info("Saving final model...")
    ensure_dir(args.final_model_dir)
    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    logger.info(f"Final model saved to {args.final_model_dir}")

    # ==== STEP 9: FINAL TEST EVALUATION ====
    logger.info("Running final evaluation on the test split...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")

    metrics_path = os.path.join(args.evaluation_dir, "test_metrics.json")
    ensure_dir(args.evaluation_dir)
    save_json(test_metrics, metrics_path)
    logger.info(f"Test metrics saved to {metrics_path}")

    if use_wandb:
        wandb.log(test_metrics)
        wandb.finish()


# -----------------
# CLI
# -----------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune DistilBART on the Lit2Vec TL;DR dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config")

    # Command-line overrides (applied on top of the YAML file)
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    cli = parser.parse_args(argv)

    config = {}
    if cli.config:
        with open(cli.config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    overrides = {k: v for k, v in vars(cli).items() if v is not None and k not in ("config", "no_wandb")}
    if cli.no_wandb:
        overrides["use_wandb"] = False

    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        print(f"Warning: ignoring unknown config keys: {', '.join(unknown)}")

    merged = {**DEFAULTS, **{k: v for k, v in config.items() if k in DEFAULTS}, **overrides}
    for key in INT_KEYS:
        merged[key] = int(merged[key])
    merged["learning_rate"] = float(merged["learning_rate"])   # YAML reads 2e-5 as a string
    for key in ("max_train_samples", "max_eval_samples"):
        merged[key] = int(merged[key]) if merged[key] else None
    return argparse.Namespace(**merged)


if __name__ == "__main__":
    main(parse_args())
