# src/lit2vec_tldr/train.py

import os
import argparse
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

import wandb

from preprocess import tokenize_dataset
from metrics import compute_metrics
from utils import set_seed, setup_logger, ensure_dir, get_device, get_device_name, save_json


# -----------------
# Config
# -----------------

DATASET_ID = "Bocklitz-Lab/lit2vec-tldr-bart-dataset"
MODEL_CHECKPOINT = "sshleifer/distilbart-cnn-12-6"

TOKENIZED_CACHE_PATH = "./saved_data/cached_tokenized_data"
CHECKPOINT_DIR = "./saved_data/checkpoints"
FINAL_MODEL_DIR = "./saved_data/final_model"

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
SEED = 42


# -----------------
# Training Function
# -----------------

def main(args):
    logger = setup_logger("train")

    # Reproducibility
    set_seed(SEED)

    # Device
    device = get_device()
    logger.info(f"✅ Using device: {get_device_name(device)}")

    # Init Weights & Biases
    wandb.init(
        project=args.project_name,
        entity=args.wandb_entity,
        config={
            "model_checkpoint": MODEL_CHECKPOINT,
            "max_input_length": MAX_INPUT_LENGTH,
            "max_target_length": MAX_TARGET_LENGTH,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "seed": SEED,
            "device": str(device),
        },
        notes="Fine-tuning DistilBART on Lit2Vec TL;DR dataset",
        tags=["summarization", "chemistry", "distilbart"],
    )

    # ==== STEP 1: LOAD DATASET ====
    logger.info("📄 Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/data/train.jsonl",
            "validation": f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/data/validation.jsonl",
            "test": f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/data/test.jsonl",
        }
    )

    # ==== STEP 2: TOKENIZER & MODEL ====
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # ==== STEP 3: TOKENIZE OR LOAD CACHE ====
    if os.path.exists(TOKENIZED_CACHE_PATH):
        logger.info("🔁 Loading tokenized dataset from disk...")
        tokenized_datasets = load_from_disk(TOKENIZED_CACHE_PATH)
    else:
        logger.info("✂️  Tokenizing dataset...")
        tokenized_datasets = tokenize_dataset(
            dataset,
            tokenizer,
            input_col="abstract",
            target_col="summary",
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
            remove_columns=[
                "corpus_id", "doi", "title", "authors", "venue", "year",
                "fields_of_study", "publication_date", "abstract", "summary",
                "license_type", "license_publisher", "license_url", "source_url",
            ],
        )
        ensure_dir(TOKENIZED_CACHE_PATH)
        tokenized_datasets.save_to_disk(TOKENIZED_CACHE_PATH)
        logger.info(f"✅ Tokenized dataset saved to {TOKENIZED_CACHE_PATH}")

    # ==== STEP 4: DATA COLLATOR ====
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ==== STEP 5: TRAINING ARGS ====
    ensure_dir(CHECKPOINT_DIR)
    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=500,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
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
            eval_preds, tokenizer, log_to_wandb=True, wandb=wandb
        ),
    )

    # ==== STEP 7: TRAIN OR RESUME ====
    last_checkpoint = None
    if os.path.isdir(CHECKPOINT_DIR):
        checkpoints = [
            os.path.join(CHECKPOINT_DIR, d)
            for d in os.listdir(CHECKPOINT_DIR)
            if d.startswith("checkpoint")
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getctime)

    if last_checkpoint:
        logger.info(f"🚀 Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("🚀 Starting training from scratch...")
        trainer.train()

    # ==== STEP 8: SAVE FINAL MODEL ====
    logger.info("💾 Saving final model...")
    ensure_dir(FINAL_MODEL_DIR)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    logger.info(f"✅ Final model saved to {FINAL_MODEL_DIR}")

    # ==== STEP 9: FINAL TEST EVALUATION ====
    logger.info("🧪 Running final evaluation on test split...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")
    logger.info(f"📊 Test metrics: {test_metrics}")

    # Save test metrics JSON
    ensure_dir("evaluation")
    save_json(test_metrics, "evaluation/test_metrics.json")
    logger.info("✅ Test metrics saved to evaluation/test_metrics.json")

    # Log to W&B
    wandb.log(test_metrics)
    wandb.finish()


# -----------------
# CLI
# -----------------

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Fine-tune DistilBART on Lit2Vec TL;DR dataset")
    parser.add_argument("--config", type=str, help="Path to YAML config", default=None)

    # CLI overrides
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    cli_args = parser.parse_args()

    # Load YAML config if provided
    config = {}
    if cli_args.config:
        with open(cli_args.config) as f:
            config = yaml.safe_load(f)

    # Merge CLI overrides on top of config values
    merged = {**config, **{k: v for k, v in vars(cli_args).items() if v is not None}}

    # Wrap back into argparse.Namespace so main() doesn’t change
    args = argparse.Namespace(**merged)
    
    args.train_batch_size = int(args.train_batch_size)
    args.eval_batch_size = int(args.eval_batch_size)
    args.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    args.learning_rate = float(args.learning_rate)
    args.num_train_epochs = int(args.num_train_epochs)
    args.max_input_length = int(args.max_input_length)
    args.max_target_length = int(args.max_target_length)
    args.seed = int(args.seed)

    main(args)

