"""Original single-file training script of the released model.

src/train.py is the configurable version of the same procedure (see the README).
To train on a particular GPU, set CUDA_VISIBLE_DEVICES in the shell.
"""
import os
from tqdm import tqdm

import nltk
nltk.download('punkt')

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate
import wandb

# ==== CONFIGURATION ====
PROJECT_NAME = "distilbart_cnn_summarizer"
WANDB_ENTITY = None   # or your W&B team name

DATASET_ID = "Bocklitz-Lab/lit2vec-tldr-bart-dataset"
TOKENIZED_CACHE_PATH = "./saved_data/cached_tokenized_data"
CHECKPOINT_DIR = "./saved_data/checkpoints"
FINAL_MODEL_DIR = "./saved_data/final_model"
MODEL_CHECKPOINT = "sshleifer/distilbart-cnn-12-6"

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128

# ==== DEVICE ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== WANDB INIT ====
wandb.init(
    project=PROJECT_NAME,
    entity=WANDB_ENTITY,
    config={
        "model_checkpoint": MODEL_CHECKPOINT,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "num_train_epochs": 5,
        "device": str(device),
    },
    notes="Finetuning DistilBART on Lit2Vec TL;DR dataset",
    tags=["NLP", "summarization", "chemistry", "finetune"]
)

# ==== STEP 1: LOAD DATASET FROM HF ====
print("Loading dataset from Hugging Face Hub...")
dataset = load_dataset(DATASET_ID)

print(dataset)

# ==== STEP 2: LOAD TOKENIZER & MODEL ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

# ==== STEP 3: TOKENIZE OR LOAD CACHE ====
if os.path.exists(TOKENIZED_CACHE_PATH):
    print("Loading tokenized dataset from disk...")
    from datasets import load_from_disk
    tokenized_datasets = load_from_disk(TOKENIZED_CACHE_PATH)
else:
    print("Tokenizing dataset...")

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["abstract"],
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True
        )
        labels = tokenizer(
            examples["summary"],
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=[
            "corpus_id", "doi", "title", "authors", "venue", "year",
            "fields_of_study", "publication_date", "abstract", "summary",
            "license_type", "license_publisher", "license_url", "source_url"
        ]
    )

    os.makedirs(os.path.dirname(TOKENIZED_CACHE_PATH), exist_ok=True)
    tokenized_datasets.save_to_disk(TOKENIZED_CACHE_PATH)
    print(f"Tokenized dataset saved to {TOKENIZED_CACHE_PATH}")

# ==== STEP 4: DATA COLLATOR ====
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ==== STEP 5: EVALUATION METRICS ====
rouge = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    preds = ["\n".join(nltk.sent_tokenize(p)) for p in preds]
    labels = ["\n".join(nltk.sent_tokenize(l)) for l in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    result = {k: round(v * 100, 2) for k, v in result.items()}
    wandb.log(result)
    return result

# ==== STEP 6: TRAINING ARGUMENTS ====
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=500,
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
    report_to="wandb",
    run_name="lit2vec_distilbart_run"
)

# ==== STEP 7: TRAINER ====
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ==== STEP 8: TRAIN OR RESUME ====
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
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting training from scratch...")
    trainer.train()

# ==== STEP 9: SAVE FINAL MODEL ====
print("Saving final model...")
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"Final model saved to {FINAL_MODEL_DIR}")

# ==== STEP 10: FINAL TEST EVALUATION ====
print("Running final evaluation on test split...")
test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")
print("Test metrics:", test_metrics)

# log to W&B as well
wandb.log(test_metrics)

wandb.finish()

