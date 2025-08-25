# src/lit2vec_tldr/metrics.py

import nltk
import evaluate

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt", quiet=True)

# Load ROUGE metric once
rouge = evaluate.load("rouge")


def postprocess_text(preds, labels):
    """
    Clean up model predictions and labels:
    - Strip leading/trailing spaces
    - Sentence split with nltk
    """
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]

    preds = ["\n".join(nltk.sent_tokenize(p)) for p in preds]
    labels = ["\n".join(nltk.sent_tokenize(l)) for l in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer, log_to_wandb=False, wandb=None):
    """
    Compute ROUGE scores for Seq2SeqTrainer.

    Args:
        eval_preds (tuple): predictions and labels returned by HF Trainer
        tokenizer (PreTrainedTokenizer): tokenizer used for decoding
        log_to_wandb (bool): if True, logs metrics to Weights & Biases
        wandb (module): wandb instance if logging is enabled
    """
    preds, labels = eval_preds

    # HF Trainer returns a tuple sometimes
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode generated tokens
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 (ignore_index) in labels before decoding
    labels = [[l if l != -100 else tokenizer.pad_token_id for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean + sentence split
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Compute ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Format nicely
    result = {k: round(v * 100, 2) for k, v in result.items()}

    if log_to_wandb and wandb is not None:
        wandb.log(result)

    return result
