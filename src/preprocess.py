# src/preprocess.py

from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


def _to_str_list(values: Union[List[object], object]) -> List[str]:
    """
    Ensure we always pass a list of strings to the tokenizer.
    Converts None -> "", non-strings -> str(value).
    """
    if isinstance(values, list):
        return [("" if v is None else str(v)) for v in values]
    # Single value (shouldn't happen with batched=True, but safe-guard anyway)
    return ["" if values is None else str(values)]


def get_preprocess_function(
    tokenizer: PreTrainedTokenizerBase,
    *,
    input_col: str = "abstract",
    target_col: str = "summary",
    max_input_length: int = 1024,
    max_target_length: int = 128,
    padding: str = "max_length",  # "max_length" or "longest"
    truncation: bool = True,
    mask_label_padding: bool = False,
) -> Callable[[Dict[str, List[object]]], Dict[str, List[List[int]]]]:
    """
    Returns a batched preprocessing function for Seq2Seq tokenization.

    Produces:
      - input_ids, attention_mask for inputs
      - labels for targets

    Note on label padding: with padding="max_length" the labels are already padded
    to max_target_length with the tokenizer's pad id, and DataCollatorForSeq2Seq does
    not change them, so the loss is also computed on the padding positions. This is how
    the released model was trained, so it is the default. Set mask_label_padding=True to
    replace the pad ids in the labels with -100 so that the loss ignores them.

    Args:
        tokenizer: HF tokenizer for the model.
        input_col: dataset column name containing the source text.
        target_col: dataset column name containing the target text.
        max_input_length: max tokens for inputs.
        max_target_length: max tokens for targets.
        padding: "max_length" or "longest".
        truncation: whether to truncate sequences.
        mask_label_padding: replace pad ids in the labels with -100 (see above).

    Returns:
        A function suitable for Dataset.map(batched=True).
    """
    def _preprocess(examples: Dict[str, List[object]]) -> Dict[str, List[List[int]]]:
        # Coerce to strings to avoid TypeError: TextEncodeInput...
        inputs = _to_str_list(examples.get(input_col, []))
        targets = _to_str_list(examples.get(target_col, []))

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            padding=padding,
            truncation=truncation,
        )

        # Tokenize targets (labels)
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=truncation,
        )

        label_ids = labels["input_ids"]
        if mask_label_padding:
            pad_id = tokenizer.pad_token_id
            label_ids = [[(-100 if t == pad_id else t) for t in seq] for seq in label_ids]

        model_inputs["labels"] = label_ids
        return model_inputs

    return _preprocess


def tokenize_dataset(
    dataset: DatasetDict | Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    input_col: str = "abstract",
    target_col: str = "summary",
    max_input_length: int = 1024,
    max_target_length: int = 128,
    padding: str = "max_length",
    truncation: bool = True,
    remove_columns: Optional[Iterable[str]] = None,
    num_proc: Optional[int] = None,
    mask_label_padding: bool = False,
) -> DatasetDict | Dataset:
    """
    Applies seq2seq tokenization to a Dataset or DatasetDict.

    Args:
        dataset: HF Dataset or DatasetDict (with train/validation/test).
        tokenizer: HF tokenizer.
        input_col, target_col: column names for text fields.
        max_input_length, max_target_length, padding, truncation: tokenization params.
        remove_columns: columns to drop after tokenization. If None, auto-drop input/target.
        num_proc: optional multiprocessing workers for `.map`.
        mask_label_padding: replace pad ids in the labels with -100.

    Returns:
        Tokenized Dataset / DatasetDict containing input_ids, attention_mask, labels.
    """
    preprocess_fn = get_preprocess_function(
        tokenizer,
        input_col=input_col,
        target_col=target_col,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        padding=padding,
        truncation=truncation,
        mask_label_padding=mask_label_padding,
    )

    # Auto-remove text columns if not specified
    if remove_columns is None:
        if isinstance(dataset, DatasetDict):
            sample_split = next(iter(dataset.values()))
            cols = sample_split.column_names
        else:
            cols = dataset.column_names
        remove_columns = [c for c in cols if c in (input_col, target_col)]

    tokenized = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=list(remove_columns),
        desc="Tokenizing dataset",
        num_proc=num_proc,
    )
    return tokenized
