# lit2vec-tldr-bart

Training and evaluation code for the TL;DR summarisation model of the Lit2Vec workflow. The model is [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) fine-tuned on 19,992 CC-BY chemistry abstracts with GPT-4o-generated summaries. The code accompanies the paper:

> Amiri M, Jafari JM, Mostafapour S, Bocklitz T. *Lit2Vec: A Reproducible Workflow for Building a Legally Screened Chemistry Corpus from S2ORC for Downstream Retrieval and Text Mining.* arXiv:2604.12498 (2026). https://doi.org/10.48550/arXiv.2604.12498

Related resources: [model](https://huggingface.co/Bocklitz-Lab/lit2vec-tldr-bart-model) · [dataset](https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-tldr-bart-dataset) · [demo](https://huggingface.co/spaces/Bocklitz-Lab/lit2vec-tldr-bart-space) · [main Lit2Vec repository](https://github.com/Bocklitz-Lab/Lit2Vec-code)

## Repository layout

```
configs/train_distilbart.yaml   training settings of the released model
src/train.py                    training script (reads the YAML config)
src/preprocess.py               tokenisation of abstracts and summaries
src/metrics.py                  ROUGE with NLTK sentence splitting
src/utils.py                    seeding, logging, small helpers
scripts/train.py                the original single-file training script
scripts/run_train.sh            wrapper around src/train.py that writes a log
scripts/flatten_and_split.py    builds the train/validation/test files
scripts/download_dataset.sh     downloads the three JSONL files
scripts/export_model.sh         ONNX export and check
notebooks/01_explore_dataset    statistics of the dataset
notebooks/02_eval_and_visualize benchmark against the baselines of the paper
```

## Setup

You need Python 3.10 or newer (we used 3.11) and, for training, an NVIDIA GPU. The pinned versions match the environment that was used for the released model.

```bash
git clone https://github.com/Bocklitz-Lab/lit2vec-tldr-bart.git
cd lit2vec-tldr-bart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

The extra index provides the CUDA 12.1 build of PyTorch 2.4.1. A `Dockerfile` with the same dependencies is included; we have not built it as part of this clean-up.

## Training

```bash
python src/train.py --config configs/train_distilbart.yaml
```

Logging to Weights & Biases is on by default (`wandb login` first); add `--no_wandb` to turn it off. Checkpoints go to `saved_data/checkpoints/`, the final model to `saved_data/final_model/`, and the metrics of the test split to `evaluation/test_metrics.json`. If checkpoints exist, training resumes from the newest one. Command-line options override the YAML file, and any setting that is missing falls back to the default in `src/train.py`. The data are read directly from the dataset repository on Hugging Face.

The released model was trained with these settings, which the saved `training_args.bin` of the model confirms:

| | |
|---|---|
| Base model | `sshleifer/distilbart-cnn-12-6` |
| Input and target length | 1,024 and 128 tokens |
| Optimiser | AdamW, learning rate 2e-5, linear schedule, no warm-up |
| Batch size | 4 per device, 4 gradient-accumulation steps (effective 16) |
| Epochs | 5 |
| Precision | fp16 |
| Seed | 42 |
| Hardware | one NVIDIA RTX 3090 |

`scripts/train.py` is the original single-file script with the same settings. `src/train.py` is the configurable version of it.

## Data

The dataset has 17,992 training, 999 validation and 1,001 test records, and it is described in its [dataset card](https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-tldr-bart-dataset). `scripts/flatten_and_split.py` produced the three files from one annotated JSON file with a seeded random 90% / 5% / 5% split. That annotated input file is not published, so the split cannot be regenerated from this repository. Use the released files.

## Evaluation

`notebooks/02_eval_and_visualize.ipynb` evaluates the fine-tuned model and the baselines of the paper on the test split. It keeps the 997 test records that have both an abstract and a summary, generates with beam size 4, and computes ROUGE and BERTScore. The notebook output (recorded in the committed notebook) is:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BERTScore F1 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| lit2vec_tldr_finetuned | 56.38 | 31.00 | 44.28 | 45.64 | 91.46 | 255 |
| distilbart-cnn-12-6 (zero-shot) | 37.67 | 15.57 | 26.68 | 30.54 | 87.57 | 319 |
| bart-large-cnn (zero-shot) | 37.00 | 15.02 | 25.90 | 30.01 | 87.44 | 403 |
| LexRank | 35.65 | 14.78 | 24.73 | 28.57 | 87.22 | 15 |
| Lead-3 | 33.74 | 13.79 | 23.53 | 27.44 | 86.71 | 13 |

Two remarks on the baselines. The SciBERT-based extractive row of the notebook is identical to Lead-3 in every metric, which suggests that every call fell back to the Lead-3 branch of the code, so it is left out of the table. The PEGASUS baseline was skipped because `sentencepiece` was not installed. It is listed as an optional requirement.

The numbers on the model card (ROUGE-1 56.11, ROUGE-2 30.78, ROUGE-Lsum 45.43) come from the evaluation at the end of training. That evaluation uses all 1,001 test records, including 4 whose reference summary is empty. We regenerated the test summaries with the released model and obtained 56.12, 30.78 and 45.44 on all 1,001 records, and 56.34, 30.90 and 45.62 on the 997 records with a reference.

## Notes on the released model

Two properties of the training run are worth knowing if you reuse the code. Both are kept as the default so that `src/train.py` reproduces the released model, and both can be changed in the config.

- **Padded label positions are part of the loss.** The targets are padded to 128 tokens with the pad token, and `DataCollatorForSeq2Seq` only masks padding that it adds itself, so the padded positions are predicted and counted in the loss. In the training split they make up 56% of all label positions (72 of 128 on average). As a consequence the training and evaluation losses that were logged are dominated by padding and are not comparable with losses from a setup that masks it. In a short test run on 16 examples, the evaluation loss was 8.4 with padding included and 2.2 with `mask_label_padding: true`. Setting that option replaces the pad ids in the labels with -100.
- **Records without a summary were used.** 84 records of the dataset have no summary (79 of them in the training split), and `scripts/train.py` does not filter them. Set `drop_empty_pairs: true` to drop records without an abstract or a summary.

The tokenised data are cached on disk. The cache stores the settings it was built with and is rebuilt when they change.

## Export

`scripts/export_model.sh` exports a trained model to ONNX and runs a short check with ONNX Runtime. It needs `onnx` and `onnxruntime` (see `requirements.txt`) and uses `transformers.onnx`, which is deprecated in recent versions of Transformers. `scripts/safetensor2pytorchbin.py` re-saves a model in the PyTorch `.bin` format.

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The model weights are released under Apache-2.0 and the dataset under CC BY 4.0. The summaries in the dataset were generated with GPT-4o, so check OpenAI's terms of use if you plan to use them to train other models.

## Citation

Please cite the Lit2Vec paper:

```bibtex
@misc{amiri2026lit2vec,
  title         = {Lit2Vec: A Reproducible Workflow for Building a Legally Screened Chemistry Corpus from S2ORC for Downstream Retrieval and Text Mining},
  author        = {Amiri, Mahmoud and Jafari, Jamile Mohammad and Mostafapour, Sara and Bocklitz, Thomas},
  year          = {2026},
  eprint        = {2604.12498},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2604.12498},
  url           = {https://arxiv.org/abs/2604.12498}
}
```

## Acknowledgements

The work was developed in the Bocklitz Lab at the Leibniz Institute of Photonic Technology (Leibniz IPHT) and Friedrich Schiller University Jena and was funded by the BMFTR within the programme Photonics Research Germany (13N15466, 13N15710, 13N15715). The abstracts come from S2ORC, provided by the Allen Institute for AI and the Semantic Scholar team. Questions: mahmoud.amiri@uni-jena.de.
