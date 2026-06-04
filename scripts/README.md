# scripts

The `scripts` directory contains a variety of files used for dataset construction, model training, inference, and analysis. This document briefly describes their purpose.

## Constructing the Dataset

* `corpus/privacy_sensitive_interactions.py` generates the raw dataset, including the underlying statements and associated metadata.
* `corpus/calculate_proportions.py` calculates the proportions of utterance-response pair types and retailer assignments on both a per-person and population-wide basis.
* `corpus/convert_to_sft.py` extracts utterance-response pairs from the raw dataset and partitions them into training and test splits.

## Fine-Tuning Models

### `llama*`

* `encoding/encoding.py` encodes the training data for the `llama*` models.
* `training/train_lora_csa.py` fine-tunes the `llama*` models and performs evaluation on the test set after each training epoch.

### `gpt-4.1`

By default, `corpus/convert_to_sft.py` produces a `.jsonl` file in the format required for fine-tuning `gpt-4.1` on OpenAI infrastructure. Since OpenAI's training infrastructure does not provide training callbacks equivalent to those available in `training/train_lora_csa.py`, evaluation is performed after training.

* `inference/convert_to_batch.py` converts a JSONL file into a text file in which each line is a prompt to be issued to a model.
* `inference/prompt_model.py` issues prompts to one or more models via a RELP-style interface or through batch processing.
* `inference/common.py` contains helper functions used by `prompt_model.py`.
* `inference/check_gift_cards.py` generates a model performance summary similar to that produced during local training.

## Analysis

* `analysis/model_roa.py` evaluates the ROA capabilities of fine-tuned models. Results from our ROA experiments can be found [here](../roa-tests/README.md).
