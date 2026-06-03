# dataset
In this folder, there are four files, the primary one being `psi523_dataset.jsonl`. This is the programmatically-seeded, GPT-expanded dataset, containing 10,000 data points. The naming behind this file is rooted in the ratio of utterance-response pairs: 5(0)% $(P_i, R^*_i)$, 2(0)% $(P_i, R_i)$, and 3(0)% $(P^*_i, R_i)$.

The remaining files are derived files:
- `psi523_train.jsonl`: contains all of the $(P_i, R^*_i)$ and $(P_i, R_i)$ utterance-response pairs, as well as all $(P^*_i, R_i)$ for 80% of the individuals in the dataset.
- `psi523_train_no_pi_gc.jsonl`: a subset of `psi523_train.jsonl`, containing only the $(P^*_i, R_i)$ pairs, the type that contains no private information in the utterance but distributes a gift card in the response.
- `psi523_test_no_pi_gc.jsonl`: contains the remaining $(P^*_i, R_i)$ pairs and is used for model evaluation in terms of ROA breakage.