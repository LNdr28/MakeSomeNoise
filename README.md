# 💥 Let’s Make Some Noise: Improving Noise Selection for Retrieval-Augmented Generation

This repository contains the code and data for ETH CSNLP 2025 project. 
This project is built upon the work of Cuconasu et al., ["The Power of Noise: Redefining Retrieval for RAG Systems"](http://dx.doi.org/10.1145/3626772.3657834).
## Installation

1. Set up a conda environment.

```
conda create -n make_some_noise python=3.9 --yes
conda activate make_some_noise
```

2. Install package and requirements.

```
pip install -r requirements.txt
```

## Data
To replicate the experiments for the CSNLP project, we provide pre-processed data via [ETH Polybox](https://polybox.ethz.ch/index.php/s/Lr4NLQDP7BNbYQH) (stored until 1 Aug 2025).


The data that was used by Cuconasu et al. and that we utilize for our experiments for compatibility can be accessed using the following instructions, provided by the authors of the original paper:
> The corpus and NQ datasets can be downloaded from HuggingFace using the code in the respective sections.
>
> The full training set was not used for the experiments; instead, a smaller sample of 10K entries was employed, and is available in the `data` folder of this repository. For the experiments described in the "RAG in Practice" section, the test set was utilized.
>
> Data not present in this repository or not downloadable from HuggingFace can be found in this [Google Drive](https://drive.google.com/drive/folders/1MfR7mJ76tyVpjbMwUkMVbEjOQKpdL-Lq?usp=sharing).


## Noise Interpetability Analysis

The original pipeline of Cuconasu et al. includes running `src/generate_answers_llm.py` to generate LLM answers, and then running `src/read_generation_results.py` on the result files for processing the generated answers.
We extend this pipeline to collect statistics for inerpretability analysis, by providing `src/add_interpretability_stats.py` that needs to be run on the result of the previous computation. 

To support token-wise attention analysis, we implement `OffsetNormalizer` (located in `src/find_in_normalized.py`), needed to search for ground truth answers in the documents.

It is challenging as searching without normalization only finds exact matches, missing valid instances that differ in whitespace, punctuation, casing, etc.
However, searching with normalized text finds more matches but returns indices relative to the normalized version, not the original text.

`OffsetNormalizer` overcomes these challenges by performing text normalization while maintaining a mapping between the normalized and original text positions. This allows us to:
1. Find substring matches using normalized text 
2. Return the corresponding locations in the original, unnormalized text
3. Preserve the exact character positions needed to extract ground truth answers

The output files of `src/add_interpretability_stats.py` are then used for the interpretability analysis. 
Note that we already provide these files at [ETH Polybox](https://polybox.ethz.ch/index.php/s/Lr4NLQDP7BNbYQH) for easier replication of our experiments. Just put the files under `interpretability-anaysis-data` from Polybox to `data/csnlp` folder of this repository; this should allow for a smooth run of the analysis.
The analysis notebook is located in `src/noise_interpretability_analysis.ipynb`, and is shared together with cell outputs to showcase generated charts, that are also saved in the Polybox.

## Retrieving Noise

To replicate the noise retrieval experiments there are two steps (given that the subset of the NQ training set is already prepared):
1. For each document in the corpus, noise is retrieved and stored to a file. This is done for each noise separately (Random Noise, Low Score Noise, Mid Score Noise)
2. Given the "pre-calculated" noise documents from step 1, we can now run benchmarks.

### Commands for retrieving Noise

For the Random Noise setting, which is the baseline we use the same random noise as Cuconasu et al. The respective file is at `data/10k_random_results_at60.pkl`.
To retrieve the noise for Low Score and Mid Score setting we implemented the script `src/generate_search_results_bm25.py`. In the following, we provide the commands used to generate the noise:

Generate Low Score Noise (10 noise documents for each query)
```bash
python src/generate_search_results_bm25.py --op_mode=low_score_noise
```

Generate Mid Score Noise (10 noise documents for each query)
```bash
python src/generate_search_results_bm25.py --op_mode=mid_score_noise
```

### Commands to run benchmarks

Given the three files containing the noise documents for each sample in the train dataset, one can now run the Benchmarks with the help of the following script: `example_scripts/run_generation_gold_doc_custom_search_res.sh`
It contains the following command that can be modified to replicate the settings listed in our report:

```bash
python src/generate_answers_llm.py \
    --output_dir data/gen_res \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
    --model_max_length 4096 \
    --load_full_corpus True \
    --search_results data/search_results/bm25_mid_score_noise_10k_train_dataset_search_results_at10.pkl \
    --gold_position 0 \
    --num_documents_in_context 2 \
    --get_documents_without_answer False \
    --batch_size 1 \
    --save_every 250
```

The interesting variables to change for the experiments are:
1. `gold_position`: the position of the gold document in the context. 
2. `num_documents_in_context`: to overall number of documents in the context (noise documents and gold document)
3. `llm_id`: huggingface tag of the LLm to use for the benchmark


## Re-ranking Noise
To train the Electra reranker, use the script `src/train_colbert_e2e.py`. The required bm_25 index is generated by running `src/generate_search_results_bm25.py`, independent of `--op_mode`. The noise type used for training can be set in `--noise_type`. Example use:

```bash
python src/train_colbert_e2e.py \
    --load_idx .../bm25_search_results_idx \
    --noise_type low_score_noise \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
    --gold_position 6 \
    --num_documents_in_context 7 \
    --batch_size 1 \
    --load_full_corpus True
```

The electra model checkpoint can then be used in the script `src/generate_answers_llm_colbert.py` by setting the argument `--checkpoint` to generate answers using reranked noise documents. Omit the argument to use the pretrained Electra reranker. The required bm_25 index is generated by running `src/generate_search_results_bm25.py`, independent of `--op_mode`. The noise type used for training can be set in `--noise_type`. The other arguments follow the default `src/generate_answers_llm.py` script. Example use:

```bash
python src/generate_answers_llm_colbert.py \
    --output_dir data/gen_res \
    --checkpoint colbert_low_noise/colbert_last/ \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
	--noise_type low_score_noise \
    --model_max_length 4096 \
    --gold_position 6 \
    --num_documents_in_context 7 \
    --save_every 250 \
	--load_idx .../bm25_search_results_idx/
```