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
Note that we already provide these files at [ETH Polybox](https://polybox.ethz.ch/index.php/s/Lr4NLQDP7BNbYQH) for easier replication. 
To analysis notebook is located in `src/noise_interpretability_analysis.ipynb`, and is shared together with cell outputs to showcase generated charts, that are also saved in the Polybox.

## Retrieving Noise


## Re-ranking Noise
