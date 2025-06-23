#!/bin/bash

# Here run experiment:
#   - Gold document with documents from search results
#   - load the full corpus, no subset
#   - use specific search results file (ex. bm25 low/mid/high score noise)
#   - hence ignore use random or use adore

# llm_id microsoft/phi-2
# llm_id tiiuae/falcon-7b-instruct
# llm_id mosaicml/mpt-7b-instruct
# llm_id meta-llama/Llama-2-7b-chat-hf

CUDA_VISIBLE_DEVICES=0 python src/generate_answers_llm.py \
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

