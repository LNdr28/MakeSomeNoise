#!/bin/bash

# Experiment:
#   - No gold document, just use retrieved documents and random/distracting ones as "noise" if wanted.
#   - allows for custom random noise and retrieved documents apart form fixed defined ones in the script
#   - currently uses retrieval on test set (can be changed to train)

# llm_id microsoft/phi-2
# llm_id tiiuae/falcon-7b-instruct
# llm_id mosaicml/mpt-7b-instruct
# llm_id meta-llama/Llama-2-7b-chat-hf

CUDA_VISIBLE_DEVICES=0 python src/generate_answers_llm_mixed.py \
    --output_dir data/gen_res \
    --llm_id microsoft/phi-2 \
    --model_max_length 2048 \
    --load_full_corpus True \
    --custom_random_docs data/search_results/bm25_low_score_noise_10k_train_dataset_search_results_at10.pkl \
    --custom_retrieved_docs data/bm25_test_search_results_at250.pkl \
    --num_retrieved_documents 1 \
    --num_random_documents 2 \
    --put_retrieved_first False \
    --use_test True \
    --batch_size 16 \
    --save_every 250

