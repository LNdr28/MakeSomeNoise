#!/bin/bash

# This file serves to generate the results (accuracy) from llm answers generated with the 'run_generation_gold_doc_custom_search_res.sh'
# File. In the classic setting, save path is contriver no random. Remember that the results were generated with "noise" search results from bm25

python src/read_generation_results.py \
    --output_dir data/gen_res \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
    --use_test False \
    --prompt_type classic \
    --use_adore False \
    --gold_position 0 \
    --num_documents_in_context 2 \
    --get_documents_without_answer False \

