import os
import re
import json
import pickle
import argparse
import gc
import random
from typing import List, Dict

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from find_in_normalized import safe_char_to_token, OffsetNormalizer
from utils import str2bool, read_json
from normalize_answers import *
from llm import LLM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_generated_string(entry: dict):
    return entry['prompt'] + " " + entry['generated_answer']


def clear_cache():
    if 'outputs' in locals():
        del outputs
    gc.collect()
    torch.cuda.empty_cache()


def init_stats(entry: dict, gold_doc_position: int):
    entry_stats = {'gold_doc_position': gold_doc_position, 'gold_document_idx': entry['gold_document_idx']}
    return entry_stats


def get_avg_answers_number(entries: List[dict]):
    avg_len = 0
    more_than_1 = 0
    for i in range(len(entries)):
        avg_len += len(entries[i]['answers'])
        if len(entries[i]['answers']) > 5:
            # print(i)
            more_than_1 += 1
    print(avg_len / len(entries))
    print(more_than_1)


def pretty_print_stats(entry_stats: dict,
                       skip_keys=['answer_attention_to_gold_tokens', 'gt_answer_attention_to_gold_tokens']):
    print("{")
    for key, value in entry_stats.items():
        if skip_keys and key in skip_keys:
            continue
        if isinstance(value, list):
            print(f"  {key} (size: {len(value)}): {value}")
        else:
            print(f"  {key}: {value}")
    print("}")


#  should be the same tokenizer as used in the model
def tokenize(tokenizer, generated_string: str):
    output_tokenized = tokenizer(
        generated_string,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    return output_tokenized


###################################### Finding Positions ######################################
def fill_position_stats(entry: dict, entry_stats: dict, output_tokenized) -> dict:
    generated_string = get_generated_string(entry)
    output_token_ids = output_tokenized['input_ids'][0]

    def find_documents_positions(generated_string: str):
        # Matching only documents in the context (up until string 'Question:')
        pattern = r"Document\s*\[\d+\].*?(?=Document\s*\[\d+\]|Question:)"
        matches = re.finditer(pattern, generated_string, re.DOTALL)
        documents_positions = [(match.start(), match.end()) for match in matches]
        assert len(entry['document_indices']) == len(
            documents_positions), "fill_position_stats: Document's number doesn't match"
        return [(output_tokenized.char_to_token(i), output_tokenized.char_to_token(j)) for i, j in documents_positions]

    def find_question_position(generated_string: str):
        pattern = r"(Question:\s*.+?)(?=Answer:)"
        matches = re.finditer(pattern, generated_string, re.DOTALL)
        question_positions = [(match.start(), match.end()) for match in matches]
        assert len(question_positions) == 1, "fill_position_stats: More than one question found"
        question_position = question_positions[0]
        return output_tokenized.char_to_token(question_position[0]), output_tokenized.char_to_token(
            question_position[1])

    def find_answer_position(generated_string: str):
        # Get answer up until the newline '\n'
        pattern = r"^\s*Answer:\s*.*$"
        pattern = r"(Answer:(?!.*Answer:).*)"
        matches = re.finditer(pattern, generated_string, re.DOTALL)

        answer_positions = [(match.start(), min(match.end(), len(generated_string))) for match in matches]
        assert len(answer_positions) == 1, "fill_position_stats: More than one answer found"
        answer_position = answer_positions[0]
        assert answer_position[1] == len(
            generated_string), "fill_position_stats: Answer doesn't span the last generated token"
        # spans till the last token (not including)
        return output_tokenized.char_to_token(answer_position[0]), len(output_token_ids)

    # mostly for debug
    # def token_positions_to_str(token_start, token_end, prefix):
    #     print(token_start, token_end)
    #     token_ids = output_token_ids[token_start:token_end]
    #     print(f"{prefix} token IDs:", token_ids)
    #     token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    #     print(f"{prefix} token strings:", token_strings)
    #

    documents_token_positions = find_documents_positions(generated_string)
    for doc_start, doc_end in documents_token_positions:
        assert doc_start is not None, "fill_position_stats: Document start not found"
        assert doc_end is not None, "fill_position_stats: Document end not found"

    question_start, question_end = find_question_position(generated_string)
    assert question_start is not None, "fill_position_stats: Question start not found"
    assert question_end is not None, "fill_position_stats: Question end not found"

    answer_start, answer_end = find_answer_position(generated_string)
    assert answer_start is not None, "fill_position_stats: Answer start not found"
    assert answer_end is not None, "fill_position_stats: Answer end not found"

    # NB: all positions excluding the end token
    entry_stats['documents_token_positions'] = documents_token_positions
    entry_stats['question_token_positions'] = (question_start, question_end)
    entry_stats['answer_token_positions'] = (answer_start, answer_end)

    return entry_stats


def fill_normalised_GT_positions(entry: dict, entry_stats: dict, output_tokenized, offset_normalizer) -> dict:
    output_token_ids = output_tokenized['input_ids'][0]
    generated_string = get_generated_string(entry)
    gt_answers_text_positions = offset_normalizer.find_in_normalized(generated_string, entry['answers'])
    assert len(entry['answers']) == len(
        gt_answers_text_positions), "fill_normalised_GT_positions: GT answer's number doesn't match"
    # NB: it's positions in the generated text (i.e. char-wise, not token-wise)
    entry_stats['gt_answers_text_positions'] = gt_answers_text_positions

    gt_answers_token_positions = []
    gt_answers_in_generated_token_positions = []
    gt_answers_in_gold_token_positions = []
    gold_doc_position = entry_stats['gold_doc_position']
    at_least_one_gt_in_gold = False
    # answer_positions corresponds to 1 GT answer
    for answer_positions in gt_answers_text_positions:
        # need to use safe in case it's the last token of the generated answer
        answer_token_positions = [(safe_char_to_token(output_tokenized, start, len(generated_string))[0],
                                   safe_char_to_token(output_tokenized, end, len(generated_string), move_right=False)[
                                       0]) for start, end in answer_positions]
        gt_answers_token_positions.append(answer_token_positions)

        answer_in_generated_token_positions = []
        answer_in_gold_token_positions = []
        for token_start, token_end in answer_token_positions:
            if token_start >= entry_stats['answer_token_positions'][0] and token_end <= \
                    entry_stats['answer_token_positions'][1]:
                answer_in_generated_token_positions.append((token_start, token_end))
            if token_start >= entry_stats['documents_token_positions'][gold_doc_position][0] and token_end <= \
                    entry_stats['documents_token_positions'][gold_doc_position][1]:
                answer_in_gold_token_positions.append((token_start, token_end))
                at_least_one_gt_in_gold = True

        gt_answers_in_generated_token_positions.append(answer_in_generated_token_positions)
        gt_answers_in_gold_token_positions.append(answer_in_gold_token_positions)

    # assert at_least_one_gt_in_gold, 'Gold document doesn\'t contain any GT answer'
    # all entries of the gt answers inside the whole input + output
    entry_stats['gt_answers_token_positions'] = gt_answers_token_positions
    # all entries of the gt answers inside the generated output
    entry_stats['gt_answers_in_generated_token_positions'] = gt_answers_in_generated_token_positions
    # all entries of the gt answers inside the gold document
    entry_stats['gt_answers_in_gold_token_positions'] = gt_answers_in_gold_token_positions
    entry_stats['at_least_one_gt_in_gold'] = at_least_one_gt_in_gold

    return entry_stats


######################################  Confidence scores & entropy ######################################

epsilon = 1e-8


def fill_confidence_scores(entry: dict, entry_stats: dict, entry_output_tokenized, entry_model_outputs) -> dict:
    # get logits for the next token position
    answer_start, answer_end = entry_stats['answer_token_positions']
    answer_tokens = entry_output_tokenized['input_ids'][0][answer_start:answer_end]

    # to get the logits that were used to predict a token at index i, we need logits[0, i-1, :]
    # [batch, tokens, vocabulary] -> [len(answer), vocab_size]
    next_token_logits = entry_model_outputs.logits[0, answer_start - 1:answer_end - 1,
                        :]  # Last position, all vocab (32000)

    # convert to probabilities
    # [len(answer), vocab_size]
    probs = F.softmax(next_token_logits, dim=-1)

    # get confidence for each actual token that was generated
    token_confidences = probs[torch.arange(len(answer_tokens)), answer_tokens]
    entry_stats['answer_tokens_output_confidences_softmax'] = token_confidences.tolist()
    assert len(entry_stats['answer_tokens_output_confidences_softmax']) == len(
        answer_tokens), "fill_confidence_scores: Number of tokens doesn't match (answer_tokens_output_confidences_softmax)"

    # calculate entropy for each position: -sum(p * log(p)) for all tokens
    entropies = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)  # [len(answer)]
    entry_stats['answer_tokens_output_entropy'] = entropies.tolist()
    assert len(entry_stats['answer_tokens_output_entropy']) == len(
        answer_tokens), "fill_confidence_scores: Number of tokens doesn't match (answer_tokens_output_entropy)"

    # calculate top5 mass for each position
    top5_masses = torch.sum(torch.topk(probs, 5, dim=-1).values, dim=-1)  # [len(answer)]
    entry_stats['answer_tokens_output_top5mass'] = top5_masses.tolist()
    assert len(entry_stats['answer_tokens_output_top5mass']) == len(
        answer_tokens), "fill_confidence_scores: Number of tokens doesn't match (answer_tokens_output_top5mass)"

    return entry_stats


###################################### Attention scores ######################################

def get_gt_answer_mask_for_gold_doc(entry: dict, entry_stats: dict, entry_output_tokenized, gt_answer_id: int):
    assert gt_answer_id < len(entry['answers']), "GT answer ID is out of range"

    gold_doc_position = entry_stats['gold_doc_position']
    output_token_ids = entry_output_tokenized['input_ids'][0]

    mask = np.zeros(len(output_token_ids), dtype=int)
    for start, end in entry_stats['gt_answers_in_gold_token_positions'][gt_answer_id]:
        mask[start:end] = 1

    return mask[entry_stats['documents_token_positions'][gold_doc_position][0]:
                entry_stats['documents_token_positions'][gold_doc_position][1]]


def get_all_gt_answers_masks_for_gold_doc(entry: dict, entry_stats: dict, gt_answer_ids, entry_output_tokenized):
    gold_doc_position = entry_stats['gold_doc_position']

    gold_doc_length = entry_stats['documents_token_positions'][gold_doc_position][1] - \
                      entry_stats['documents_token_positions'][gold_doc_position][0]
    combined_mask = np.zeros(gold_doc_length, dtype=int)
    for gt_answer_idx in gt_answer_ids:
        mask = get_gt_answer_mask_for_gold_doc(entry, entry_stats, entry_output_tokenized, gt_answer_idx)
        assert len(mask) == len(
            combined_mask), f"get_all_gt_answers_masks_for_gold_doc: mask length {len(mask)} doesn't match combined_mask length {len(combined_mask)}"
        combined_mask = combined_mask | mask

    return combined_mask


def fill_averaged_attentions(entry: dict, entry_stats: dict, entry_output_tokenized, delete_attentions: bool = True):
    all_mask = get_all_gt_answers_masks_for_gold_doc(entry, entry_stats, range(len(entry['answers'])),
                                                     entry_output_tokenized)
    entry_stats['answer_attention_to_gold_tokens_avg_1'] = np.mean(
        entry_stats['answer_attention_to_gold_tokens'][:, all_mask == 1], axis=1).tolist()
    entry_stats['answer_attention_to_gold_tokens_avg_0'] = np.mean(
        entry_stats['answer_attention_to_gold_tokens'][:, all_mask == 0], axis=1).tolist()

    if entry_stats['gt_answer_attention_to_gold_tokens'][0] is not None:
        entry_stats['gt_answer_attention_to_gold_tokens_avg_1'] = np.mean(
            entry_stats['gt_answer_attention_to_gold_tokens'][:, all_mask == 1], axis=1).tolist()
        entry_stats['gt_answer_attention_to_gold_tokens_avg_0'] = np.mean(
            entry_stats['gt_answer_attention_to_gold_tokens'][:, all_mask == 0], axis=1).tolist()

        generated_gt_answer_ids = [i for i, gt_answer_positions in
                                   enumerate(entry_stats['gt_answers_in_generated_token_positions']) if
                                   gt_answer_positions]
        generated_gt_answer_mask = get_all_gt_answers_masks_for_gold_doc(entry, entry_stats, generated_gt_answer_ids,
                                                                         entry_output_tokenized)

        entry_stats['answer_attention_to_gold_tokens_avg_1_generated_gt'] = np.mean(
            entry_stats['answer_attention_to_gold_tokens'][:, generated_gt_answer_mask == 1], axis=1).tolist()
        entry_stats['answer_attention_to_gold_tokens_avg_0_generated_gt'] = np.mean(
            entry_stats['answer_attention_to_gold_tokens'][:, generated_gt_answer_mask == 0], axis=1).tolist()
        entry_stats['gt_answer_attention_to_gold_tokens_avg_1_generated_gt'] = np.mean(
            entry_stats['gt_answer_attention_to_gold_tokens'][:, generated_gt_answer_mask == 1], axis=1).tolist()
        entry_stats['gt_answer_attention_to_gold_tokens_avg_0_generated_gt'] = np.mean(
            entry_stats['gt_answer_attention_to_gold_tokens'][:, generated_gt_answer_mask == 0], axis=1).tolist()

    if delete_attentions:
        del entry_stats['answer_attention_to_gold_tokens']
        del entry_stats['gt_answer_attention_to_gold_tokens']

    return entry_stats


def fill_attention_scores(entry: dict, entry_stats: dict, entry_output_tokenized, entry_model_outputs,
                          to_normalize: bool = False) -> dict:
    gold_doc_position = entry_stats['gold_doc_position']

    answer_attention_to_gold_tokens = []
    gt_answer_attention_to_gold_tokens = []

    def get_gold_doc_attention(attentions):
        # Select answers tokens only: [ans_len, seq_len]
        # Average answers tokens: [seq_len]
        mean_attentions = attentions.mean(0)
        # 'answer_attention' contains the attention scores from the answer
        # to all other tokens in the sequence
        # add .tolist() if needed
        gold_doc_attentions = mean_attentions[entry_stats['documents_token_positions'][gold_doc_position][0]:
                                              entry_stats['documents_token_positions'][gold_doc_position][1]]
        if to_normalize:
            gold_doc_attentions = gold_doc_attentions / gold_doc_attentions.sum()
        return gold_doc_attentions

    for hidden_layer in entry_model_outputs.attentions:
        # [batch_size, num_attention_heads, seq_len, seq_len]
        attention_np = hidden_layer.float()
        # Average over the attention heads: [batch_size, seq_len, seq_len]
        attention_np = attention_np.mean(1)
        # Squeeze the first position since it is assumed batch_size == 1
        # [seq_len, seq_len]
        attention_np = attention_np.squeeze(0).detach().cpu().numpy()

        ########## if we want to only consider that part of the generated answer that contains a GT asnwer ########
        # We're summing from all the GT answers in the generated answer (assuming there could be several)
        # Collect attention vectors from all answer spans
        gt_answers_rows = []
        # #  all_stats['gt_answers_in_generated_token_positions'] stores all positionos of the all GT answer in generated answers
        for gt_answer_positions in entry_stats['gt_answers_in_generated_token_positions']:
            for start, end in gt_answer_positions:
                gt_answers_rows.append(attention_np[start:end])  # [span_len, seq_len]

        if gt_answers_rows:
            assert entry['ans_match_after_norm'], "fill_attention_scores: GT answer should be in the generated answer "
            # # Stack all answer token rows and average: [seq_len]
            gt_answers_attention = np.concatenate(gt_answers_rows, axis=0)
            gt_answer_attention_to_gold_tokens.append(get_gold_doc_attention(gt_answers_attention))
        else:
            assert not entry[
                'ans_match_after_norm'], "fill_attention_scores: GT answer should not be in the generated answer "
            gt_answer_attention_to_gold_tokens.append(None)

        ########## if we want to take the full generated answer (in case there was no GT answer in the output) ####
        # Select answers tokens only: [ans_len, seq_len]
        answer_attention = attention_np[
                           entry_stats['answer_token_positions'][0]:entry_stats['answer_token_positions'][1]]
        answer_attention_to_gold_tokens.append(get_gold_doc_attention(answer_attention))

    assert len(
        answer_attention_to_gold_tokens) == 32, "fill_attention_scores: Number of attention layers doesn't match in answer_attention_to_gold_tokens"
    assert len(answer_attention_to_gold_tokens[0]) == entry_stats['documents_token_positions'][gold_doc_position][1] - \
           entry_stats['documents_token_positions'][gold_doc_position][
               0], "fill_attention_scores: Number of tokens in gold document doesn't match in answer_attention_to_gold_tokens"

    assert len(
        gt_answer_attention_to_gold_tokens) == 32, "fill_attention_scores: Number of attention layers doesn't match in gt_answer_attention_to_gold_tokens"

    # for every layer separately
    entry_stats['answer_attention_to_gold_tokens'] = np.array(answer_attention_to_gold_tokens)
    entry_stats['gt_answer_attention_to_gold_tokens'] = np.array(gt_answer_attention_to_gold_tokens)

    return entry_stats


######################################  Internal Activations & Variance ######################################

def fill_internal_activations(entry: dict, entry_stats: dict, entry_output_tokenized, entry_model_outputs,
                              to_normalize: bool = False) -> dict:
    activations_variance_accross_tokens = []
    activations_variance_accross_hidden_dims = []
    activation_magnitudes = []
    for hidden_layer in entry_model_outputs.hidden_states:
        # hidden_layer: [batch_size, seq_len, hidden_dim]
        activation_np = hidden_layer.float()
        # answer_activations shape: [n_generated_tokens, hidden_dim]
        answer_activations = activation_np[0,
                             entry_stats['answer_token_positions'][0]:entry_stats['answer_token_positions'][1],
                             :].cpu().numpy()  # Take activation for generated tokens

        # for each hidden dimension, how much does it vary across generated tokens? (aka Neural consistency during generation)
        token_variance_per_dim = np.var(answer_activations, axis=0)  # shape: [hidden_dim]
        # average this variance across all hidden dimensions
        activations_variance_accross_tokens.append(float(np.mean(token_variance_per_dim)))

        # for each generated token, how much do the hidden dims vary? (aka Activation pattern diversity at each time step)
        dim_variance_per_token = np.var(answer_activations, axis=1)  # shape: [n_generated_tokens]
        # average across all generated tokens
        activations_variance_accross_hidden_dims.append(float(np.mean(dim_variance_per_token)))

        # for each generated token, calculate L2 norm across hidden dimensions
        token_magnitudes = np.linalg.norm(answer_activations, axis=1)  # shape: [n_generated_tokens]
        # average across all generated tokens
        activation_magnitudes.append(float(np.mean(token_magnitudes)))

    entry_stats['activations_variance_accross_tokens'] = activations_variance_accross_tokens
    entry_stats['activations_variance_accross_hidden_dims'] = activations_variance_accross_hidden_dims
    entry_stats['activation_magnitudes'] = activation_magnitudes

    return entry_stats


###################################### All stats together ######################################
def collect_all_stats_for_entry(entry: dict, gold_doc_position: int, tokenizer, llm, offset_normalizer) -> dict:
    entry_stats = init_stats(entry, gold_doc_position)

    entry_output_tokenized = tokenize(tokenizer=tokenizer,
                                      generated_string=get_generated_string(entry))  # tokenize once
    entry_stats = fill_position_stats(entry=entry, entry_stats=entry_stats, output_tokenized=entry_output_tokenized)
    entry_stats = fill_normalised_GT_positions(entry=entry, entry_stats=entry_stats,
                                               output_tokenized=entry_output_tokenized,
                                               offset_normalizer=offset_normalizer)

    with torch.no_grad():  # call the model once
        entry_model_outputs = llm.model(
            **entry_output_tokenized,
            output_attentions=True,
            output_hidden_states=True
        )

    entry_stats = fill_confidence_scores(entry=entry, entry_stats=entry_stats, entry_output_tokenized=entry_output_tokenized, entry_model_outputs=entry_model_outputs)
    entry_stats = fill_attention_scores(entry=entry, entry_stats=entry_stats, entry_output_tokenized=entry_output_tokenized, entry_model_outputs=entry_model_outputs)
    entry_stats = fill_averaged_attentions(entry=entry, entry_stats=entry_stats, entry_output_tokenized=entry_output_tokenized, delete_attentions=True)
    entry_stats = fill_internal_activations(entry=entry, entry_stats=entry_stats, entry_output_tokenized=entry_output_tokenized, entry_model_outputs=entry_model_outputs)

    del entry_model_outputs
    gc.collect()
    torch.cuda.empty_cache()

    return entry_stats

def collect_interpretability_stats(input_file: str, gold_doc_position: int, save_interval: int = 100):
    # auto-generate output filename
    name, ext = os.path.splitext(input_file)
    output_file = f"{name}_interpr_stats{ext}"

    with open(input_file, 'r') as f:
        data = json.load(f)

    results, start_idx = load_or_resume(output_file)
    errors = []

    llm_id = "meta-llama/Llama-2-7b-chat-hf"
    model_max_length = 4096
    llm = LLM(llm_id, device, quantization_bits=4, model_max_length=model_max_length)
    tokenizer = llm.tokenizer
    offset_normalizer = OffsetNormalizer()

    pbar = tqdm(range(start_idx, len(data)), desc="Collecting interpretability stats", initial=start_idx, total=len(data))

    for i in pbar:
        try:
            stats = collect_all_stats_for_entry(entry=data[i], gold_doc_position=gold_doc_position, tokenizer=tokenizer, llm=llm, offset_normalizer=offset_normalizer)
            results.append(stats)
        except Exception as e:
            error_result = {
                "index": i,
                "error": str(e),
            }
            results.append(error_result)
            errors.append((i, str(e)))

        if (i + 1) % save_interval == 0:
            save_json(results, output_file)
            pbar.set_postfix({"saved": i + 1, "errors": len(errors)})

    save_json(results, output_file)

    # Print error statistics
    print(f"\nProcessing complete!")
    print(f"Total entries: {len(data)}")
    print(f"Errors: {len(errors)}")

    if errors:
      print("Error details:")
      for idx, error in errors:
          print(f"  Index {idx}: {error}")


def load_or_resume(output_file: str):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} entries")
        return results, len(results)
    return [], 0


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser(description="Collect interpretability statistics")
    parser.add_argument("input_file", type=str,
                        help="Path to the input file (an extended .json)")
    parser.add_argument("gold_doc_position", type=int,
                        help="Position of the gold document")
    parser.add_argument("save_interval", type=int, default=100,
                        help="Interval for saving results")

    parser.add_argument("--seed", type=int, default=10,
                        help="Random seed (default: 10)")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    collect_interpretability_stats(args.input_file, args.gold_doc_position, args.save_interval)
