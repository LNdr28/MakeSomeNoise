import re
import string
from typing import List, Tuple

from tokenizers import normalizers, Regex
from tokenizers.normalizers import Sequence, Replace, Lowercase
from transformers import AutoTokenizer

from src.normalize_text import CONTROLS, HYPHENS, MINUSES, DOUBLE_QUOTES, SINGLE_QUOTES, APOSTROPHES, ACCENTS, SLASHES, \
    TILDES
from src.utils import read_json


# TODO:
#  figure out hot to filter UNK tokens correctly
#  test on all dataset entries that we can extract normalised answer at least once from the generated answer
#  handle errors for processing (eka not throwing exceptions but logging them somewhere)


# follows srs/normalize_text.py#normalize for compatibility
def get_normalizers():
    normalizers = []
    for control in CONTROLS:
        normalizers.append(Replace(control, ''))

    normalizers.append(Replace('\u000b',' '))
    normalizers.append(Replace('\u000c',' '))
    normalizers.append(Replace(u'\u0085',' '))

    for hyphen in HYPHENS | MINUSES:
        normalizers.append(Replace(hyphen, '-'))
    normalizers.append(Replace('\u00ad', ''))

    for double_quote in DOUBLE_QUOTES:
        normalizers.append(Replace(double_quote, '"'))   # \u0022

    for single_quote in (SINGLE_QUOTES | APOSTROPHES | ACCENTS):
        normalizers.append(Replace(single_quote, "'"))  # \u0027

    normalizers.append(Replace('′', "'"))       # \u2032 prime
    normalizers.append(Replace('‵', "'"))       # \u2035 reversed prime
    normalizers.append(Replace('″', "''"))      # \u2033 double prime
    normalizers.append(Replace('‶', "''"))      # \u2036 reversed double prime
    normalizers.append(Replace('‴', "'''"))     # \u2034 triple prime
    normalizers.append(Replace('‷', "'''"))     # \u2037 reversed triple prime
    normalizers.append(Replace('⁗', "''''"))    # \u2057 quadruple prime
    normalizers.append(Replace('…', '...'))     # \u2026
    normalizers.append(Replace(' . . . ', ' ... '))     # \u2026

    for slash in SLASHES:
        normalizers.append(Replace(slash, '/'))

    for tilde in TILDES:
        normalizers.append(Replace(tilde, '~'))

    return normalizers


# follows srs/normalize_answer.py#normalize_answer for compatibility
def get_answer_normalizers():
    normalizers = []
    normalizers.append(Lowercase())
    normalizers.extend(get_normalizers())

    for punct in string.punctuation:
        normalizers.append(Replace(punct, ' '))

    normalizers.append(Replace(Regex(r'\b(a|an|the)\b'), ' '))
    # removes leading / trailing whitespaces
    normalizers.append(Replace(Regex(r'^\s+|\s+$'), ''))
    # fixes extra whitespaces
    normalizers.append(Replace(Regex(r'\s+'), ' '))

    # remove non-printable characters (otw they are tokenised as UNK)
    # normalizers.append(Replace(Regex(r'[^\x20-\x7E\xA0-\xFF]'), ''))

    return normalizers



DEFAULT_NORMALIZER = Sequence([
    normalizers.NFC(),
    normalizers.NFKC()
    # ] + get_normalizers())
] + get_answer_normalizers())




class OffsetNormalizer:
    """
    Supports mapping indices in normalised text to indices of tokens in the original text to search between them
    """

    def __init__(self, tokenizer=None, normalizer=None):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", max_length=4096)
        if normalizer is None:
            normalizer = DEFAULT_NORMALIZER

        tokenizer.backend_tokenizer.normalizer = normalizer
        self.tokenizer = tokenizer

    def normalize(self, text):
        tokenized_text = self.tokenizer(text=text, return_offsets_mapping=True, return_tensors=None)
        text_indices = tokenized_text['input_ids']
        normalized_text = self.tokenizer.decode(text_indices, skip_special_tokens=True)
        return normalized_text

    def find_list_dif(self, text1, text2, token_ids1, token_ids2, offsets1, offsets2):
        differences = []
        max_length = max(len(token_ids1), len(token_ids2))
        print(f"Length dif: {len(token_ids1) - len(token_ids2)}")

        for i in range(max_length):
            # Get elements at position i (or None if list is too short)
            a = token_ids1[i] if i < len(token_ids1) else None
            b = token_ids2[i] if i < len(token_ids2) else None

            if a != b:
                differences.append((i, a, b))

                if a == self.tokenizer.unk_token_id:
                    start_char, end_char = offsets1[i]
                    problematic_text = text1[start_char:end_char]
                    print(f"{problematic_text} in first")

                if b == self.tokenizer.unk_token_id:
                    start_char, end_char = offsets2[i]
                    problematic_text = text2[start_char:end_char]
                    print(f"{problematic_text} in second")

                # Handle different cases for printing
                if a is None:
                    token_b = self.tokenizer.convert_ids_to_tokens(b)
                    print(f"{i}: missing in list1, list2 has ({b}): {token_b}")
                elif b is None:
                    token_a = self.tokenizer.convert_ids_to_tokens(a)
                    print(f"{i}: list1 has ({a}): {token_a}, missing in list2")
                else:
                    token_a = self.tokenizer.convert_ids_to_tokens(a)
                    token_b = self.tokenizer.convert_ids_to_tokens(b)
                    print(f"{i}: list1 ({a}): {token_a}, list2 ({b}): {token_b}")

        print(f"Total differences: {len(differences)}")
        return differences


    # def tokenize_no_unk(self, text):
    #     # Method 1: Test what causes UNK tokens
    #     tokenized_text = self.tokenizer(text=text, return_offsets_mapping=True, return_tensors=None)
    #     text_offsets = tokenized_text['offset_mapping']
    #     text_indices = tokenized_text['input_ids']
    #     unk_token_id = self.tokenizer.unk_token_id
    #
    #     if unk_token_id not in text_indices:
    #         return text  # no UNK tokens, text is fine
    #
    #     # otherwise we need to find what symbol is causing it
    #     cleaned_chars = []
    #     for char in text:
    #         # Test if this character alone would create UNK
    #         char_tokens = self.tokenizer(char, return_tensors=None)['input_ids']
    #         if unk_token_id in char_tokens:
    #             # Replace problematic character with space NOT changing the text length (important for mapping back)
    #             cleaned_chars.append(' ')
    #         else:
    #             cleaned_chars.append(char)
    #
    #
    #     return cleaned_text

    def find_in_normalized(self, text: str, substring: str) -> List[Tuple[int, int]]:
        # we now have the mapping text <-> tokens
        tokenized_text = self.tokenizer(text=text, return_offsets_mapping=True, return_tensors=None, add_special_tokens=False)
        text_offsets = tokenized_text['offset_mapping']
        text_indices = tokenized_text['input_ids']

        # getting normalized text and tokenizing it again to get a mapping normalized_text <-> tokens
        normalized_text = self.tokenizer.decode(text_indices, skip_special_tokens=True)
        tokenized_normalized_text = self.tokenizer(text=normalized_text, return_offsets_mapping=True, return_tensors=None, add_special_tokens=False)
        normalized_text_offsets = tokenized_normalized_text['offset_mapping']
        normalized_text_indices = tokenized_normalized_text['input_ids']

        # since the tokens are the same, we now have text <-> tokens <-> normalized_text
        # todo: move it to an error field to not throw exception while processing a big batch
        self.find_list_dif(text, normalized_text, text_indices, normalized_text_indices, text_offsets, normalized_text_offsets)
        assert text_indices == normalized_text_indices

        tokenized_substring = self.tokenizer(text=substring, return_offsets_mapping=True, return_tensors=None, add_special_tokens=False)
        normalized_substring = self.tokenizer.decode(tokenized_substring['input_ids'], skip_special_tokens=True)

        print(f"Normalized substring: {normalized_substring}")

        # build map from normalized char index to token index
        token_strings = [self.tokenizer.decode([idx], skip_special_tokens=True) for idx in text_indices]

        normalized_text_indices = [(m.start(), m.end()) for m in re.finditer(re.escape(normalized_substring), normalized_text)]
        orig_text_indices = []

        for substring_start, substring_end in normalized_text_indices:
            print(substring_start, substring_end, len(normalized_text))
            print("Found normalised substring: " + normalized_text[substring_start:substring_end])
            start_token_idx = tokenized_normalized_text.char_to_token(substring_start)
            end_token_idx = tokenized_normalized_text.char_to_token(substring_end)
            # could be an issue that substring_end points at a whitespace so there is actually no token at that place
            # in that case, it will return None
            # we then go left until there is some token (should work)
            while end_token_idx is None:
                substring_end -= 1
                end_token_idx = tokenized_normalized_text.char_to_token(substring_end)

            print(start_token_idx, end_token_idx)
            start_char, _ = text_offsets[start_token_idx]
            _, end_char = text_offsets[end_token_idx]
            orig_text_indices.append((start_char, end_char))
            print(f'Non-normalised it was: {text[start_char:end_char]}')

        return orig_text_indices




if __name__ == '__main__':
    idx = 3867
    example_idx = read_json('../data/csnlp/numdoc7_gold_at6_answerless_info_all_extended.json')[idx]
    generated_string = example_idx['prompt'] + " " + example_idx['generated_answer']
    gt_answer = example_idx['answers'][0]
    # print(generated_string)
    offset_normaliser = OffsetNormalizer()
    print(f'GT answer: {gt_answer}')
    indices = offset_normaliser.find_in_normalized(generated_string, gt_answer)
    # for s, e in indices:
    #     print(generated_string[s:e])


