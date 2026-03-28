import gc

import datasets
import torch
import torch.cuda
import torch.nn
import transformers

import mlable.shapes
import deformers.tokenizers.byte

# META #########################################################################

DATA_CFG = {}

TOKEN_CFG = {
    'pretrained_model_name_or_path': 'qwen/qwen3.5-9b',
    'dtype': 'auto',
    'use_fast': True,}

MODEL_CFG = {
    'pretrained_model_name_or_path': 'qwen/qwen3.5-9b', # 'openai/gpt-oss-20b' 'qwen/qwen3.5-9b' 'qwen/qwen3.5-27b' 'google/gemma-3-27b-it'
    # 'attn_implementation': 'eager',
    'device_map': 'cuda' if torch.cuda.is_available() else 'cpu',}

TEXT_CFG = {
    'wiki': '''Lexical tokenization is conversion of a text into (semantically or syntactically) meaningful lexical tokens belonging to categories defined by a "lexer" program. In case of a natural language, those categories include nouns, verbs, adjectives, punctuations etc. In case of a programming language, the categories include identifiers, operators, grouping symbols, data types and language keywords. Lexical tokenization is related to the type of tokenization used in large language models (LLMs) but with two differences. First, lexical tokenization is usually based on a lexical grammar, whereas LLM tokenizers are usually probability-based. Second, LLM tokenizers perform a second step that converts the tokens into numerical values.''',}

# LOAD #########################################################################

BYTE_OBJ = deformers.tokenizers.byte.ByteTokenizer()
TOKEN_OBJ = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
MODEL_OBJ = transformers.AutoModelForCausalLM.from_pretrained(**MODEL_CFG)

# TOKENS #######################################################################

# batch of samples made of a single sentence
__texts = TEXT_CFG['wiki'].split('. ')
# list of couples of indices (start, end), for each sample
__offsets = TOKEN_OBJ(texts, return_offsets_mapping=True)['offset_mapping']
# list of token sub-strings, for each sample
__tokens = [[__t[__s:__e] for (__s, __e) in __o] for (__t, __o) in zip(texts, offsets)]
# fixed size patches of bytes, for each sample
__bytes = [BYTE_OBJ(__s, max_length=16, truncation=True, padding='max_length', padding_side='right')['input_ids'] for __s in tokens]

# RESET ########################################################################

def free_memory(model: torch.nn.modules.Module) -> None:
    # move to CPU first (optional, helps if GPU memory is fragmented)
    model.cpu()
    # drop references
    del model
    # run garbage collection
    gc.collect()
    # free CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
