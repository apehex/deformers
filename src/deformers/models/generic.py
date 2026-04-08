import copy
import gc

import torch
import torch.cuda
import torch.nn
import transformers

# FREE #########################################################################

def free_memory(
    model: object=None
) -> None:
    # drop references
    if model is not None:
        del model
    # run garbage collection
    gc.collect()
    # free CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# PREFIX #######################################################################

def copy_prefix(
    parent_obj: object,
    layer_num: int
) -> object:
    # init from the config
    __child = parent_obj.__class__(parent_obj.config)
    # share decoder core
    __child.model.embed_tokens = parent_obj.model.embed_tokens
    __child.model.norm = parent_obj.model.norm
    __child.model.rotary_emb = parent_obj.model.rotary_emb
    # share prefix layers (same objects)
    __child.model.layers = torch.nn.ModuleList(parent_obj.model.layers[:layer_num])
    # keep LM head
    __child.lm_head = parent_obj.lm_head
    # config hygiene
    __child.model.config.num_hidden_layers = layer_num
    __child.config.num_hidden_layers = layer_num
    # layer types
    if getattr(__child.config, "layer_types", None) is not None:
        __child.config.layer_types = __child.config.layer_types[:layer_num]
        __child.model.config.layer_types = __child.config.layer_types
    # wrapper with the first N hidden layers, pointing at the parent weights
    return __child

def truncate_model(
    model_obj: object,
    layer_num: int,
) -> object:
    __model = model_obj.model
    # keep the first k layers only
    __model.layers = torch.nn.ModuleList(list(__model.layers[:layer_num]))
    # keep the config consistent
    __model.config.num_hidden_layers = layer_num
    model_obj.config.num_hidden_layers = layer_num
    # Keep layer_types consistent if present
    if getattr(__model.config, "layer_types", None) is not None:
        __model.config.layer_types = __model.config.layer_types[:layer_num]
        model_obj.config.layer_types = __model.config.layer_types
    # return the wrapper, with the head
    return model_obj

def truncate_config(
    config_dict: dict[str, object],
    layer_num: int,
) -> dict[str, object]:
    # copy the original config
    __config = dict(config_dict)
    # sanitize the layer count
    __config['num_hidden_layers'] = min(
        __config.get('num_hidden_layers', max(1, int(layer_num))),
        max(1, int(layer_num)))
    # keep only the relevant layer types
    if 'layer_types' in __config.keys():
        __config['layer_types'] = list(__config['layer_types'][:layer_num])
    # base dict[str, object]
    return __config
