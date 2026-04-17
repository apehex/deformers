import copy
import gc

import torch
import torch.cuda
import torch.nn
import transformers

# PREFIX #######################################################################

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
    config_obj: object,
    layer_num: int,
    target_key: str='__dummy__',
) -> object:
    __config = copy.deepcopy(config_obj)
    # get the sub-config or default to the full config
    __target = getattr(__config, target_key, __config)
    # sanitize the layer count
    __target.num_hidden_layers = min(
        int(__target.num_hidden_layers),
        max(1, int(layer_num)))
    # keep only the relevant layer types
    if hasattr(__target, 'layer_types'):
        __target.layer_types = list(__target.layer_types[:__target.num_hidden_layers])
    # return the full configuration (not just the targeted sub-config)
    return __config
