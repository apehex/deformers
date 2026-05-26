import os.path
import time

import torch
import torch.utils.tensorboard
import tqdm

import mlable.utils

import deformers.pipelines.monitor

# GENERIC ######################################################################

def noop(*args, **kwargs) -> None:
    pass

# CHECK ########################################################################

def is_callback(callback: dict) -> bool:
    return (
        all(__k in callback for __k in ['name', 'trigger', 'operation', 'cleanup'])
        and isinstance(callback['name'], str)
        and callable(callback['trigger'])
        and callable(callback['operation'])
        and callable(callback['cleanup']))

# FORMAT #######################################################################

def format_state(state: dict) -> dict:
    """Group and format the state variables to export them."""
    __switches = ' '.join(
        state['switch/test'] * ['test']
        + state['switch/grad'] * ['grad']
        + state['switch/log'] * ['log']
        + state['switch/save'] * ['save'])
    return {
        'switch': f"[{__switches}]",
        'epoch': f"({state['epoch/current']}/{state['epoch/total']})",
        'step': f"({state['step/current']}/{state['step/total']})",
        'loss': f"(ema: {state['loss/ema']:.6f} total: {state['loss/total']:.6f} mse(0: {state['loss/mse/0']:.6f} k: {state['loss/mse/k']:.6f}) cos(0: {state['loss/cos/0']:.6f} k: {state['loss/cos/k']:.6f}))",
        'metric': f"(kld/k: {state['metric/kld/k']:.6f} topk/k: {state['metric/topk/k']:.4f})",
        'gradient': f"(rate: {state['gradient/rate']:.2e} norm: {state['gradient/norm']:.4f})",
        'iter': f"(time: {state['iter/time'] * 1000.0:.0f} tok/s: {state['iter/tps']:.0f})",}

# SPEED ########################################################################

def prepare_speed_callback(
    every_num: int,
    batch_len: int,
) -> dict:
    # test whether the callback should be run
    def __trigger(state: dict) -> bool:
        return (state['scalars']['step/current'] % every_num) == 0
    # write the state to the target file
    def __operation(state: dict) -> None:
        __scalars = state['scalars']
        # time in seconds
        __scalars['iter/time'] = time.monotonic() - __scalars['iter/start']
        # tokens per second
        __scalars['iter/tps'] = deformers.pipelines.monitor.throughput(every_num * batch_len, __scalars['iter/time'])
        # reset the timer
        __scalars['iter/start'] = time.monotonic()
    # format as a callback
    return {
        'name': 'speed',
        'trigger': __trigger,
        'operation': __operation,
        'cleanup': noop,}

# LOSS EMA #####################################################################

def prepare_ema_callback(
    every_num: int,
    start_num: int,
    smooth_rate: float,
) -> dict:
    # test whether the callback should be run
    def __trigger(state: dict) -> bool:
        return (state['scalars']['step/current'] % every_num) == 0
    # write the state to the target file
    def __operation(state: dict) -> None:
        __scalars = state['scalars']
        # track the loss EMA, default to the current loss for the first few steps
        __scalars['loss/ema'] = mlable.utils.ema(
            average=float(__scalars['loss/ema']),
            current=float(__scalars['loss/total']),
            factor=smooth_rate * float(__scalars['step/current'] > start_num))
    # format as a callback
    return {
        'name': 'ema',
        'trigger': __trigger,
        'operation': __operation,
        'cleanup': noop,}

# LOGGING ######################################################################

def prepare_logging_callback(
    every_num: int,
    path_str: str,
) -> dict:
    # create the parent directory
    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    # open the file
    __file = open(path_str, 'w')
    # test whether the callback should be run
    def __trigger(state: dict) -> bool:
        return (state['scalars']['step/current'] % every_num) == 0
    # write the state to the target file
    def __operation(state: dict) -> None:
        # aggregate and format
        __stats = format_state(state=state['scalars'])
        # write as a single line
        __file.write(deformers.pipelines.monitor.serialize_state(state=__stats, prefix='') + '\n')
    # close the file on cleanup
    def __cleanup() -> None:
        __file.close()
    # format as a callback
    return {
        'name': 'log',
        'trigger': __trigger,
        'operation': __operation,
        'cleanup': __cleanup,}

def prepare_tensorboard_callback(
    every_num: int,
    path_str: str,
) -> dict:
    # create the parent directory
    os.makedirs(path_str, exist_ok=True)
    # open the file
    __writer = torch.utils.tensorboard.SummaryWriter(log_dir=path_str)
    # test whether the callback should be run
    def __trigger(state: dict) -> bool:
        return (state['scalars']['step/current'] % every_num) == 0
    # write the state to the target file
    def __operation(state: dict) -> None:
        __scalars = state['scalars']
        # write all the scalars
        deformers.pipelines.monitor.log_scalars(writer=__writer, step=__scalars['step/current'], scalars=__scalars)
    # close the file on cleanup
    def __cleanup() -> None:
        __writer.close()
    # format as a callback
    return {
        'name': 'log',
        'trigger': __trigger,
        'operation': __operation,
        'cleanup': __cleanup,}

# CHECKPOINT ###################################################################

def prepare_saving_callback(
    every_num: int,
    path_str: str,
    model_obj: object,
) -> dict:
    # create the parent directory
    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    # test whether the callback should be run
    def __trigger(state: dict) -> bool:
        return (state['scalars']['step/current'] % every_num) == 0
    # write the state to the target file
    def __operation(state: dict) -> None:
        # save the configuration and state of the model in a single file
        model_obj.save_checkpoint(path=path_str)
    # format as a callback
    return {
        'name': 'save',
        'trigger': __trigger,
        'operation': __operation,
        'cleanup': noop,}

# PROGRESS #####################################################################

def prepare_progress_callback(
    every_num: int,
    pbar_obj: object,
) -> dict:
    # test whether the callback should be run
    def __trigger(state: dict) -> bool:
        return (state['scalars']['step/current'] % every_num) == 0
    # write the state to the target file
    def __operation(state: dict) -> None:
        # aggregate and format
        __stats = format_state(state=state['scalars'])
        # filter the epoch and step since they are already in the pbar
        pbar_obj.set_postfix({__k: __v for (__k, __v) in __stats.items() if (__k not in ['epoch', 'step'])})
    # format as a callback
    return {
        'name': 'pbar',
        'trigger': __trigger,
        'operation': __operation,
        'cleanup': noop,}
