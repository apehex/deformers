import time

import torch
import torch.nn.utils
import tqdm

import mlable.models
import mlable.utils

import deformers.pipelines.monitor as _monitor
import deformers.pipelines.prefix.callbacks as _callbacks
import deformers.pipelines.prefix.processors as _processors

# GENERIC ######################################################################

def is_iterable(data: object) -> bool:
    try:
        data[0]
    except:
        return False
    return True

def is_text(data: object) -> bool:
    return (
        not isinstance(data, torch.Tensor)
        and (
            isinstance(data, str)
            or (
                is_iterable(data)
                and (
                    isinstance(data[0], str)
                    or (
                        is_iterable(data[0])
                        and isinstance(data[0][0], str))))))

# TRAINER ######################################################################

class PrefixTrainer:
    """Thin orchestration class for prefix training phases.

    The trainer owns:
    - mutable runtime state
    - teacher/student references
    - optimizer/scheduler/scaler handles
    - phase configuration
    - core training loop operations

    Stateless transforms such as vectorization and loss computation remain in
    `deformers.pipelines.prefix.processors`.
    """

    def __init__(
        self,
        text_tok: object,
        byte_tok: object,
        teacher_mod: object,
        student_mod: object,
        optimizer_obj: object,
        scheduler_obj: object,
        scaler_obj: object,
        context_obj: object,
        batch_cfg: dict,
        loss_cfg: dict,
        gradient_cfg: dict,
        training_cfg: dict,
        logging_cfg: dict,
        saving_cfg: dict=None,
        testing_cfg: dict=None,
        state_cfg: dict=None,
        callbacks_arr: list=None,
    ) -> None:
        # text <=> bytes utilities
        self._text_tok = text_tok
        self._byte_tok = byte_tok
        # targets of the training
        self._teacher = teacher_mod
        self._student = student_mod
        # training utilities
        self._optimizer = optimizer_obj
        self._scheduler = scheduler_obj
        self._scaler = scaler_obj
        self._context = context_obj
        # list of all the configuration parameters
        self._config = {
            'batch': dict(batch_cfg),
            'loss': dict(loss_cfg),
            'gradient': dict(gradient_cfg),
            'training': dict(training_cfg),
            'logging': dict(logging_cfg),
            'testing': dict(testing_cfg or {}),
            'saving': dict(saving_cfg or {}),}
        # secondary operations performed at the end of the step
        self._callbacks = callbacks_arr or []
        # initialize the state with the provided values and defaults
        self._state = self.init_state(dict(state_cfg or {}))

    # STATE ####################################################################

    def init_state(self, override: dict={}) -> dict[str, dict]:
        """Reset the nested runtime state."""
        return {
            'tensors': override.get('tensors', {}),
            'scalars': {
                **{
                    'switch/train': 1,
                    'switch/grad': 0,
                    'switch/log': 0,
                    'switch/save': 0,
                    'epoch/total': int(self._config['training'].get('epoch_num', 4)),
                    'epoch/current': 1,
                    'step/total': 0,
                    'step/global': 1,
                    'step/current': 1,
                    'iter/start': time.monotonic(),
                    'iter/time': 0.0,
                    'iter/tps': 0.0,
                    'gradient/rate': 0.0,
                    'gradient/norm': 0.0,
                    'loss/ema': 0.0,
                    'loss/total': 0.0,
                    'loss/mse/0': 0.0,
                    'loss/mse/k': 0.0,
                    'loss/cos/0': 0.0,
                    'loss/cos/k': 0.0,
                    'vocab/seen': 0.0,
                    'vocab/min': 0,
                    'vocab/max': 0,},
                **override.get('scalars', {}),},}

    # PHASE ####################################################################

    def run_phase(
        self,
        epoch_tot: int,
        dataset_obj: object,
        column_str: str,
    ) -> None:
        """Run a named phase for one or more epochs."""
        for __epoch in range(epoch_tot):
            self.run_epoch(
                epoch_num=__epoch,
                epoch_tot=epoch_tot,
                dataset_obj=dataset_obj,
                column_str=column_str)

    # EPOCH ####################################################################

    def init_epoch(
        self,
        epoch_num: int,
        epoch_tot: int,
        dataset_obj: object,
    ) -> object:
        """Start a new epoch."""
        __step_tot = len(dataset_obj)
        # track the iteration counters
        self._state['scalars']['epoch/total'] = int(epoch_tot)
        self._state['scalars']['epoch/current'] = int(epoch_num) + 1
        self._state['scalars']['step/total'] = int(__step_tot)
        # create a fresh iterator on the dataset
        return tqdm.tqdm(
            iter(dataset_obj),
            total=__step_tot,
            desc=f'epoch {epoch_num + 1}/{epoch_tot}',
            unit='batch',
            leave=True)

    def run_epoch(
        self,
        epoch_num: int,
        epoch_tot: int,
        dataset_obj: object,
        column_str: str,
    ) -> None:
        """Run one epoch over a dataset."""
        __pbar = self.init_epoch(
            epoch_num=epoch_num,
            epoch_tot=epoch_tot,
            dataset_obj=dataset_obj)
        # fresh iterator on the dataset
        for __step, __batch in enumerate(__pbar):
            # track the counters in the state
            self.init_step(step_num=__step)
            # vectorize => forward => loss => backward => update => callbacks => reset
            self.run_step(batch_arr=__batch, column_str=column_str)
            # reset the loss and free the memory
            self.close_step()
            # format and display the main stats
            self.step_progress(__pbar)
        # terminate the progress bar
        self.close_epoch(__pbar)

    def close_epoch(self, pbar_obj: object) -> None:
        """Terminate the temporary state of the epoch."""
        pbar_obj.close()

    # STEP #####################################################################

    def init_step(self, step_num: int) -> None:
        """Update counters and step switches."""
        __step_num = int(step_num) + 1
        __step_tot = self._state['scalars']['step/total']
        __epoch_num = max(0, self._state['scalars']['epoch/current'] - 1)
        # track the counters
        self._state['scalars']['step/global'] = __step_num + __epoch_num * __step_tot
        self._state['scalars']['step/current'] = __step_num
        # frequency of the main operations
        __test_every = int(self._config['testing'].get('every_num', 0))
        __log_every = int(self._config['logging'].get('every_num', 0))
        __save_every = int(self._config['saving'].get('every_num', 0))
        __grad_every = int(self._config['gradient'].get('every_num', 1))
        # tracks the operation that (will) run on the current step
        self._state['scalars']['switch/train'] = int((__test_every < 1) or ((__step_num % __test_every) != 0))
        self._state['scalars']['switch/log'] = int((__log_every > 0) and ((__step_num % __log_every) == 0))
        self._state['scalars']['switch/save'] = int((__save_every > 0) and ((__step_num % __save_every) == 0))
        self._state['scalars']['switch/grad'] = int((__step_num % __grad_every) == 0)

    def run_step(
        self,
        batch_arr: object,
        column_str: str,
    ) -> None:
        """Run one training step."""
        self.step_batch(batch_arr=batch_arr, column_str=column_str)
        self.step_forward()
        self.step_losses()
        self.step_backward()
        self.step_optimizer()
        self.step_callbacks()

    def close_step(self) -> None:
        """Reset the state after updating the weights."""
        if bool(self._state['scalars']['switch/grad']):
            # only reset after a weight update, because the mini batch losses accumulate accross steps
            self._state['tensors'] = {}
            self._state['scalars']['loss/total'] = 0.0
            self._state['scalars']['loss/mse/0'] = 0.0
            self._state['scalars']['loss/mse/k'] = 0.0
            self._state['scalars']['loss/cos/0'] = 0.0
            self._state['scalars']['loss/cos/k'] = 0.0
            # garbage collection
            mlable.models.free_memory()

    # VECTORIZE ################################################################

    def step_batch(self, batch_arr: object, column_str: str) -> None:
        """Vectorize a raw batch into mask, token ids, and byte patches."""
        __pad = self._config['batch'].get('padding_str', '')
        # common args
        __args = {
            'text_tok': self._text_tok,
            'byte_tok': self._byte_tok,
            'dtype_obj': self._config['training'].get('dtype', torch.long),
            'device_str': self._config['training'].get('device', 'cpu'),
            'sequence_dim': int(self._config['batch']['sequence_dim']),
            'patch_dim': int(self._config['batch']['patch_dim']),
            'left_pad': bool(self._config['batch'].get('left_pad', True)),}
        # check the content of the batch
        if 'indice' in column_str:
            __tensors = _processors.vectorize_indices(indices_arr=batch_arr[column_str], padding_str=__pad, **__args)
        # already byte encoded
        else:
            __tensors = _processors.vectorize_strings(text_arr=batch_arr[column_str], **__args)
        # save the tensors
        self._state['tensors']['inputs/mask'] = __tensors[0]
        self._state['tensors']['inputs/indices'] = __tensors[1]
        self._state['tensors']['inputs/bytes'] = __tensors[2]

    # FORWARD ##################################################################

    def step_forward(self) -> None:
        """Run teacher and student forwards for the current batch."""
        __hidden = (
            (float(self._config['loss'].get('mse_k_rate', 0.0)) > 0.0)
            or (float(self._config['loss'].get('cos_k_rate', 0.0)) > 0.0))
        # mixed precision context
        with self._context:
            with torch.no_grad():
                # teacher forward: get original embeddings and hidden states (no grad)
                self._state['tensors']['outputs/teacher/0'] = _processors.embed(
                    indices_arr=self._state['tensors']['inputs/indices'],
                    model_obj=self._teacher)
                # do not compute the hidden activations by default
                self._state['tensors']['outputs/teacher/k'] = torch.zeros(
                    tuple(self._state['tensors']['outputs/teacher/0'].shape),
                    dtype=self._state['tensors']['outputs/teacher/0'].dtype,
                    device=self._state['tensors']['outputs/teacher/0'].device)
                # compute the hidden activations only if they are used in the loss
                if __hidden:
                    self._state['tensors']['outputs/teacher/k'] = _processors.forward(
                        embeds_arr=self._state['tensors']['outputs/teacher/0'],
                        mask_arr=self._state['tensors']['inputs/mask'],
                        model_obj=self._teacher)
            # student forward: prefix -> inputs_embeds -> trunk -> hidden_k
            self._state['tensors']['outputs/student/0'] = self._student(
                self._state['tensors']['inputs/bytes']
            ).to(dtype=self._state['tensors']['outputs/teacher/0'].dtype)
            # do not compute the hidden activations by default
            self._state['tensors']['outputs/student/k'] = torch.zeros(
                tuple(self._state['tensors']['outputs/student/0'].shape),
                dtype=self._state['tensors']['outputs/student/0'].dtype,
                device=self._state['tensors']['outputs/student/0'].device)
            # compute the hidden activations only if they are used in the loss
            if __hidden:
                self._state['tensors']['outputs/student/k'] = _processors.forward(
                    embeds_arr=self._state['tensors']['outputs/student/0'],
                    mask_arr=self._state['tensors']['inputs/mask'],
                    model_obj=self._teacher)

    # LOSS #####################################################################

    def step_losses(self) -> None:
        """Compute tensor loss and detached scalar metrics for the current batch."""
        __outputs = _processors.compute_losses(
            mask_arr=self._state['tensors']['inputs/mask'],
            student_0_arr=self._state['tensors']['outputs/student/0'],
            student_k_arr=self._state['tensors']['outputs/student/k'],
            teacher_0_arr=self._state['tensors']['outputs/teacher/0'],
            teacher_k_arr=self._state['tensors']['outputs/teacher/k'],
            step_num=int(self._config['gradient'].get('every_num', 1)),
            mse_0_rate=float(self._config['loss'].get('mse_0_rate', 1.0)),
            mse_k_rate=float(self._config['loss'].get('mse_k_rate', 0.0)),
            cos_0_rate=float(self._config['loss'].get('cos_0_rate', 1.0)),
            cos_k_rate=float(self._config['loss'].get('cos_k_rate', 0.0)),
            relative_opt=bool(self._config['loss'].get('relative_opt', True)))
        # the tensor loss is needed for the backward computation
        self._state['tensors']['loss/total'] = __outputs[-1]
        # track the loss components
        self._state['scalars']['loss/mse/0'] += float(__outputs[0].item())
        self._state['scalars']['loss/mse/k'] += float(__outputs[1].item())
        self._state['scalars']['loss/cos/0'] += float(__outputs[2].item())
        self._state['scalars']['loss/cos/k'] += float(__outputs[3].item())
        self._state['scalars']['loss/total'] += float(__outputs[4].item())

    # BACKWARD #################################################################

    def step_backward(self) -> None:
        """Accumulate gradients from the current tensor loss."""
        __loss = self._state['tensors'].get('loss/total', None)
        # expect a tensor, not a scalar
        assert hasattr(__loss, 'shape'), 'Missing `tensors["loss/total"]` before step_backward().'
        # undo the float scaling before computing the backward pass
        self._scaler.scale(__loss).backward()

    # UPDATE ###################################################################

    def step_optimizer(self) -> None:
        """Apply optimizer, scaler, scheduler, and gradient clipping on accumulation boundary."""
        __norm_max = float(self._config['gradient'].get('max_norm', 1.0))
        # only work every few steps, after accumulating the loss on a few batches
        if bool(self._state['scalars']['switch/grad']):
            # gradient clipping; unscale first to get true grad norm
            self._scaler.unscale_(self._optimizer)
            self._state['scalars']['gradient/rate'] = _monitor.current_lr(self._optimizer)
            self._state['scalars']['gradient/norm'] = float(torch.nn.utils.clip_grad_norm_(
                self._student.parameters(),
                max_norm=__norm_max).item())
            # udpate the weights
            self._scaler.step(self._optimizer)
            self._scaler.update()
            # update the learning rate
            if self._scheduler is not None:
                self._scheduler.step()
            # reset the gradients
            self._optimizer.zero_grad()

    # CALLBACKS ################################################################

    def step_callbacks(self) -> None:
        """Run all triggered callbacks on the current state."""
        for __callback in self._callbacks:
            if _callbacks.is_callback(__callback):
                if __callback['trigger'](self._state['scalars']):
                    __callback['operation'](self._state['scalars'])

    def cleanup_callbacks(self) -> None:
        """Run cleanup on all registered callbacks."""
        for __callback in self._callbacks:
            if _callbacks.is_callback(__callback):
                __callback['cleanup']()

    # PROGRESS #################################################################

    def step_progress(self, pbar_obj: object) -> None:
        # aggregate and format
        __stats = _callbacks.format_state(state=self._state['scalars'])
        # filter the epoch and step since they are already in the pbar
        pbar_obj.set_postfix({__k: __v for (__k, __v) in __stats.items() if (__k not in ['epoch', 'step'])})
