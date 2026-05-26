"""The prefix runner owns:
- mutable runtime state
- teacher/student references (external)
- tokenizer references (external)
- training utility setup from per-setup configuration (optimizer/scaler/context/scheduler/callbacks)
- current configuration
- core training loop operations

Stateless transforms such as vectorization and loss computation remain in
`deformers.pipelines.prefix.processors`.

Typical lifecycle:
    runner = PrefixTrainer(text_tok, byte_tok, teacher, student)
    runner.setup_global(context_cfg=..., optimizer_cfg=..., scaler_cfg=...)
    runner.setup_phase(dataset, epoch_num, column, batch_cfg=..., loss_cfg=..., ...)
    runner.run_phase()                     # runs all epochs
    runner.close_callbacks()

    runner.setup_phase(dataset2, epoch_num2, column2, batch_cfg=..., loss_cfg=..., ...)
    runner.run_phase()
    runner.close_callbacks()
"""

import contextlib
import time

import torch
import torch.amp
import torch.nn.utils
import torch.optim
import tqdm

import mlable.models
import mlable.schedulers
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

class BaseRunner:
    """Shared orchestration class for prefix training / testing phases."""

    # INIT #####################################################################

    def __init__(
        self,
        text_tok: object,
        byte_tok: object,
        teacher_mod: object,
        student_mod: object,
    ) -> None:
        # text <=> bytes utilities (external)
        self._text_tok = text_tok
        self._byte_tok = byte_tok
        # targets of the training (external)
        self._teacher = teacher_mod
        self._student = student_mod
        # training utilities (populated by setup_* methods, not the constructor)
        self._optimizer = None
        self._scaler = None
        # current phase attributes (set by setup_phase)
        self._scheduler = None
        self._dataset = None
        # can be changed at any time
        self._callbacks = []
        # empty configuration
        self._config = self.init_config()
        # initialize the state
        self._state = self.init_state()

    def init_config(self) -> dict[str, dict]:
        return {
            'context': {}, # dtype and device
            'optimizer': {}, # global level
            'scaler': {}, # global level
            'phase': {}, # phase level
            'batch': {}, # phase level
            'loss': {}, # phase level
            'gradient': {}, # phase level
            'scheduler': {}, # phase level
            'testing': {}, # callback
            'ema': {}, # callback
            'speed': {}, # callback
            'logging': {}, # callback
            'tboard': {}, # callback
            'saving': {},} # callback

    def init_state(self, override: dict={}) -> dict[str, dict]:
        """Build the nested runtime state, optionally overriding `tensors` and `scalars`."""
        override = dict(override or {})
        return {
            'tensors': override.get('tensors', {}),
            'scalars': {
                **{
                    'switch/grad': 0,
                    'switch/test': 0,
                    'switch/log': 0,
                    'switch/save': 0,
                    'switch/progress': 0,
                    'switch/cleanup': 0,
                    'epoch/total': 1,
                    'epoch/current': 1,
                    'step/total': 0,
                    'step/global': 0,
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

    # CONFIG ###################################################################

    def _check_config(self, config_dict: object, required_keys: tuple[str]=()) -> bool:
        """Check whether an input configuration has the right format."""
        return isinstance(config_dict, dict) and all(__key in config_dict for __key in required_keys)

    def _check_setup(self) -> None:
        """Raise AssertionError if the runner is not ready to run a phase."""
        assert self._dataset is not None, 'dataset is None; call setup_phase() first.'
        self._check_runtime()

    def _check_runtime(self) -> None:
        """Hook for mode-specific setup checks."""
        pass

    # SWITCH ###################################################################

    def _trigger_test(self, step_num: int) -> bool:
        __every = int(self._config['testing'].get('every_num', 0))
        return (__every > 0) and ((int(step_num) % __every) == 0)

    def _trigger_log(self, step_num: int) -> bool:
        __every = int(self._config['logging'].get('every_num', 0))
        return (__every > 0) and ((int(step_num) % __every) == 0)

    def _trigger_save(self, step_num: int) -> bool:
        __every = int(self._config['saving'].get('every_num', 0))
        return (__every > 0) and ((int(step_num) % __every) == 0)

    def _trigger_update(self, step_num: int) -> bool:
        __every = int(self._config['gradient'].get('every_num', 1))
        return (int(step_num) % __every) == 0

    def _trigger_progress(self, step_num: int) -> bool:
        return self._trigger_update(step_num=step_num)

    def _trigger_cleanup(self, step_num: int) -> bool:
        return self._trigger_update(step_num=step_num)

    def _trigger_hidden(self, step_num: int) -> bool:
        return (
            self._trigger_test(step_num=step_num)
            or (float(self._config['loss'].get('mse_k_rate', 0.0)) > 0.0)
            or (float(self._config['loss'].get('cos_k_rate', 0.0)) > 0.0))

    # GLOBAL ###################################################################

    def _context(self, dtype_obj: object=None, gradient_opt: bool=False) -> object:
        __context = contextlib.ExitStack()
        __config = self._config.get('context', {})
        __dtype = __config.get('dtype', torch.float32) if (dtype_obj is None) else dtype_obj
        __device = __config.get('device', 'cpu')
        if __dtype != torch.float32:
            __context.enter_context(torch.amp.autocast(device_type=__device, dtype=__dtype))
        if not bool(gradient_opt):
            __context.enter_context(torch.no_grad())
        return __context

    def setup_context(self, context_cfg: dict={}) -> None:
        """Create the autocast context from training config."""
        if self._check_config(context_cfg, ('device', 'dtype')):
            # save the configuration for import / export
            self._config['context'] = context_cfg

    def setup_optimizer(self, optimizer_cfg: dict={}) -> None:
        """Create the AdamW optimizer from config."""
        if self._check_config(optimizer_cfg, ('lr',)):
            # save the configuration for import / export
            self._config['optimizer'] = optimizer_cfg
            # operate on the student model only
            self._optimizer = torch.optim.AdamW(self._student.parameters(), **optimizer_cfg)
            # make sure the gradients start at 0
            self._optimizer.zero_grad()

    def setup_scaler(self, scaler_cfg: dict={}) -> None:
        """Create the GradScaler from config."""
        if self._check_config(scaler_cfg, ('enabled',)):
            # save the configuration for import / export
            self._config['scaler'] = scaler_cfg
            # reduce gradient underflow with dtype float16
            self._scaler = torch.amp.GradScaler(**dict(scaler_cfg))

    def setup_global(
        self,
        context_cfg: dict={},
        optimizer_cfg: dict={},
        scaler_cfg: dict={},
        overwrite_opt: bool=False,
    ) -> None:
        """Initialize long-lived utilities: optimizer, scaler, and mixed-precision context."""
        # create the persistent objects
        if overwrite_opt or self._optimizer is None:
            self._optimizer = None
            self.setup_optimizer(optimizer_cfg=optimizer_cfg)
        if overwrite_opt or self._scaler is None:
            self._scaler = None
            self.setup_scaler(scaler_cfg=scaler_cfg)
        # no fixed object, the context is created on each invocation of `BaseRunner._context(...)`
        self.setup_context(context_cfg=context_cfg)

    # PHASE ####################################################################

    def setup_configs(
        self,
        phase_cfg: dict={},
        batch_cfg: dict={},
        loss_cfg: dict={},
        gradient_cfg: dict={},
    ) -> None:
        if self._check_config(phase_cfg, ('column_str', 'epoch_num')):
            self._config['phase'] = phase_cfg
            self._state['scalars']['epoch/total'] = self._config['phase']['epoch_num']
        if self._check_config(batch_cfg, ('batch_dim', 'sequence_dim', 'patch_dim')):
            self._config['batch'] = batch_cfg
        if self._check_config(loss_cfg, ('mse_0_rate', 'mse_k_rate', 'cos_0_rate', 'cos_k_rate')):
            self._config['loss'] = loss_cfg
        if self._check_config(gradient_cfg, ('every_num', 'max_norm')):
            self._config['gradient'] = gradient_cfg

    def setup_scheduler(self, scheduler_cfg: dict={}) -> None:
        """Create a WaveLR scheduler from config."""
        if self._check_config(scheduler_cfg, ('start_rate', 'end_rate', 'total_num', 'warmup_num')):
            # save the configuration for import / export
            self._config['scheduler'] = scheduler_cfg
            # linear warmup + cosine decay
            self._scheduler = mlable.schedulers.WaveLR(optimizer_obj=self._optimizer, **scheduler_cfg)

    def setup_callbacks(
        self,
        testing_cfg: dict={},
        ema_cfg: dict={},
        speed_cfg: dict={},
        logging_cfg: dict={},
        tboard_cfg: dict={},
        saving_cfg: dict={},
    ) -> None:
        """Build the phase callbacks from the provided configs."""
        self._callbacks = []
        # todo
        self._config['testing'] = testing_cfg
        # overwrite the list of callbacks, even when there are none
        if self._check_config(ema_cfg, ('every_num', 'start_num', 'smooth_rate')):
            self._config['ema'] = ema_cfg
            self._callbacks.append(_callbacks.prepare_ema_callback(**ema_cfg))
        if self._check_config(speed_cfg, ('every_num', 'batch_len')):
            self._config['speed'] = speed_cfg
            self._callbacks.append(_callbacks.prepare_speed_callback(**speed_cfg))
        if self._check_config(logging_cfg, ('every_num', 'path_str')):
            self._config['logging'] = logging_cfg
            self._callbacks.append(_callbacks.prepare_logging_callback(**logging_cfg))
        if self._check_config(tboard_cfg, ('every_num', 'path_str')):
            self._config['tboard'] = tboard_cfg
            self._callbacks.append(_callbacks.prepare_tensorboard_callback(**tboard_cfg))
        if self._check_config(saving_cfg, ('every_num', 'path_str')):
            self._config['saving'] = saving_cfg
            self._callbacks.append(_callbacks.prepare_saving_callback(model_obj=self._student, **saving_cfg))

    def setup_phase(
        self,
        dataset_obj: object,
        phase_cfg: dict={},
        batch_cfg: dict={},
        loss_cfg: dict={},
        gradient_cfg: dict={},
        scheduler_cfg: dict={},
        testing_cfg: dict={},
        ema_cfg: dict={},
        speed_cfg: dict={},
        logging_cfg: dict={},
        tboard_cfg: dict={},
        saving_cfg: dict={},
    ) -> None:
        """Configure a training phase: store config, dataset info, rebuild scheduler and callbacks."""
        self._dataset = dataset_obj
        # overwrite the configurations
        self.setup_configs(
            phase_cfg=phase_cfg,
            batch_cfg=batch_cfg,
            loss_cfg=loss_cfg,
            gradient_cfg=gradient_cfg)
        # always recreate phase-local utilities
        self._scheduler = None
        self.setup_scheduler(scheduler_cfg=scheduler_cfg)
        # could remain empty
        self._callbacks = []
        self.setup_callbacks(
            testing_cfg=testing_cfg,
            ema_cfg=ema_cfg,
            speed_cfg=speed_cfg,
            logging_cfg=logging_cfg,
            tboard_cfg=tboard_cfg,
            saving_cfg=saving_cfg)

    def run_phase(self) -> None:
        """Run all epochs of the current phase using config set by setup_phase()."""
        # make sure all the training utilities are there 
        self._check_setup()
        # the previous step implies that the configs are valid
        for __epoch in range(self._config['phase']['epoch_num']):
            self.run_epoch(
                epoch_num=__epoch,
                epoch_tot=self._config['phase']['epoch_num'],
                dataset_obj=self._dataset,
                column_str=self._config['phase']['column_str'])

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
        for __step_num, __batch in enumerate(__pbar, start=1):
            # track the counters in the state
            self.init_step(step_num=__step_num)
            # vectorize => forward => loss => backward => update => callbacks => reset
            self.run_step(step_num=__step_num, batch_arr=__batch, column_str=column_str)
            # format and display the main stats
            self.step_progress(step_num=__step_num, pbar_obj=__pbar)
            # reset the loss and free the memory
            self.close_step(step_num=__step_num)
        # terminate the progress bar
        self.close_epoch(__pbar)

    def close_epoch(self, pbar_obj: object) -> None:
        """Terminate the temporary state of the epoch."""
        pbar_obj.close()

    # STEP #####################################################################

    def init_step(self, step_num: int) -> None:
        """Update counters and step switches."""
        __step_num = int(step_num)
        # step/global is a monotonically increasing counter that persists across epochs and phases
        self._state['scalars']['step/global'] += 1
        self._state['scalars']['step/current'] = __step_num
        # tracks the operation that (will) run on the current step
        self._state['scalars']['switch/test'] = int(self._trigger_test(step_num=__step_num))
        self._state['scalars']['switch/log'] = int(self._trigger_log(step_num=__step_num))
        self._state['scalars']['switch/save'] = int(self._trigger_save(step_num=__step_num))
        self._state['scalars']['switch/grad'] = int(self._trigger_update(step_num=__step_num))
        self._state['scalars']['switch/progress'] = int(self._trigger_progress(step_num=__step_num))
        self._state['scalars']['switch/cleanup'] = int(self._trigger_cleanup(step_num=__step_num))

    def run_step(
        self,
        step_num: int,
        batch_arr: object,
        column_str: str,
    ) -> None:
        """Run one training/testing step."""
        self.step_batch(step_num=step_num, batch_arr=batch_arr, column_str=column_str)
        self.step_forward(step_num=step_num)
        self.step_objective(step_num=step_num)
        self.step_callbacks(step_num=step_num)

    def close_step(self, step_num: int) -> None:
        """Reset the state after updating the weights."""
        if self._trigger_cleanup(step_num=step_num):
            # reset transient tensors and accumulated scalar losses for the current step window
            self._state['tensors'] = {}
            self._state['scalars']['loss/total'] = 0.0
            self._state['scalars']['loss/mse/0'] = 0.0
            self._state['scalars']['loss/mse/k'] = 0.0
            self._state['scalars']['loss/cos/0'] = 0.0
            self._state['scalars']['loss/cos/k'] = 0.0
            # garbage collection
            mlable.models.free_memory()

    # VECTORIZE ################################################################

    def step_batch(self, step_num: int, batch_arr: object, column_str: str) -> None:
        """Vectorize a raw batch into mask, token ids, and byte patches."""
        __pad = self._config['batch'].get('padding_str', '')
        # common args
        __args = {
            'text_tok': self._text_tok,
            'byte_tok': self._byte_tok,
            'dtype_obj': self._config['batch'].get('dtype_obj', torch.long),
            'device_str': self._config['batch'].get('device_str', 'cpu'),
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

    def _teacher_forward(self, hidden_opt: bool, gradient_opt: bool=False) -> None:
        """Teacher forward path with explicit gradient mode."""
        with self._context(gradient_opt=gradient_opt):
            # teacher forward: get original embeddings and hidden states (no grad)
            self._state['tensors']['outputs/teacher/0'] = _processors.embed(
                indices_arr=self._state['tensors']['inputs/indices'],
                model_obj=self._teacher)
            # do not compute the hidden activations by default
            self._state['tensors']['outputs/teacher/k'] = torch.zeros(
                tuple(self._state['tensors']['outputs/teacher/0'].shape),
                dtype=self._state['tensors']['outputs/teacher/0'].dtype,
                device=self._state['tensors']['outputs/teacher/0'].device)
            # compute hidden activations only when a triggered computation needs them
            if hidden_opt:
                self._state['tensors']['outputs/teacher/k'] = _processors.forward(
                    embeds_arr=self._state['tensors']['outputs/teacher/0'],
                    mask_arr=self._state['tensors']['inputs/mask'],
                    model_obj=self._teacher)

    def _student_forward(self, hidden_opt: bool, gradient_opt: bool=True) -> None:
        """Student forward path with explicit gradient mode."""
        with self._context(gradient_opt=gradient_opt):
            self._state['tensors']['outputs/student/0'] = self._student(
                self._state['tensors']['inputs/bytes']
            ).to(dtype=self._state['tensors']['outputs/teacher/0'].dtype)
            self._state['tensors']['outputs/student/k'] = torch.zeros(
                tuple(self._state['tensors']['outputs/student/0'].shape),
                dtype=self._state['tensors']['outputs/student/0'].dtype,
                device=self._state['tensors']['outputs/student/0'].device)
            if hidden_opt:
                self._state['tensors']['outputs/student/k'] = _processors.forward(
                    embeds_arr=self._state['tensors']['outputs/student/0'],
                    mask_arr=self._state['tensors']['inputs/mask'],
                    model_obj=self._teacher)

    def step_forward(self, step_num: int) -> None:
        raise NotImplementedError('Subclasses of BaseRunner must implement step_forward().')

    # LOSS #####################################################################

    def _step_losses(self, gradient_opt: bool=True) -> None:
        """Compute tensor loss and detached scalar metrics for the current batch."""
        with self._context(gradient_opt=gradient_opt):
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

    def step_objective(self, step_num: int) -> None:
        """Base objective implementation that computes/accumulates losses without backward."""
        self._step_losses(gradient_opt=True)

    # BACKWARD #################################################################

    def _step_backward(self) -> None:
        """Accumulate gradients from the current tensor loss."""
        __loss = self._state['tensors'].get('loss/total', None)
        # expect a tensor, not a scalar
        assert hasattr(__loss, 'shape'), 'Missing `tensors["loss/total"]` before _step_backward().'
        # undo the float scaling before computing the backward pass
        self._scaler.scale(__loss).backward()

    # UPDATE ###################################################################

    def _step_optimizer(self, step_num: int) -> None:
        """Apply optimizer, scaler, scheduler, and gradient clipping on accumulation boundary."""
        __norm_max = float(self._config['gradient'].get('max_norm', 1.0))
        # only work every few steps, after accumulating the loss on a few batches
        if self._trigger_update(step_num=step_num):
            # gradient clipping; unscale first to get true grad norm
            self._scaler.unscale_(self._optimizer)
            self._state['scalars']['gradient/rate'] = _monitor.current_lr(self._optimizer)
            self._state['scalars']['gradient/norm'] = float(torch.nn.utils.clip_grad_norm_(
                self._student.parameters(),
                max_norm=__norm_max).item())
            # update the weights
            self._scaler.step(self._optimizer)
            self._scaler.update()
            # update the learning rate
            if self._scheduler is not None:
                self._scheduler.step()
            # reset the gradients
            self._optimizer.zero_grad()

    # CALLBACKS ################################################################

    def step_callbacks(self, step_num: int) -> None:
        """Run all triggered callbacks on the current state."""
        for __callback in self._callbacks:
            if __callback['trigger'](self._state):
                __callback['operation'](self._state)

    def close_callbacks(self) -> None:
        """Run cleanup on all registered callbacks."""
        for __callback in self._callbacks:
            __callback['cleanup']()

    # PROGRESS #################################################################

    def step_progress(self, step_num: int, pbar_obj: object) -> None:
        # only work every few steps, after accumulating the loss on a few batches
        if self._trigger_progress(step_num=step_num):
            # aggregate and format
            __stats = _callbacks.format_state(state=self._state['scalars'])
            # filter the epoch and step since they are already in the pbar
            pbar_obj.set_postfix({__k: __v for (__k, __v) in __stats.items() if (__k not in ['epoch', 'step'])})

# TRAINER ######################################################################

class PrefixTrainer(BaseRunner):
    """Prefix runner with gradient-based optimization behavior."""

    def _check_runtime(self) -> None:
        assert self._optimizer is not None, 'optimizer is None; call setup_global() first.'
        assert self._scaler is not None, 'scaler is None; call setup_global() first.'
        assert self._scheduler is not None, 'scheduler is None; call setup_phase() first.'

    def step_forward(self, step_num: int) -> None:
        __hidden = self._trigger_hidden(step_num=step_num)
        self._teacher_forward(hidden_opt=__hidden, gradient_opt=False)
        self._student_forward(hidden_opt=__hidden, gradient_opt=True)

    def step_objective(self, step_num: int) -> None:
        self._step_losses(gradient_opt=True)
        self._step_backward()
        self._step_optimizer(step_num=step_num)

# TESTER #######################################################################

class PrefixTester(BaseRunner):
    """Prefix runner for evaluation/benchmark phases without parameter updates."""

    def setup_global(
        self,
        context_cfg: dict={},
        optimizer_cfg: dict={},
        scaler_cfg: dict={},
        overwrite_opt: bool=False,
    ) -> None:
        if self._check_config(context_cfg, ('device', 'dtype')):
            self._config['context'] = context_cfg
        if self._check_config(optimizer_cfg, ('lr',)):
            self._config['optimizer'] = optimizer_cfg
        if self._check_config(scaler_cfg, ('enabled',)):
            self._config['scaler'] = scaler_cfg

    def step_forward(self, step_num: int) -> None:
        __hidden = self._trigger_hidden(step_num=step_num)
        self._teacher_forward(hidden_opt=__hidden, gradient_opt=False)
        self._student_forward(hidden_opt=__hidden, gradient_opt=False)

    def step_objective(self, step_num: int) -> None:
        self._step_losses(gradient_opt=False)

    def _trigger_test(self, step_num: int) -> bool:
        return True

    def _trigger_update(self, step_num: int) -> bool:
        return False

    def _trigger_progress(self, step_num: int) -> bool:
        return True

    def _trigger_cleanup(self, step_num: int) -> bool:
        return True
