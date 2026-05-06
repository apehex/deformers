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

class PrefixTrainer:
    """Thin orchestration class for prefix training phases.

    The trainer owns:
    - mutable runtime state
    - teacher/student references (external)
    - tokenizer references (external)
    - training utility setup from per-setup configuration (optimizer/scaler/context/scheduler/callbacks)
    - current configuration
    - core training loop operations

    Stateless transforms such as vectorization and loss computation remain in
    `deformers.pipelines.prefix.processors`.

    Typical lifecycle:
        trainer = PrefixTrainer(text_tok, byte_tok, teacher, student)
        trainer.setup_global(training_cfg=..., optimizer_cfg=..., scaler_cfg=...)
        trainer.setup_phase(dataset, epoch_num, column, batch_cfg=..., loss_cfg=..., ...)
        trainer.run_phase()                     # runs all epochs
        trainer.cleanup_callbacks()

        trainer.setup_phase(dataset2, epoch_num2, column2, batch_cfg=..., loss_cfg=..., ...)
        trainer.run_phase()
        trainer.cleanup_callbacks()
    """

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
        # current configuration
        self._config = {
            'batch': {},
            'loss': {},
            'gradient': {},
            'training': {},
            'logging': {},
            'optimizer': {},
            'scheduler': {},
            'scaler': {},
            'saving': {},
            'testing': {},
            'ema': {},
            'speed': {},
            'tboard': {},}
        # training utilities (populated by setup_* methods, not the constructor)
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        self._context = None
        self._callbacks = []
        # current phase attributes (set by setup_phase)
        self._dataset_obj = None
        self._column_str = None
        self._epoch_num = None
        # initialize the state
        self._state = self.init_state()

    # STATE ####################################################################

    def init_state(self, override: dict | None = None) -> dict[str, dict]:
        """Build the nested runtime state, optionally overriding `tensors` and `scalars`."""
        override = dict(override or {})
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

    # SETUP ####################################################################

    def _valid_config(self, config_dict: object, required_keys: tuple[str, ...]=()) -> bool:
        return isinstance(config_dict, dict) and all(__key in config_dict for __key in required_keys)

    def setup_state(self, override_cfg: dict={}) -> None:
        """Reinitialize the runtime state."""
        self._state = self.init_state(override_cfg)

    def setup_optimizer(self, optimizer_cfg: dict={}) -> None:
        """Create the AdamW optimizer from config."""
        if not self._valid_config(optimizer_cfg, ('lr',)):
            return
        __cfg = dict(optimizer_cfg)
        self._optimizer = torch.optim.AdamW(self._student.parameters(), **__cfg)
        self._optimizer.zero_grad()

    def setup_scaler(self, scaler_cfg: dict={}) -> None:
        """Create the GradScaler from config."""
        if not self._valid_config(scaler_cfg, ('enabled',)):
            return
        __cfg = dict(scaler_cfg)
        self._scaler = torch.amp.GradScaler(**__cfg)

    def setup_context(self, training_cfg: dict={}) -> None:
        """Create the autocast context from training config."""
        if not self._valid_config(training_cfg, ('device', 'dtype')):
            return
        __cfg = dict(training_cfg)
        __device = __cfg.get('device', 'cpu')
        __dtype = __cfg.get('dtype', torch.float32)
        if __dtype != torch.float32:
            self._context = torch.amp.autocast(device_type=__device, dtype=__dtype)
        else:
            self._context = contextlib.nullcontext()

    def setup_scheduler(self, scheduler_cfg: dict={}) -> None:
        """Create a WaveLR scheduler from config."""
        if not self._valid_config(scheduler_cfg, ('start_rate', 'end_rate', 'total_num', 'warmup_num')):
            return
        __cfg = dict(scheduler_cfg)
        self._scheduler = mlable.schedulers.WaveLR(optimizer_obj=self._optimizer, **__cfg)

    def setup_callbacks(
        self,
        speed_cfg: dict={},
        ema_cfg: dict={},
        logging_cfg: dict={},
        tboard_cfg: dict={},
        saving_cfg: dict={},
    ) -> None:
        """Build the phase callbacks from the provided configs."""
        __speed = dict(speed_cfg) if isinstance(speed_cfg, dict) else {}
        __ema = dict(ema_cfg) if isinstance(ema_cfg, dict) else {}
        __logging = dict(logging_cfg) if isinstance(logging_cfg, dict) else {}
        __tboard = dict(tboard_cfg) if isinstance(tboard_cfg, dict) else {}
        __saving = dict(saving_cfg) if isinstance(saving_cfg, dict) else {}
        __result = []
        if self._valid_config(__speed, ('every_num', 'batch_len')) and int(__speed.get('every_num', 0)) > 0 and int(__speed.get('batch_len', 0)) > 0:
            __result.append(_callbacks.prepare_speed_callback(**__speed))
        if self._valid_config(__ema, ('every_num', 'start_num', 'smooth_rate')) and int(__ema.get('every_num', 0)) > 0:
            __result.append(_callbacks.prepare_ema_callback(**__ema))
        if self._valid_config(__logging, ('every_num', 'path_str')) and int(__logging.get('every_num', 0)) > 0 and __logging.get('path_str'):
            __result.append(_callbacks.prepare_logging_callback(**__logging))
        if self._valid_config(__tboard, ('every_num', 'path_str')) and int(__tboard.get('every_num', 0)) > 0 and __tboard.get('path_str'):
            __result.append(_callbacks.prepare_tensorboard_callback(**__tboard))
        if self._valid_config(__saving, ('every_num', 'path_str')) and int(__saving.get('every_num', 0)) > 0 and __saving.get('path_str'):
            __result.append(_callbacks.prepare_saving_callback(model_obj=self._student, **__saving))
        self._callbacks = __result

    def setup_global(
        self,
        training_cfg: dict={},
        optimizer_cfg: dict={},
        scaler_cfg: dict={},
        overwrite_opt: bool=False,
    ) -> None:
        """Initialize long-lived utilities: optimizer, scaler, and mixed-precision context.

        Utilities that already exist are left unchanged unless overwrite_opt=True.
        Call once before the first phase; the optimizer persists across all phases.
        """
        self._config['training'] = dict(training_cfg) if isinstance(training_cfg, dict) else {}
        self._config['optimizer'] = dict(optimizer_cfg) if isinstance(optimizer_cfg, dict) else {}
        self._config['scaler'] = dict(scaler_cfg) if isinstance(scaler_cfg, dict) else {}
        if overwrite_opt or self._optimizer is None:
            self._optimizer = None
            self.setup_optimizer(optimizer_cfg=self._config['optimizer'])
        if overwrite_opt or self._scaler is None:
            self._scaler = None
            self.setup_scaler(scaler_cfg=self._config['scaler'])
        if overwrite_opt or self._context is None:
            self._context = None
            self.setup_context(training_cfg=self._config['training'])

    def setup_phase(
        self,
        dataset_obj: object,
        epoch_num: int,
        column_str: str,
        batch_cfg: dict={},
        loss_cfg: dict={},
        gradient_cfg: dict={},
        training_cfg: dict={},
        logging_cfg: dict={},
        scheduler_cfg: dict={},
        saving_cfg: dict={},
        testing_cfg: dict={},
        ema_cfg: dict={},
        speed_cfg: dict={},
        tboard_cfg: dict={},
    ) -> None:
        """Configure a training phase: store config, dataset info, rebuild scheduler and callbacks.

        The current phase config is replaced on each call.
        The scheduler is always recreated to use the phase-specific schedule.
        The callbacks are always recreated to use phase-specific paths and settings.
        The optimizer is preserved unless overwrite_opt=True is passed to setup_global().

        Args:
            dataset_obj: iterable dataset that supports len() (batches per epoch).
            epoch_num: number of epochs for this phase.
            column_str: dataset column to read; if 'indice' in name, use vectorize_indices.
            *_cfg: phase-local configuration dictionaries stored as the current config.
        """
        self._config['batch'] = dict(batch_cfg) if isinstance(batch_cfg, dict) else {}
        self._config['loss'] = dict(loss_cfg) if isinstance(loss_cfg, dict) else {}
        self._config['gradient'] = dict(gradient_cfg) if isinstance(gradient_cfg, dict) else {}
        self._config['training'] = dict(training_cfg) if isinstance(training_cfg, dict) else {}
        self._config['logging'] = dict(logging_cfg) if isinstance(logging_cfg, dict) else {}
        self._config['scheduler'] = dict(scheduler_cfg) if isinstance(scheduler_cfg, dict) else {}
        self._config['saving'] = dict(saving_cfg) if isinstance(saving_cfg, dict) else {}
        self._config['testing'] = dict(testing_cfg) if isinstance(testing_cfg, dict) else {}
        self._config['ema'] = dict(ema_cfg) if isinstance(ema_cfg, dict) else {}
        self._config['speed'] = dict(speed_cfg) if isinstance(speed_cfg, dict) else {}
        self._config['tboard'] = dict(tboard_cfg) if isinstance(tboard_cfg, dict) else {}
        # store phase attributes
        self._dataset_obj = dataset_obj
        self._column_str = column_str
        self._epoch_num = int(epoch_num)
        self._config['training']['epoch_num'] = self._epoch_num
        self._state['scalars']['epoch/total'] = self._epoch_num
        # always recreate phase-local utilities
        self._scheduler = None
        self._callbacks = []
        self.setup_scheduler(scheduler_cfg=self._config['scheduler'])
        self.setup_callbacks(
            speed_cfg=self._config['speed'],
            ema_cfg=self._config['ema'],
            logging_cfg=self._config['logging'],
            tboard_cfg=self._config['tboard'],
            saving_cfg=self._config['saving'])

    def validate_setup(self) -> None:
        """Raise AssertionError if the trainer is not ready to run a phase."""
        assert self._optimizer is not None, 'optimizer is None; call setup_global() first.'
        assert self._scaler is not None, 'scaler is None; call setup_global() first.'
        assert self._context is not None, 'context is None; call setup_global() first.'
        assert self._dataset_obj is not None, 'dataset is None; call setup_phase() first.'
        assert self._column_str is not None, 'column is None; call setup_phase() first.'
        assert self._epoch_num is not None, 'epoch count is None; call setup_phase() first.'

    # PHASE ####################################################################

    def run_phase(self) -> None:
        """Run all epochs of the current phase using config set by setup_phase()."""
        self.validate_setup()
        for __epoch in range(self._epoch_num):
            self.run_epoch(
                epoch_num=__epoch,
                epoch_tot=self._epoch_num,
                dataset_obj=self._dataset_obj,
                column_str=self._column_str)

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
        # step/global is a monotonically increasing counter that persists across epochs and phases
        self._state['scalars']['step/global'] += 1
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
            # update the weights
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
            if __callback['trigger'](self._state['scalars']):
                __callback['operation'](self._state['scalars'])

    def cleanup_callbacks(self) -> None:
        """Run cleanup on all registered callbacks."""
        for __callback in self._callbacks:
            __callback['cleanup']()

    # PROGRESS #################################################################

    def step_progress(self, pbar_obj: object) -> None:
        # aggregate and format
        __stats = _callbacks.format_state(state=self._state['scalars'])
        # filter the epoch and step since they are already in the pbar
        pbar_obj.set_postfix({__k: __v for (__k, __v) in __stats.items() if (__k not in ['epoch', 'step'])})
