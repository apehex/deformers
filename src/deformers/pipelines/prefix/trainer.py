"""Prefix pipeline runners.

The runners own:
- mutable runtime state
- teacher/student references (external)
- tokenizer references (external)
- setup and validation of runtime utilities
- shared phase / epoch / step orchestration

Stateless transforms such as vectorization and loss computation remain in
`deformers.pipelines.prefix.processors`.

Typical lifecycle:
    runner = PrefixTrainer(text_tok, byte_tok, teacher, student)
    runner.setup_global(context_cfg=..., optimizer_cfg=..., scaler_cfg=...)
    runner.setup_phase(dataset, phase_cfg=..., batch_cfg=..., loss_cfg=..., ...)
    runner.run_phase()
    runner.cleanup_callbacks()
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

import deformers.pipelines.monitor as _monitor
import deformers.pipelines.prefix.callbacks as _callbacks
import deformers.pipelines.prefix.processors as _processors

# GENERIC ######################################################################

def is_iterable(data: object) -> bool:
    try:
        data[0]
    except (IndexError, KeyError, TypeError):
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


# RUNNERS ######################################################################

class BaseRunner:
    """Common lifecycle and orchestration logic for prefix pipeline runners."""

    def __init__(
        self,
        text_tok: object,
        byte_tok: object,
        teacher_mod: object,
        student_mod: object,
    ) -> None:
        self._text_tok = text_tok
        self._byte_tok = byte_tok
        self._teacher = teacher_mod
        self._student = student_mod
        self._context = None
        self._dataset = None
        self._callbacks = []
        self._config = self.init_config()
        self._state = self.init_state()

    def init_config(self) -> dict[str, dict]:
        return {
            'context': {},
            'optimizer': {},
            'scaler': {},
            'phase': {},
            'batch': {},
            'loss': {},
            'gradient': {},
            'scheduler': {},
            'testing': {},
            'ema': {},
            'speed': {},
            'logging': {},
            'tboard': {},
            'saving': {},}

    def _default_train_switch(self) -> int:
        return 1

    def init_state(self, override: dict=None) -> dict[str, dict]:
        """Build the nested runtime state."""
        override = dict(override or {})
        return {
            'scalars': {
                **{
                    'switch/train': self._default_train_switch(),
                    'switch/grad': 0,
                    'switch/log': 0,
                    'switch/save': 0,
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
                **override.get('scalars', {}),},
            'tensors': override.get('tensors', {}),
            'metadata': {
                **{
                    'runner/name': self.__class__.__name__,
                    'phase/column': '',},
                **override.get('metadata', {}),},}

    # CONFIG ###################################################################

    def _check_config(self, config_dict: object, required_keys: tuple[str]=()) -> bool:
        return isinstance(config_dict, dict) and all(__key in config_dict for __key in required_keys)

    def _check_runner_setup(self) -> None:
        pass

    def _check_setup(self) -> None:
        assert self._context is not None, 'context is None; call setup_global() first.'
        assert self._dataset is not None, 'dataset is None; call setup_phase() first.'
        assert self._check_config(self._config['phase'], ('column_str', 'epoch_num')), 'phase config is incomplete; call setup_phase() with phase_cfg.'
        assert self._check_config(self._config['batch'], ('batch_dim', 'sequence_dim', 'patch_dim')), 'batch config is incomplete; call setup_phase() with batch_cfg.'
        assert self._check_config(self._config['loss'], ('mse_0_rate', 'mse_k_rate', 'cos_0_rate', 'cos_k_rate')), 'loss config is incomplete; call setup_phase() with loss_cfg.'
        self._check_runner_setup()

    # GLOBAL ###################################################################

    def setup_context(self, context_cfg: dict=None) -> None:
        context_cfg = dict(context_cfg or {})
        if self._check_config(context_cfg, ('device', 'dtype')):
            self._config['context'] = context_cfg
            self._context = contextlib.nullcontext()
            __dtype = context_cfg.get('dtype', torch.float32)
            if __dtype != torch.float32:
                self._context = torch.amp.autocast(
                    device_type=context_cfg.get('device', 'cpu'),
                    dtype=__dtype)

    def setup_global(
        self,
        context_cfg: dict=None,
        optimizer_cfg: dict=None,
        scaler_cfg: dict=None,
        overwrite_opt: bool=False,
    ) -> None:
        # keep the shared signature aligned with PrefixTrainer.setup_global()
        if overwrite_opt or self._context is None:
            self._context = None
            self.setup_context(context_cfg=context_cfg)

    # PHASE ####################################################################

    def setup_configs(
        self,
        phase_cfg: dict=None,
        batch_cfg: dict=None,
        loss_cfg: dict=None,
        gradient_cfg: dict=None,
    ) -> None:
        phase_cfg = dict(phase_cfg or {})
        batch_cfg = dict(batch_cfg or {})
        loss_cfg = dict(loss_cfg or {})
        gradient_cfg = dict(gradient_cfg or {})
        if self._check_config(phase_cfg, ('column_str', 'epoch_num')):
            self._config['phase'] = phase_cfg
            self._state['scalars']['epoch/total'] = int(self._config['phase']['epoch_num'])
            self._state['metadata']['phase/column'] = self._config['phase']['column_str']
        if self._check_config(batch_cfg, ('batch_dim', 'sequence_dim', 'patch_dim')):
            self._config['batch'] = batch_cfg
        if self._check_config(loss_cfg, ('mse_0_rate', 'mse_k_rate', 'cos_0_rate', 'cos_k_rate')):
            self._config['loss'] = loss_cfg
        if self._check_config(gradient_cfg, ('every_num', 'max_norm')):
            self._config['gradient'] = gradient_cfg

    def setup_callbacks(
        self,
        testing_cfg: dict=None,
        ema_cfg: dict=None,
        speed_cfg: dict=None,
        logging_cfg: dict=None,
        tboard_cfg: dict=None,
        saving_cfg: dict=None,
    ) -> None:
        testing_cfg = dict(testing_cfg or {})
        ema_cfg = dict(ema_cfg or {})
        speed_cfg = dict(speed_cfg or {})
        logging_cfg = dict(logging_cfg or {})
        tboard_cfg = dict(tboard_cfg or {})
        saving_cfg = dict(saving_cfg or {})
        self._callbacks = []
        self._config['testing'] = testing_cfg
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
        phase_cfg: dict=None,
        batch_cfg: dict=None,
        loss_cfg: dict=None,
        gradient_cfg: dict=None,
        scheduler_cfg: dict=None,
        testing_cfg: dict=None,
        ema_cfg: dict=None,
        speed_cfg: dict=None,
        logging_cfg: dict=None,
        tboard_cfg: dict=None,
        saving_cfg: dict=None,
    ) -> None:
        # keep the shared signature aligned with PrefixTrainer.setup_phase()
        self._dataset = dataset_obj
        self.setup_configs(
            phase_cfg=phase_cfg,
            batch_cfg=batch_cfg,
            loss_cfg=loss_cfg,
            gradient_cfg=gradient_cfg)
        self.setup_callbacks(
            testing_cfg=testing_cfg,
            ema_cfg=ema_cfg,
            speed_cfg=speed_cfg,
            logging_cfg=logging_cfg,
            tboard_cfg=tboard_cfg,
            saving_cfg=saving_cfg)

    def run_phase(self) -> None:
        self._check_setup()
        __epoch_tot = int(self._config['phase']['epoch_num'])
        __column_str = self._config['phase']['column_str']
        for __epoch in range(__epoch_tot):
            self.run_epoch(
                epoch_num=__epoch,
                epoch_tot=__epoch_tot,
                dataset_obj=self._dataset,
                column_str=__column_str)

    # EPOCH ####################################################################

    def init_epoch(
        self,
        epoch_num: int,
        epoch_tot: int,
        dataset_obj: object,
    ) -> object:
        __step_tot = len(dataset_obj)
        self._state['scalars']['epoch/total'] = int(epoch_tot)
        self._state['scalars']['epoch/current'] = int(epoch_num) + 1
        self._state['scalars']['step/total'] = int(__step_tot)
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
        __pbar = self.init_epoch(
            epoch_num=epoch_num,
            epoch_tot=epoch_tot,
            dataset_obj=dataset_obj)
        for __step, __batch in enumerate(__pbar):
            self.init_step(step_num=__step)
            self.run_step(batch_arr=__batch, column_str=column_str)
            self.step_progress(__pbar)
            self.close_step()
        self.close_epoch(__pbar)

    def close_epoch(self, pbar_obj: object) -> None:
        pbar_obj.close()

    # STEP #####################################################################

    def _resolve_train_switch(self, step_num: int, test_every: int) -> int:
        return int((test_every < 1) or ((step_num % test_every) != 0))

    def _resolve_grad_switch(self, step_num: int, grad_every: int) -> int:
        return int((step_num % grad_every) == 0)

    def init_step(self, step_num: int) -> None:
        __step_num = int(step_num) + 1
        self._state['scalars']['step/global'] += 1
        self._state['scalars']['step/current'] = __step_num
        __test_every = int(self._config['testing'].get('every_num', 0))
        __log_every = int(self._config['logging'].get('every_num', 0))
        __save_every = int(self._config['saving'].get('every_num', 0))
        __grad_every = int(self._config['gradient'].get('every_num', 1))
        self._state['scalars']['switch/train'] = self._resolve_train_switch(__step_num, __test_every)
        self._state['scalars']['switch/log'] = int((__log_every > 0) and ((__step_num % __log_every) == 0))
        self._state['scalars']['switch/save'] = int((__save_every > 0) and ((__step_num % __save_every) == 0))
        self._state['scalars']['switch/grad'] = self._resolve_grad_switch(__step_num, __grad_every)

    def run_step(
        self,
        batch_arr: object,
        column_str: str,
    ) -> None:
        self.step_batch(batch_arr=batch_arr, column_str=column_str)
        self.step_forward()
        self.step_losses()
        self.step_update()
        self.step_callbacks()

    def step_update(self) -> None:
        pass

    def _should_reset_state(self) -> bool:
        return True

    def close_step(self) -> None:
        if self._should_reset_state():
            self._state['tensors'] = {}
            self._state['scalars']['loss/total'] = 0.0
            self._state['scalars']['loss/mse/0'] = 0.0
            self._state['scalars']['loss/mse/k'] = 0.0
            self._state['scalars']['loss/cos/0'] = 0.0
            self._state['scalars']['loss/cos/k'] = 0.0
            mlable.models.free_memory()

    # VECTORIZE ################################################################

    def step_batch(self, batch_arr: object, column_str: str) -> None:
        __pad = self._config['batch'].get('padding_str', '')
        __args = {
            'text_tok': self._text_tok,
            'byte_tok': self._byte_tok,
            'dtype_obj': self._config['batch'].get('dtype_obj', torch.long),
            'device_str': self._config['batch'].get('device_str', 'cpu'),
            'sequence_dim': int(self._config['batch']['sequence_dim']),
            'patch_dim': int(self._config['batch']['patch_dim']),
            'left_pad': bool(self._config['batch'].get('left_pad', True)),}
        if 'indice' in column_str:
            __tensors = _processors.vectorize_indices(indices_arr=batch_arr[column_str], padding_str=__pad, **__args)
        else:
            __tensors = _processors.vectorize_strings(text_arr=batch_arr[column_str], **__args)
        self._state['tensors']['inputs/mask'] = __tensors[0]
        self._state['tensors']['inputs/indices'] = __tensors[1]
        self._state['tensors']['inputs/bytes'] = __tensors[2]

    # FORWARD ##################################################################

    def _hidden_outputs_enabled(self) -> bool:
        return (
            (float(self._config['loss'].get('mse_k_rate', 0.0)) > 0.0)
            or (float(self._config['loss'].get('cos_k_rate', 0.0)) > 0.0))

    def _student_grad_context(self) -> object:
        return contextlib.nullcontext()

    def _teacher_embed(self) -> torch.Tensor:
        return _processors.embed(
            indices_arr=self._state['tensors']['inputs/indices'],
            model_obj=self._teacher)

    def _teacher_forward(self, embeds_arr: torch.Tensor) -> torch.Tensor:
        return _processors.forward(
            embeds_arr=embeds_arr,
            mask_arr=self._state['tensors']['inputs/mask'],
            model_obj=self._teacher)

    def _student_embed(self) -> torch.Tensor:
        return self._student(self._state['tensors']['inputs/bytes'])

    def _student_forward(self, embeds_arr: torch.Tensor) -> torch.Tensor:
        return _processors.forward(
            embeds_arr=embeds_arr,
            mask_arr=self._state['tensors']['inputs/mask'],
            model_obj=self._teacher)

    def step_forward(self) -> None:
        __hidden = self._hidden_outputs_enabled()
        with self._context:
            with torch.no_grad():
                __teacher_0 = self._teacher_embed()
                __teacher_k = torch.zeros(
                    tuple(__teacher_0.shape),
                    dtype=__teacher_0.dtype,
                    device=__teacher_0.device)
                if __hidden:
                    __teacher_k = self._teacher_forward(embeds_arr=__teacher_0)
            with self._student_grad_context():
                __student_0 = self._student_embed().to(dtype=__teacher_0.dtype)
                __student_k = torch.zeros(
                    tuple(__student_0.shape),
                    dtype=__student_0.dtype,
                    device=__student_0.device)
                if __hidden:
                    __student_k = self._student_forward(embeds_arr=__student_0)
        self._state['tensors']['outputs/teacher/0'] = __teacher_0
        self._state['tensors']['outputs/teacher/k'] = __teacher_k
        self._state['tensors']['outputs/student/0'] = __student_0
        self._state['tensors']['outputs/student/k'] = __student_k

    # LOSS #####################################################################

    def step_losses(self) -> None:
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
        self._state['tensors']['loss/total'] = __outputs[-1]
        self._state['scalars']['loss/mse/0'] += float(__outputs[0].item())
        self._state['scalars']['loss/mse/k'] += float(__outputs[1].item())
        self._state['scalars']['loss/cos/0'] += float(__outputs[2].item())
        self._state['scalars']['loss/cos/k'] += float(__outputs[3].item())
        self._state['scalars']['loss/total'] += float(__outputs[4].item())

    # CALLBACKS ################################################################

    def step_callbacks(self) -> None:
        for __callback in self._callbacks:
            if __callback['trigger'](self._state):
                __callback['operation'](self._state)

    def cleanup_callbacks(self) -> None:
        for __callback in self._callbacks:
            __callback['cleanup']()

    def close_callbacks(self) -> None:
        self.cleanup_callbacks()

    # PROGRESS #################################################################

    def _should_report_progress(self) -> bool:
        return True

    def step_progress(self, pbar_obj: object) -> None:
        if self._should_report_progress():
            __stats = _callbacks.format_state(state=self._state)
            pbar_obj.set_postfix({__k: __v for (__k, __v) in __stats.items() if (__k not in ['epoch', 'step'])})


class PrefixTrainer(BaseRunner):
    """Training runner with gradient updates, optimizer, scaler, and scheduler."""

    def __init__(
        self,
        text_tok: object,
        byte_tok: object,
        teacher_mod: object,
        student_mod: object,
    ) -> None:
        super().__init__(
            text_tok=text_tok,
            byte_tok=byte_tok,
            teacher_mod=teacher_mod,
            student_mod=student_mod)
        self._optimizer = None
        self._scaler = None
        self._scheduler = None

    def _check_runner_setup(self) -> None:
        assert self._optimizer is not None, 'optimizer is None; call setup_global() first.'
        assert self._scaler is not None, 'scaler is None; call setup_global() first.'
        assert self._scheduler is not None, 'scheduler is None; call setup_phase() first.'

    def setup_optimizer(self, optimizer_cfg: dict=None) -> None:
        optimizer_cfg = dict(optimizer_cfg or {})
        if self._check_config(optimizer_cfg, ('lr',)):
            self._config['optimizer'] = optimizer_cfg
            self._optimizer = torch.optim.AdamW(self._student.parameters(), **optimizer_cfg)
            self._optimizer.zero_grad()

    def setup_scaler(self, scaler_cfg: dict=None) -> None:
        scaler_cfg = dict(scaler_cfg or {})
        if self._check_config(scaler_cfg, ('enabled',)):
            self._config['scaler'] = scaler_cfg
            self._scaler = torch.amp.GradScaler(**scaler_cfg)

    def setup_scheduler(self, scheduler_cfg: dict=None) -> None:
        scheduler_cfg = dict(scheduler_cfg or {})
        if self._check_config(scheduler_cfg, ('start_rate', 'end_rate', 'total_num', 'warmup_num')):
            self._config['scheduler'] = scheduler_cfg
            self._scheduler = mlable.schedulers.WaveLR(optimizer_obj=self._optimizer, **scheduler_cfg)

    def setup_global(
        self,
        context_cfg: dict=None,
        optimizer_cfg: dict=None,
        scaler_cfg: dict=None,
        overwrite_opt: bool=False,
    ) -> None:
        super().setup_global(
            context_cfg=context_cfg,
            optimizer_cfg=optimizer_cfg,
            scaler_cfg=scaler_cfg,
            overwrite_opt=overwrite_opt)
        if overwrite_opt or self._optimizer is None:
            self._optimizer = None
            self.setup_optimizer(optimizer_cfg=optimizer_cfg)
        if overwrite_opt or self._scaler is None:
            self._scaler = None
            self.setup_scaler(scaler_cfg=scaler_cfg)

    def setup_phase(
        self,
        dataset_obj: object,
        phase_cfg: dict=None,
        batch_cfg: dict=None,
        loss_cfg: dict=None,
        gradient_cfg: dict=None,
        scheduler_cfg: dict=None,
        testing_cfg: dict=None,
        ema_cfg: dict=None,
        speed_cfg: dict=None,
        logging_cfg: dict=None,
        tboard_cfg: dict=None,
        saving_cfg: dict=None,
    ) -> None:
        super().setup_phase(
            dataset_obj=dataset_obj,
            phase_cfg=phase_cfg,
            batch_cfg=batch_cfg,
            loss_cfg=loss_cfg,
            gradient_cfg=gradient_cfg,
            scheduler_cfg=scheduler_cfg,
            testing_cfg=testing_cfg,
            ema_cfg=ema_cfg,
            speed_cfg=speed_cfg,
            logging_cfg=logging_cfg,
            tboard_cfg=tboard_cfg,
            saving_cfg=saving_cfg)
        self._scheduler = None
        self.setup_scheduler(scheduler_cfg=scheduler_cfg)

    def step_update(self) -> None:
        self.step_backward()
        self.step_optimizer()

    def _should_reset_state(self) -> bool:
        return bool(self._state['scalars']['switch/grad'])

    def _should_report_progress(self) -> bool:
        return bool(self._state['scalars']['switch/grad'])

    def step_backward(self) -> None:
        __loss = self._state['tensors'].get('loss/total', None)
        assert hasattr(__loss, 'shape'), 'Missing `tensors["loss/total"]` before step_backward().'
        self._scaler.scale(__loss).backward()

    def step_optimizer(self) -> None:
        __norm_max = float(self._config['gradient'].get('max_norm', 1.0))
        if bool(self._state['scalars']['switch/grad']):
            self._scaler.unscale_(self._optimizer)
            self._state['scalars']['gradient/rate'] = _monitor.current_lr(self._optimizer)
            self._state['scalars']['gradient/norm'] = float(torch.nn.utils.clip_grad_norm_(
                self._student.parameters(),
                max_norm=__norm_max).item())
            self._scaler.step(self._optimizer)
            self._scaler.update()
            if self._scheduler is not None:
                self._scheduler.step()
            self._optimizer.zero_grad()


class PrefixTester(BaseRunner):
    """Evaluation runner that shares the lifecycle but never tracks student grads."""

    def _default_train_switch(self) -> int:
        return 0

    def _resolve_train_switch(self, _step_num: int, _test_every: int) -> int:
        return 0

    def _resolve_grad_switch(self, _step_num: int, _grad_every: int) -> int:
        return 0

    def _student_grad_context(self) -> object:
        return torch.no_grad()
