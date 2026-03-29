from dataclasses import dataclass

import torch
import torch.nn
import transformers

# TRAINING #####################################################################

@dataclass
class PrefixTrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_steps: int = 100
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    hidden_state_weight: float = 1.0

def _validate_training_config(cfg: PrefixTrainingConfig) -> None:
    if cfg.grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be >= 1")

def collect_prefix_trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [__p for __p in model.parameters() if __p.requires_grad]

def build_prefix_optimizer(model: torch.nn.Module, cfg: PrefixTrainingConfig) -> torch.optim.Optimizer:
    _validate_training_config(cfg)
    __params = collect_prefix_trainable_parameters(model=model)
    if len(__params) == 0:
        raise ValueError("No trainable parameters found for prefix patch")
    return torch.optim.AdamW(__params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

def train_prefix_patch_step(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: PrefixTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    __input_ids = batch["input_ids"].to(device)
    __attention_mask = batch.get("attention_mask")
    if __attention_mask is not None:
        __attention_mask = __attention_mask.to(device)
    with torch.no_grad():
        __teacher = teacher_model.model(
            input_ids=__input_ids,
            attention_mask=__attention_mask,
            use_cache=False,)
    __amp = cfg.use_amp and (device.type == "cuda")
    with torch.autocast(device_type=device.type, enabled=__amp):
        __student = student_model.model(
            input_ids=__input_ids,
            attention_mask=__attention_mask,
            use_cache=False,)
        __loss = cfg.hidden_state_weight * torch.nn.functional.mse_loss(
            __student.last_hidden_state,
            __teacher.last_hidden_state,)
    __loss = __loss / cfg.grad_accumulation_steps
    if scaler is not None and __amp:
        scaler.scale(__loss).backward()
    else:
        __loss.backward()
    return {"loss": float(__loss.detach().cpu())}

def train_prefix_patch(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    dataloader,
    cfg: PrefixTrainingConfig,
    device: torch.device | None=None,
) -> list[dict[str, float]]:
    _validate_training_config(cfg)
    __device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(__device).train()
    teacher_model.to(__device).eval()
    optimizer = build_prefix_optimizer(model=student_model, cfg=cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and __device.type == "cuda"))
    __history = []
    __step = 0
    optimizer.zero_grad(set_to_none=True)
    while __step < cfg.max_steps:
        for __batch in dataloader:
            __metrics = train_prefix_patch_step(
                student_model=student_model,
                teacher_model=teacher_model,
                batch=__batch,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                device=__device,)
            if ((__step + 1) % cfg.grad_accumulation_steps) == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(collect_prefix_trainable_parameters(student_model), cfg.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            __history.append(__metrics)
            __step += 1
            if __step >= cfg.max_steps:
                break
    return __history
