from .lr_scheduler import LrScheduler
import zeus

if zeus.is_torch_backend():
    from .warmup_scheduler_torch import WarmupScheduler
elif zeus.is_tf_backend():
    from .warmup_scheduler_tf import WarmupScheduler
    from .multistep import MultiStepLR
    from .cosine_annealing import CosineAnnealingLR
    from .step_lr import StepLR
