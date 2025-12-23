import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import json
import numpy as np
import torch

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
class DDIMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine"):
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    a0 = alphas_bar_sqrt[0].clone()
    aT = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt = (alphas_bar_sqrt - aT) * (a0 / (a0 - aT))
    alphas_bar = alphas_bar_sqrt**2
    alphas = torch.cat([alphas_bar[0:1], alphas_bar[1:] / alphas_bar[:-1]])
    return 1 - alphas


class DDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    DDIM with TVP + mask-local η.

    Update rule (per-position local blending):
        x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat
                  + [ (1-M)*sqrt(1-alpha_bar_{t-1}-a_out^2) + M*sqrt(1-alpha_bar_{t-1}-a_in^2) ] * eps_hat
                  + [ (1-M)*a_out + M*a_in ] * z,   z ~ N(0, I)

    a_out = eta * sqrt(tilde_beta_t)
    a_in  = eta_mask_eff * sqrt(tilde_beta_t)

    eta_mask_segmented=True: remap the schedule only inside the “positive (feasible) region”
    (segment start = max, segment end = 0). Outside the region, eta_mask=0.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        skip_prk_steps: bool = True,

        # ---- eta_mask scheduler settings ----
        eta_mask_use_schedule: bool = False,
        eta_mask_schedule: str = "constant",     # "constant","linear_down","linear_up","cosine_down","cosine_up",
                                                 # "poly_down","poly_up","exp_down","exp_up",
                                                 # "sigmoid_down","sigmoid_up"
        eta_mask_min: float = 0.0,
        eta_mask_max: float = 2.1,
        eta_mask_power: float = 2.0,
        eta_mask_exp_k: float = 3.0,
        eta_mask_sigmoid_k: float = 8.0,
        eta_mask_stop_step: int = 999999999,

        # ---- segment (positive-region) remapping options ----
        eta_mask_segmented: bool = False,        # If True, remap only inside positive region to s∈[0,1]
                                                 # (segment start = max, segment end = 0). Outside region is 0.
        # ---- threshold guard ----
        eta_mask_guard: str = "none",            # Mainly meaningful when segmented=False ("none"|"clip_to_crit"|"zero_before_neg")
        eta_mask_guard_margin: float = 0.99,     # slack ratio (eta ≤ margin * eta_crit)
    ):
        super().__init__()
        self.skip_prk_steps = skip_prk_steps

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
        self._t_to_index = None
        self._step_stride = None

        # segment cache
        self._pos_segments: List[Tuple[int, int]] = []  # [(start_idx, end_idx_inclusive)]

    @classmethod
    def from_pretrained(cls, scheduler_folder: str):
        import os
        config_path = os.path.join(scheduler_folder, "scheduler_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find scheduler_config.json in {scheduler_folder}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps}."
            )
        self.num_inference_steps = num_inference_steps

        if self.config.timestep_spacing == "linspace":
            ts = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps).round()[::-1].astype(np.int64)
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            ts = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
            ts += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            ts = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64) - 1
        else:
            raise ValueError(f"{self.config.timestep_spacing} is not supported.")
        self.timesteps = torch.from_numpy(ts).to(device)
        self._t_to_index = {int(t): i for i, t in enumerate(self.timesteps.tolist())}

        # store stride
        self._step_stride = (self.config.num_train_timesteps // self.num_inference_steps)

        # pre-compute positive (feasible) segments
        self._compute_positive_segments()

    # ---- positive-segment computation ----
    def _compute_positive_segments(self):
        """Find regions where eta_mask_max is allowed (eta_mask_max ≤ margin*eta_crit),
        and store consecutive True blocks as segments."""
        self._pos_segments = []
        if not (self.config.eta_mask_use_schedule and self.config.eta_mask_segmented):
            return
        if self._t_to_index is None or self._step_stride is None:
            return

        mx = float(self.config.eta_mask_max)
        margin = float(self.config.eta_mask_guard_margin)

        feasible = []
        for i, t in enumerate(self.timesteps.tolist()):
            prev_t = t - self._step_stride
            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
            b_prev = 1 - a_prev
            # posterior var
            b_t = 1 - a_t
            tilde_beta = (b_prev / b_t) * (1 - a_t / a_prev)
            if tilde_beta.item() <= 0:
                feasible.append(False)
                continue
            eta_crit = math.sqrt(max((b_prev.item()) / tilde_beta.item(), 0.0))
            feasible.append(mx <= margin * eta_crit)

        # extract consecutive True blocks (one chunk)
        start = None
        for i, ok in enumerate(feasible):
            if ok and start is None:
                start = i
            if (not ok or i == len(feasible) - 1) and start is not None:
                end = i if not ok else i  # inclusive
                self._pos_segments.append((start, end))
                start = None

        # apply stop_step cut if needed
        stop = int(self.config.eta_mask_stop_step)
        if stop < 1e8 and self._pos_segments:
            new_segments = []
            for s, e in self._pos_segments:
                if s >= stop:
                    continue
                new_segments.append((s, min(e, stop - 1)))
            self._pos_segments = new_segments

    # ---- schedule function (0..1 -> [min,max]) ----
    def _eta_sched_shape(self, s: float) -> float:
        """Return a 0..1 → 0..1 shape only (supports both down/up)."""
        sch = self.config.eta_mask_schedule
        s = max(0.0, min(1.0, s))

        if sch == "constant":
            return 1.0
        elif sch == "linear_down":
            return 1.0 - s
        elif sch == "linear_up":
            return s
        elif sch == "cosine_down":
            return 0.5 * (1.0 + math.cos(math.pi * s))  # 1→0
        elif sch == "cosine_up":
            return 0.5 * (1.0 - math.cos(math.pi * s))  # 0→1
        elif sch == "poly_down":
            p = float(self.config.eta_mask_power)
            return (1.0 - s) ** max(p, 1e-6)
        elif sch == "poly_up":
            p = float(self.config.eta_mask_power)
            return s ** max(p, 1e-6)
        elif sch == "exp_down":
            k = float(self.config.eta_mask_exp_k)
            return math.exp(-k * s)
        elif sch == "exp_up":
            k = float(self.config.eta_mask_exp_k)
            return 1.0 - math.exp(-k * s)
        elif sch == "sigmoid_down":
            k = float(self.config.eta_mask_sigmoid_k)
            sig = 1.0 / (1.0 + math.exp(k * (s - 0.5)))
            return max(0.0, min(1.0, (sig - 0.5) / 0.5))
        elif sch == "sigmoid_up":
            k = float(self.config.eta_mask_sigmoid_k)
            sig = 1.0 / (1.0 + math.exp(-k * (s - 0.5)))
            return max(0.0, min(1.0, (sig - 0.5) / 0.5))
        else:
            return 1.0  # fallback

    def _eta_mask_scheduled_value_global(self, step_index: int, num_steps: int) -> float:
        """Legacy full-range schedule (used when segmented=False)."""
        if num_steps <= 1:
            s = 0.0
        else:
            s = step_index / float(num_steps - 1)  # 0 -> 1
        w01 = self._eta_sched_shape(s)
        mn = float(self.config.eta_mask_min)
        mx = float(self.config.eta_mask_max)
        return mn + (mx - mn) * w01

    # ----- core: one step update -----
    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,          # \hat\epsilon
        timestep: int,
        sample: torch.Tensor,                # x_t
        eta: float = 0.0,                    # global eta
        *,
        eta_mask: Optional[float] = None,    # used when schedule is off
        mask_for_anomaly: Optional[torch.Tensor] = None,
        use_random_mask: bool = False,
        random_mask_ratio: float = 1.0,
        random_mask_full_image: bool = False,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        anomaly_strength: float = 0.0,       # not used in TVP
    ) -> Union[DDIMSchedulerOutput, Tuple[torch.Tensor]]:

        if self.num_inference_steps is None:
            raise ValueError("Need to call set_timesteps(...) after creating the scheduler")

        step_stride = self._step_stride or (self.config.num_train_timesteps // self.num_inference_steps)
        prev_timestep = timestep - step_stride

        a_t    = self.alphas_cumprod[timestep]
        a_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        b_prev = 1 - a_prev

        # tilde_beta_t
        b_t = 1 - a_t
        tilde_beta_t = (b_prev / b_t) * (1 - a_t / a_prev)
        sqrt_tilde_beta = tilde_beta_t.sqrt()

        # x0_hat, eps_hat
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - (1 - a_t).sqrt() * model_output) / a_t.sqrt()
            eps_hat = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            eps_hat = (sample - a_t.sqrt() * pred_original_sample) / (1 - a_t).sqrt()
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = a_t.sqrt() * sample - (1 - a_t).sqrt() * model_output
            eps_hat = a_t.sqrt() * model_output + (1 - a_t).sqrt() * sample
        else:
            raise ValueError(f"Unknown prediction_type: {self.config.prediction_type}")

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        if use_clipped_model_output:
            eps_hat = (sample - a_t.sqrt() * pred_original_sample) / (1 - a_t).sqrt()
        # import pdb;pdb.set_trace()
        # ===== mask =====
        M = None
        if (mask_for_anomaly is not None) and (mask_for_anomaly.shape[0] == sample.shape[0]):
            M = mask_for_anomaly
            if M.ndim == 4 and M.shape[1] == 1:
                M = M.repeat(1, sample.shape[1], 1, 1)
            M = M.to(sample.dtype).to(sample.device)
        else:
            eta_mask = None  # disable local eta if mask is missing

        # ===== global std =====
        a_out = eta * sqrt_tilde_beta

        # ===== determine eta_mask =====
        eta_mask_eff = None
        if M is not None:
            step_index = self._t_to_index[int(timestep)] if self._t_to_index is not None else 0
            margin = float(self.config.eta_mask_guard_margin)

            if self.config.eta_mask_use_schedule and self.config.eta_mask_segmented:
                # if inside positive (feasible) segment, remap to s∈[0,1]
                # outside segment -> 0
                val = 0.0
                for s_idx, e_idx in self._pos_segments:
                    if s_idx <= step_index <= e_idx:
                        seg_len = max(1, e_idx - s_idx)
                        s_rel = (step_index - s_idx) / seg_len  # 0→1
                        w01 = self._eta_sched_shape(s_rel)
                        mx = float(self.config.eta_mask_max)
                        mn = 0.0  # 0 at the end of segment
                        val = mn + (mx - mn) * w01
                        break

                # force 0 after stop_step
                if step_index >= int(self.config.eta_mask_stop_step):
                    val = 0.0

                # safety clip (even for upward shapes, prevent overly large values)
                if tilde_beta_t.item() > 0:
                    eta_crit = math.sqrt(max((b_prev.item()) / tilde_beta_t.item(), 0.0))
                    val = min(val, margin * eta_crit)
                eta_mask_eff = float(val)
                

            elif self.config.eta_mask_use_schedule:
                # legacy full-range schedule + (optional) guard
                num_steps = len(self.timesteps) if self._t_to_index is not None else (self.num_inference_steps or 1)
                base_eta_mask = self._eta_mask_scheduled_value_global(step_index, num_steps)
                
                if step_index >= int(self.config.eta_mask_stop_step):
                    base_eta_mask = 0.0

                if tilde_beta_t.item() > 0:
                    eta_crit = math.sqrt(max((b_prev.item()) / tilde_beta_t.item(), 0.0))
                else:
                    eta_crit = float("inf")

                guard = self.config.eta_mask_guard
                if guard == "clip_to_crit":
                    eta_mask_eff = min(float(base_eta_mask), margin * eta_crit)
                elif guard == "zero_before_neg":
                    eta_mask_eff = float(base_eta_mask) if (float(base_eta_mask) < margin * eta_crit) else 0.0
                else:
                    eta_mask_eff = float(base_eta_mask)

            else:
                # schedule off: use step argument (+optional guard)
                base_eta_mask = eta_mask if (eta_mask is not None) else eta
                if tilde_beta_t.item() > 0:
                    eta_crit = math.sqrt(max((b_prev.item()) / tilde_beta_t.item(), 0.0))
                else:
                    eta_crit = float("inf")

                guard = self.config.eta_mask_guard
                if guard == "clip_to_crit":
                    eta_mask_eff = min(float(base_eta_mask), margin * eta_crit)
                elif guard == "zero_before_neg":
                    eta_mask_eff = float(base_eta_mask) if (float(base_eta_mask) < margin * eta_crit) else 0.0
                else:
                    eta_mask_eff = float(base_eta_mask)

        # final a_in
        a_in = (eta_mask_eff * sqrt_tilde_beta) if (eta_mask_eff is not None) else a_out

        # ===== TVP coefficients =====
        coeff_out = torch.sqrt(torch.clamp(b_prev - a_out**2, min=0.0))
        in_rad   = b_prev - a_in**2
        coeff_in = torch.sqrt(torch.clamp(in_rad, min=0.0))
        noise_in = torch.where(in_rad < 0, b_prev.sqrt(), a_in)

        if M is None:
            drift_coeff = coeff_out
            noise_coeff = a_out
        else:
            drift_coeff = (1.0 - M) * coeff_out + M * coeff_in
            noise_coeff = (1.0 - M) * a_out      + M * noise_in

        # ===== update =====
        prev_sample = a_prev.sqrt() * pred_original_sample + drift_coeff * eps_hat

        if variance_noise is None:
            z = randn_tensor(model_output.shape, generator=generator,
                             device=model_output.device, dtype=model_output.dtype)
        else:
            z = variance_noise
        prev_sample = prev_sample + noise_coeff * z

        if not return_dict:
            return (prev_sample,)
        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    # ---- diffusers compatibility utils ----
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        a = self.alphas_cumprod.to(dtype=original_samples.dtype)
        t = timesteps.to(original_samples.device)
        sa, so = a[t].sqrt().flatten(), (1 - a[t]).sqrt().flatten()
        while len(sa.shape) < len(original_samples.shape): sa = sa.unsqueeze(-1)
        while len(so.shape) < len(original_samples.shape): so = so.unsqueeze(-1)
        return sa * original_samples + so * noise

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        a = self.alphas_cumprod.to(dtype=sample.dtype)
        t = timesteps.to(sample.device)
        sa, so = a[t].sqrt().flatten(), (1 - a[t]).sqrt().flatten()
        while len(sa.shape) < len(sample.shape): sa = sa.unsqueeze(-1)
        while len(so.shape) < len(sample.shape): so = so.unsqueeze(-1)
        return sa * noise - so * sample

    def __len__(self):
        return self.config.num_train_timesteps
