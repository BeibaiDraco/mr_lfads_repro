"""MR-LFADS architecture.

This is a faithful reimplementation of the architecture described in:
"Accurate Identification of Communication Between Multiple Interacting Neural Populations"
(B. Liu, J. Sacks, M. D. Golub; ICML 2025).

Important note
--------------
The paper leaves one implementation detail ambiguous: Eq. 2 injects messages m_t into
region-i generator updates at time t, while Eqs. 7-8 infer m_t from source-region r_t,
which itself depends on generator states at time t. The public paper does not specify how
that within-timestep cycle was broken in code. This implementation therefore exposes two
causal choices:

1. message_timing='lagged' (default): messages emitted at t are computed from r_t and are
   consumed by target generators at t+1.
2. message_timing='fixed_point': a small fixed-point loop is run inside each timestep to
   approximate the instantaneous coupling implied by the equations.

Everything else follows the paper directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, Tuple

import numpy as np
import torch
from torch import nn

from .initializers import init_linear_
from .losses import diag_gaussian_kl, gaussian_nll, poisson_nll
from .recurrent import BidirectionalClippedGRU, ClippedGRU, ClippedGRUCell


Edge = Tuple[int, int]
MessageSource = Literal["rates", "factors", "generator"]
MessageTiming = Literal["lagged", "fixed_point"]
OutputType = Literal["gaussian", "poisson"]


class KernelNormalizedLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        normed_weight = nn.functional.normalize(self.weight, p=2, dim=1)
        return nn.functional.linear(input, normed_weight, self.bias)


@dataclass
class MRLFADSConfig:
    region_dims: list[int]
    tau: int
    main_steps: int
    output_type: OutputType = "gaussian"
    g0_encoder_dim: int | list[int] | None = None
    causal_encoder_dim: int | list[int] | None = None
    controller_dim: int | list[int] | None = None
    gen_dim: int | list[int] | None = None
    fac_dim: int | list[int] | None = None
    input_dim: int = 4
    message_dim: int = 4
    message_from: MessageSource = "rates"
    message_timing: MessageTiming = "lagged"
    fixed_point_iters: int = 2
    prior_variance: float = 1.0
    posterior_var_min: float = 1.0e-4
    dropout_rate: float = 0.02
    cell_clip: float = 5.0
    beta_u: float = 0.1
    beta_m: float = 0.01
    beta_g0: float | None = None
    beta_u_start_epoch: int = 50
    beta_u_increase_epoch: int = 200
    beta_m_start_epoch: int = 150
    beta_m_increase_epoch: int = 100
    l2_alpha: float = 1.0e-4
    l2_start_epoch: int = 0
    l2_increase_epoch: int = 80
    loss_scale: float = 1.0
    kl_floor: float = 1.0e-3

    def __post_init__(self) -> None:
        n_regions = len(self.region_dims)

        def resolve(spec: int | list[int] | None, default_fn) -> list[int]:
            if spec is None:
                return [int(default_fn(i, d)) for i, d in enumerate(self.region_dims)]
            if isinstance(spec, int):
                return [int(spec) for _ in range(n_regions)]
            if len(spec) != n_regions:
                raise ValueError(f"Expected {n_regions} entries, got {len(spec)}.")
            return [int(x) for x in spec]

        self.gen_dims = resolve(self.gen_dim, lambda _i, d: 2 * d)
        # Paper appendix sets Nfac_i = Nneu_i.
        self.fac_dims = resolve(self.fac_dim, lambda _i, d: d)
        self.g0_encoder_dims = resolve(self.g0_encoder_dim, lambda i, _d: self.gen_dims[i])
        self.causal_encoder_dims = resolve(self.causal_encoder_dim, lambda i, _d: self.gen_dims[i])
        self.controller_dims = resolve(self.controller_dim, lambda i, _d: self.gen_dims[i])

        if self.beta_g0 is None:
            self.beta_g0 = self.beta_u
        if self.message_timing == "fixed_point" and self.fixed_point_iters < 1:
            raise ValueError("fixed_point_iters must be >= 1.")


class RegionModule(nn.Module):
    def __init__(
        self,
        region_index: int,
        n_obs: int,
        incoming_dim: int,
        config: MRLFADSConfig,
        gen_dim: int,
        fac_dim: int,
        g0_encoder_dim: int,
        causal_encoder_dim: int,
        controller_dim: int,
    ) -> None:
        super().__init__()
        self.region_index = region_index
        self.n_obs = n_obs
        self.incoming_dim = incoming_dim
        self.output_type = config.output_type
        self.gen_dim = int(gen_dim)
        self.fac_dim = int(fac_dim)
        self.input_dim = int(config.input_dim)
        self.posterior_var_min = float(config.posterior_var_min)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.g0_enc_h0 = nn.Parameter(torch.zeros((2, 1, int(g0_encoder_dim))))
        self.g0_enc = BidirectionalClippedGRU(
            input_size=n_obs,
            hidden_size=int(g0_encoder_dim),
            clip_value=float(config.cell_clip),
        )
        self.g0_linear = nn.Linear(2 * int(g0_encoder_dim), 2 * self.gen_dim)
        init_linear_(self.g0_linear)

        self.causal_enc_h0 = nn.Parameter(torch.zeros((1, int(causal_encoder_dim))))
        self.causal_enc = ClippedGRU(
            input_size=n_obs,
            hidden_size=int(causal_encoder_dim),
            clip_value=float(config.cell_clip),
        )

        self.controller_h0 = nn.Parameter(torch.zeros((1, int(controller_dim))))
        self.controller = ClippedGRUCell(
            input_size=int(causal_encoder_dim) + self.fac_dim,
            hidden_size=int(controller_dim),
            clip_value=float(config.cell_clip),
        )
        self.input_linear = nn.Linear(int(controller_dim), 2 * self.input_dim)
        init_linear_(self.input_linear)

        self.generator = ClippedGRUCell(
            input_size=self.input_dim + incoming_dim,
            hidden_size=self.gen_dim,
            clip_value=float(config.cell_clip),
        )
        self.factor_linear = KernelNormalizedLinear(self.gen_dim, self.fac_dim, bias=False)
        init_linear_(self.factor_linear)

        self.readout_mean = nn.Linear(self.fac_dim, n_obs)
        init_linear_(self.readout_mean)
        if self.output_type == "gaussian":
            self.readout_logvar = nn.Linear(self.fac_dim, n_obs)
            init_linear_(self.readout_logvar)
        else:
            self.readout_logvar = None

    def _stable_logvar(self, raw_logvar: torch.Tensor) -> torch.Tensor:
        raw_logvar = torch.clamp(raw_logvar, min=-16.0, max=80.0)
        var = torch.exp(raw_logvar) + self.posterior_var_min
        return torch.log(var)

    def sample_diag_gaussian(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        sample: bool,
    ) -> torch.Tensor:
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode_g0(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = history.shape[0]
        h0 = torch.tile(self.g0_enc_h0, (1, batch_size, 1))
        history_drop = self.dropout(history)
        _, h_n = self.g0_enc(history_drop, h0)
        h_final = torch.cat([h_n[0], h_n[1]], dim=1)
        params = self.g0_linear(self.dropout(h_final))
        mean, raw_logvar = torch.chunk(params, 2, dim=1)
        logvar = self._stable_logvar(raw_logvar)
        return mean, logvar, h_final

    def encode_causal(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        h0 = torch.tile(self.causal_enc_h0, (batch_size, 1))
        obs_drop = self.dropout(obs)
        enc_seq, _ = self.causal_enc(obs_drop, h0)
        return enc_seq

    def init_controller_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.tile(self.controller_h0.to(device), (batch_size, 1))

    def factor_from_gen(self, gen_state: torch.Tensor) -> torch.Tensor:
        return self.factor_linear(self.dropout(gen_state))

    def infer_input(
        self,
        enc_t: torch.Tensor,
        prev_factor: torch.Tensor,
        prev_controller: torch.Tensor,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctrl_input = torch.cat([enc_t, prev_factor], dim=1)
        ctrl_state = self.controller(self.dropout(ctrl_input), prev_controller)
        params = self.input_linear(ctrl_state)
        mean, raw_logvar = torch.chunk(params, 2, dim=1)
        logvar = self._stable_logvar(raw_logvar)
        input_sample = self.sample_diag_gaussian(mean, logvar, sample=sample)
        return ctrl_state, mean, logvar, input_sample

    def generator_step(
        self,
        input_sample: torch.Tensor,
        incoming_messages: torch.Tensor,
        prev_gen: torch.Tensor,
    ) -> torch.Tensor:
        gen_input = torch.cat([input_sample, incoming_messages], dim=1)
        return self.generator(gen_input, prev_gen)

    def reconstruct(
        self,
        factors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean_or_lograte = self.readout_mean(factors)
        if self.output_type == "gaussian":
            assert self.readout_logvar is not None
            return mean_or_lograte, self.readout_logvar(factors)
        return mean_or_lograte, None

    def recurrent_weights(self) -> list[torch.Tensor]:
        return [
            self.g0_enc.fwd_gru.cell.weight_hh,
            self.g0_enc.bwd_gru.cell.weight_hh,
            self.causal_enc.cell.weight_hh,
            self.controller.weight_hh,
            self.generator.weight_hh,
        ]


class MRLFADS(nn.Module):
    def __init__(self, config: MRLFADSConfig):
        super().__init__()
        self.config = config
        self.n_regions = len(config.region_dims)
        self.edge_sources_by_target = {
            tgt: [src for src in range(self.n_regions) if src != tgt] for tgt in range(self.n_regions)
        }
        self.edge_keys: list[Edge] = [
            (src, tgt)
            for src in range(self.n_regions)
            for tgt in range(self.n_regions)
            if src != tgt
        ]
        incoming_dim = {
            tgt: len(self.edge_sources_by_target[tgt]) * int(config.message_dim) for tgt in range(self.n_regions)
        }
        self.regions = nn.ModuleList(
            [
                RegionModule(
                    i,
                    n_obs=int(config.region_dims[i]),
                    incoming_dim=incoming_dim[i],
                    config=config,
                    gen_dim=int(config.gen_dims[i]),
                    fac_dim=int(config.fac_dims[i]),
                    g0_encoder_dim=int(config.g0_encoder_dims[i]),
                    causal_encoder_dim=int(config.causal_encoder_dims[i]),
                    controller_dim=int(config.controller_dims[i]),
                )
                for i in range(self.n_regions)
            ]
        )
        rep_dims = {
            "rates": {i: int(config.region_dims[i]) for i in range(self.n_regions)},
            "factors": {i: int(config.fac_dims[i]) for i in range(self.n_regions)},
            "generator": {i: int(config.gen_dims[i]) for i in range(self.n_regions)},
        }
        self.message_mean = nn.ModuleDict()
        self.message_logvar = nn.ModuleDict()
        for src, tgt in self.edge_keys:
            key = self.edge_name(src, tgt)
            self.message_mean[key] = nn.Linear(rep_dims[config.message_from][src], int(config.message_dim))
            self.message_logvar[key] = nn.Linear(rep_dims[config.message_from][src], int(config.message_dim))
            init_linear_(self.message_mean[key])
            init_linear_(self.message_logvar[key])

    @staticmethod
    def edge_name(src: int, tgt: int) -> str:
        return f"{src}to{tgt}"

    def _ramp(self, epoch: int, start: int, increase: int) -> float:
        if increase < 0:
            raise ValueError("increase must be non-negative.")
        return float(np.clip((epoch + 1 - start) / (increase + 1), 0.0, 1.0))

    def regularization_weights(self, epoch: int) -> dict[str, float]:
        kl_floor = float(self.config.kl_floor)
        u_ramp = max(kl_floor, self._ramp(epoch, self.config.beta_u_start_epoch, self.config.beta_u_increase_epoch))
        m_ramp = max(kl_floor, self._ramp(epoch, self.config.beta_m_start_epoch, self.config.beta_m_increase_epoch))
        l2_ramp = self._ramp(epoch, self.config.l2_start_epoch, self.config.l2_increase_epoch)
        return {
            "beta_g0": float(self.config.beta_g0) * u_ramp,
            "beta_u": float(self.config.beta_u) * u_ramp,
            "beta_m": float(self.config.beta_m) * m_ramp,
            "l2_alpha": float(self.config.l2_alpha) * l2_ramp,
            "u_ramp": u_ramp,
            "m_ramp": m_ramp,
            "l2_ramp": l2_ramp,
        }

    def _stable_message_logvar(self, raw_logvar: torch.Tensor) -> torch.Tensor:
        raw_logvar = torch.clamp(raw_logvar, min=-16.0, max=80.0)
        var = torch.exp(raw_logvar) + float(self.config.posterior_var_min)
        return torch.log(var)

    def _sample_message(
        self, mean: torch.Tensor, logvar: torch.Tensor, sample: bool
    ) -> torch.Tensor:
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def _source_representation(
        self,
        region: RegionModule,
        gen_state: torch.Tensor,
        factor: torch.Tensor,
        mean_or_lograte: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.message_from == "generator":
            return gen_state
        if self.config.message_from == "factors":
            return factor
        if self.config.output_type == "poisson":
            return torch.exp(torch.clamp(mean_or_lograte, min=-7.0, max=7.0))
        return mean_or_lograte

    def _compute_messages(
        self,
        source_reps: list[torch.Tensor],
        sample_posteriors: bool,
    ) -> tuple[dict[Edge, torch.Tensor], dict[Edge, torch.Tensor], dict[Edge, torch.Tensor]]:
        means: dict[Edge, torch.Tensor] = {}
        logvars: dict[Edge, torch.Tensor] = {}
        samples: dict[Edge, torch.Tensor] = {}
        for src, tgt in self.edge_keys:
            name = self.edge_name(src, tgt)
            raw_mean = self.message_mean[name](source_reps[src])
            raw_logvar = self.message_logvar[name](source_reps[src])
            logvar = self._stable_message_logvar(raw_logvar)
            means[(src, tgt)] = raw_mean
            logvars[(src, tgt)] = logvar
            samples[(src, tgt)] = self._sample_message(raw_mean, logvar, sample_posteriors)
        return means, logvars, samples

    def _incoming_from_messages(
        self,
        message_samples: Mapping[Edge, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        incoming: list[torch.Tensor] = []
        for tgt in range(self.n_regions):
            sources = self.edge_sources_by_target[tgt]
            if not sources:
                incoming.append(torch.zeros(batch_size, 0, device=device))
                continue
            pieces = [message_samples[(src, tgt)] for src in sources]
            incoming.append(torch.cat(pieces, dim=1))
        return incoming

    def forward(
        self,
        history: list[torch.Tensor],
        obs: list[torch.Tensor],
        sample_posteriors: bool = True,
    ) -> dict[str, object]:
        if len(history) != self.n_regions or len(obs) != self.n_regions:
            raise ValueError(f"Expected {self.n_regions} regions, got {len(history)} history tensors and {len(obs)} obs tensors.")
        batch_size = obs[0].shape[0]
        device = obs[0].device
        num_steps = obs[0].shape[1]
        for tensor in obs:
            if tensor.shape[1] != num_steps:
                raise ValueError("All regions must share the same number of modeled timesteps.")

        g0_mean: dict[int, torch.Tensor] = {}
        g0_logvar: dict[int, torch.Tensor] = {}
        gen_prev: dict[int, torch.Tensor] = {}
        factor_prev: dict[int, torch.Tensor] = {}
        ctrl_prev: dict[int, torch.Tensor] = {}
        enc_seq: dict[int, torch.Tensor] = {}

        for i, region in enumerate(self.regions):
            mu0, lv0, _ = region.encode_g0(history[i])
            g0_mean[i] = mu0
            g0_logvar[i] = lv0
            gen_prev[i] = region.sample_diag_gaussian(mu0, lv0, sample=sample_posteriors)
            factor_prev[i] = region.factor_from_gen(gen_prev[i])
            ctrl_prev[i] = region.init_controller_state(batch_size, device)
            enc_seq[i] = region.encode_causal(obs[i])

        zero_messages = {(src, tgt): torch.zeros(batch_size, int(self.config.message_dim), device=device) for src, tgt in self.edge_keys}
        prev_message_samples = zero_messages

        msg_dim = int(self.config.message_dim)
        gen_states = {i: torch.zeros(batch_size, num_steps, r.gen_dim, device=device) for i, r in enumerate(self.regions)}
        factors = {i: torch.zeros(batch_size, num_steps, r.fac_dim, device=device) for i, r in enumerate(self.regions)}
        recon_mean_or_lograte = {i: torch.zeros(batch_size, num_steps, r.n_obs, device=device) for i, r in enumerate(self.regions)}
        recon_logvar: dict[int, torch.Tensor] = (
            {i: torch.zeros(batch_size, num_steps, r.n_obs, device=device) for i, r in enumerate(self.regions)}
            if self.config.output_type == "gaussian" else {}
        )
        input_means = {i: torch.zeros(batch_size, num_steps, self.config.input_dim, device=device) for i, r in enumerate(self.regions)}
        input_logvars = {i: torch.zeros(batch_size, num_steps, self.config.input_dim, device=device) for i, r in enumerate(self.regions)}
        input_samples = {i: torch.zeros(batch_size, num_steps, self.config.input_dim, device=device) for i, r in enumerate(self.regions)}
        controller_states = {i: torch.zeros(batch_size, num_steps, self.config.controller_dims[i], device=device) for i in range(self.n_regions)}
        message_means = {edge: torch.zeros(batch_size, num_steps, msg_dim, device=device) for edge in self.edge_keys}
        message_logvars = {edge: torch.zeros(batch_size, num_steps, msg_dim, device=device) for edge in self.edge_keys}
        message_samples = {edge: torch.zeros(batch_size, num_steps, msg_dim, device=device) for edge in self.edge_keys}

        for t in range(num_steps):
            ctrl_now: dict[int, torch.Tensor] = {}
            input_mean_now: dict[int, torch.Tensor] = {}
            input_logvar_now: dict[int, torch.Tensor] = {}
            input_sample_now: dict[int, torch.Tensor] = {}
            for i, region in enumerate(self.regions):
                ctrl_state, u_mean, u_logvar, u_sample = region.infer_input(
                    enc_seq[i][:, t], factor_prev[i], ctrl_prev[i], sample=sample_posteriors
                )
                ctrl_now[i] = ctrl_state
                input_mean_now[i] = u_mean
                input_logvar_now[i] = u_logvar
                input_sample_now[i] = u_sample

            if self.config.message_timing == "lagged":
                incoming = self._incoming_from_messages(prev_message_samples, batch_size, device)
                gen_now: dict[int, torch.Tensor] = {}
                factor_now: dict[int, torch.Tensor] = {}
                recon_main_now: dict[int, torch.Tensor] = {}
                recon_aux_now: dict[int, torch.Tensor | None] = {}
                source_rep_now: list[torch.Tensor] = []
                for i, region in enumerate(self.regions):
                    gen = region.generator_step(input_sample_now[i], incoming[i], gen_prev[i])
                    fac = region.factor_from_gen(gen)
                    main, aux = region.reconstruct(fac)
                    gen_now[i] = gen
                    factor_now[i] = fac
                    recon_main_now[i] = main
                    recon_aux_now[i] = aux
                    source_rep_now.append(self._source_representation(region, gen, fac, main))
                msg_mean_now, msg_logvar_now, msg_sample_now = self._compute_messages(
                    source_rep_now, sample_posteriors=sample_posteriors
                )
            else:
                incoming = self._incoming_from_messages(prev_message_samples, batch_size, device)
                msg_mean_now = zero_messages
                msg_logvar_now = zero_messages
                msg_sample_now = zero_messages
                gen_now = {i: gen_prev[i] for i in range(self.n_regions)}
                factor_now = {i: factor_prev[i] for i in range(self.n_regions)}
                recon_main_now = {i: torch.zeros(batch_size, self.regions[i].n_obs, device=device) for i in range(self.n_regions)}
                recon_aux_now: dict[int, torch.Tensor | None] = {i: None for i in range(self.n_regions)}
                for _ in range(int(self.config.fixed_point_iters)):
                    gen_cand: dict[int, torch.Tensor] = {}
                    fac_cand: dict[int, torch.Tensor] = {}
                    main_cand: dict[int, torch.Tensor] = {}
                    aux_cand: dict[int, torch.Tensor | None] = {}
                    source_reps: list[torch.Tensor] = []
                    for i, region in enumerate(self.regions):
                        gen = region.generator_step(input_sample_now[i], incoming[i], gen_prev[i])
                        fac = region.factor_from_gen(gen)
                        main, aux = region.reconstruct(fac)
                        gen_cand[i] = gen
                        fac_cand[i] = fac
                        main_cand[i] = main
                        aux_cand[i] = aux
                        source_reps.append(self._source_representation(region, gen, fac, main))
                    msg_mean_now, msg_logvar_now, msg_sample_now = self._compute_messages(
                        source_reps, sample_posteriors=sample_posteriors
                    )
                    incoming = self._incoming_from_messages(msg_sample_now, batch_size, device)
                    gen_now = gen_cand
                    factor_now = fac_cand
                    recon_main_now = main_cand
                    recon_aux_now = aux_cand

            for i in range(self.n_regions):
                gen_states[i][:, t] = gen_now[i]
                factors[i][:, t] = factor_now[i]
                recon_mean_or_lograte[i][:, t] = recon_main_now[i]
                if self.config.output_type == "gaussian":
                    assert recon_aux_now[i] is not None
                    recon_logvar[i][:, t] = recon_aux_now[i]
                input_means[i][:, t] = input_mean_now[i]
                input_logvars[i][:, t] = input_logvar_now[i]
                input_samples[i][:, t] = input_sample_now[i]
                controller_states[i][:, t] = ctrl_now[i]
                gen_prev[i] = gen_now[i]
                factor_prev[i] = factor_now[i]
                ctrl_prev[i] = ctrl_now[i]
            for edge in self.edge_keys:
                message_means[edge][:, t] = msg_mean_now[edge]
                message_logvars[edge][:, t] = msg_logvar_now[edge]
                message_samples[edge][:, t] = msg_sample_now[edge]
            prev_message_samples = msg_sample_now

        return {
            "g0_mean": g0_mean,
            "g0_logvar": g0_logvar,
            "gen_states": gen_states,
            "factors": factors,
            "recon_main": recon_mean_or_lograte,
            "recon_logvar": recon_logvar if self.config.output_type == "gaussian" else None,
            "input_means": input_means,
            "input_logvars": input_logvars,
            "input_samples": input_samples,
            "controller_states": controller_states,
            "message_means": message_means,
            "message_logvars": message_logvars,
            "message_samples": message_samples,
        }

    def compute_loss(
        self,
        obs: list[torch.Tensor],
        outputs: Mapping[str, object],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon_terms: list[torch.Tensor] = []
        for i in range(self.n_regions):
            recon_main = outputs["recon_main"][i]  # type: ignore[index]
            if self.config.output_type == "gaussian":
                recon_lv = outputs["recon_logvar"][i]  # type: ignore[index]
                recon_terms.append(gaussian_nll(obs[i], recon_main, recon_lv).mean())
            else:
                recon_terms.append(poisson_nll(obs[i], recon_main).mean())
        recon = torch.stack(recon_terms).mean()

        kl_g0_terms = []
        for i in range(self.n_regions):
            mu = outputs["g0_mean"][i]  # type: ignore[index]
            lv = outputs["g0_logvar"][i]  # type: ignore[index]
            kl_g0_terms.append(diag_gaussian_kl(mu, lv, prior_var=self.config.prior_variance).mean())
        kl_g0 = torch.stack(kl_g0_terms).mean()

        kl_u_terms = []
        for i in range(self.n_regions):
            mu = outputs["input_means"][i]  # type: ignore[index]
            lv = outputs["input_logvars"][i]  # type: ignore[index]
            kl_u_terms.append(diag_gaussian_kl(mu, lv, prior_var=self.config.prior_variance).mean())
        kl_u = torch.stack(kl_u_terms).mean()

        kl_m_terms = []
        for edge in self.edge_keys:
            mu = outputs["message_means"][edge]  # type: ignore[index]
            lv = outputs["message_logvars"][edge]  # type: ignore[index]
            kl_m_terms.append(diag_gaussian_kl(mu, lv, prior_var=self.config.prior_variance).mean())
        kl_m = torch.stack(kl_m_terms).mean() if kl_m_terms else torch.tensor(0.0, device=obs[0].device)

        reg = self.regularization_weights(epoch)
        l2_penalty = self.recurrent_l2()
        total = self.config.loss_scale * (
            recon
            + reg["beta_g0"] * kl_g0
            + reg["beta_u"] * kl_u
            + reg["beta_m"] * kl_m
            + reg["l2_alpha"] * l2_penalty
        )
        metrics = {
            "loss": float(total.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "kl_g0": float(kl_g0.detach().cpu()),
            "kl_u": float(kl_u.detach().cpu()),
            "kl_m": float(kl_m.detach().cpu()),
            "l2": float(l2_penalty.detach().cpu()),
            **reg,
        }
        return total, metrics

    def recurrent_l2(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        size = 0
        for region in self.regions:
            for weight in region.recurrent_weights():
                penalty = penalty + 0.5 * torch.norm(weight, 2) ** 2
                size += weight.numel()
        if size == 0:
            return penalty
        return penalty / float(size)

    @torch.no_grad()
    def posterior_means(self, history: list[torch.Tensor], obs: list[torch.Tensor]) -> dict[str, object]:
        return self.forward(history=history, obs=obs, sample_posteriors=False)
