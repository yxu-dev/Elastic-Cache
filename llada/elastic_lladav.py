"""Elastic KV-cache controller for the LLaDA-V eager-attention backend.

The generation sampler stays in LLaDA-V.  This module owns the Elastic cache,
tracked-token attention monitor, and layer-aware refresh boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


DecisionObserver = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class ElasticLLaDAVConfig:
    gamma: float = 0.9
    track_num: int = 1
    # The controlled LLaDA-V port queries every still-masked response token.
    # Restricting queries to the active block belongs to the separate
    # paper-style replication protocol because it can change native logits.
    block_caching: bool = False
    always_refresh: bool = False

    def __post_init__(self) -> None:
        if not -1.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between -1 and 1")
        if self.track_num < 1:
            raise ValueError("track_num must be positive")


class ElasticLLaDAVController:
    """Generation-local implementation of the official Elastic policy.

    The observer is deliberately one-way: its return value is ignored, so
    tracing cannot alter a cache decision.
    """

    def __init__(
        self,
        config: ElasticLLaDAVConfig,
        *,
        observer: DecisionObserver | None = None,
    ) -> None:
        self.config = config
        self.observer = observer
        self.reset()

    def reset(self) -> None:
        self.step = -1
        self.block = -1
        self.sequence_length = 0
        self.refresh_start_layer = 0
        self.query_positions: Any = None
        self.masked_positions: Any = None
        self.all_masked_positions: Any = None
        self.track_positions: Any = None
        self._next_track_positions: list[Any] = []
        self.multimodal_layout: dict[str, Any] | None = None
        self._layer_runtime: dict[int, dict[str, Any]] = {}
        self._step_record_indices: list[int] = []
        self.layer_cache: dict[int, dict[str, Any]] = {}
        self.decisions: list[dict[str, Any]] = []
        self._previous_response_top1: dict[int, int] = {}

    @staticmethod
    def _stable_unique(values: Any) -> Any:
        import torch

        if values.numel() == 0:
            return values
        seen: set[int] = set()
        ordered = []
        for value in values.detach().cpu().tolist():
            integer = int(value)
            if integer not in seen:
                seen.add(integer)
                ordered.append(integer)
        return torch.tensor(ordered, dtype=torch.long, device=values.device)

    def begin_step(
        self,
        *,
        step: int,
        block: int,
        sequence_length: int,
        masked_positions: Any,
        active_masked_positions: Any,
        newly_decoded_positions: Any,
        multimodal_layout: dict[str, Any] | None = None,
    ) -> Any:
        import torch

        self.step = int(step)
        self.block = int(block)
        self.sequence_length = int(sequence_length)
        self.multimodal_layout = multimodal_layout
        self.all_masked_positions = masked_positions
        self.masked_positions = (
            active_masked_positions if self.config.block_caching else masked_positions
        )
        if self.track_positions is None:
            self.track_positions = masked_positions[:0]
        self._next_track_positions = []
        self._layer_runtime = {}
        self._step_record_indices = []

        first_step = not self.layer_cache
        if first_step or self.config.always_refresh:
            self.refresh_start_layer = 0
            self.query_positions = torch.arange(
                self.sequence_length,
                dtype=torch.long,
                device=masked_positions.device,
            )
        else:
            self.refresh_start_layer = len(self.layer_cache)
            self.query_positions = self._stable_unique(
                torch.cat(
                    (
                        self.track_positions,
                        newly_decoded_positions,
                        self.masked_positions,
                    )
                )
            )
        return self.query_positions

    def restore_sequence(self, values: Any, *, fill_value: float = 0.0) -> Any:
        if values.shape[1] == self.sequence_length:
            return values
        shape = (values.shape[0], self.sequence_length, *values.shape[2:])
        restored = values.new_full(shape, fill_value)
        restored.index_copy_(1, self.query_positions, values)
        return restored

    @property
    def uses_partial_query(self) -> bool:
        return int(self.query_positions.numel()) != self.sequence_length

    def position_ids(self, *, layer: int, device: Any) -> Any:
        import torch

        positions = (
            torch.arange(self.sequence_length, device=device)
            if int(layer) >= self.refresh_start_layer
            else self.query_positions
        )
        return positions.unsqueeze(0)

    def before_layer(self, *, layer: int, hidden_states: Any) -> Any:
        layer = int(layer)
        previous = self.layer_cache.get(layer)
        refreshed = layer >= self.refresh_start_layer or previous is None
        if previous is None or hidden_states.shape[1] == self.sequence_length:
            full_hidden = hidden_states
        else:
            full_hidden = previous["hidden"].clone()
            full_hidden.index_copy_(1, self.query_positions, hidden_states)
        self._layer_runtime[layer] = {
            "previous": previous,
            "refreshed": refreshed,
            "hidden": full_hidden.detach().clone(),
        }
        return full_hidden if refreshed else hidden_states

    def full_hidden_for_audit(self, *, layer: int) -> Any:
        """Expose the materialized layer input read-only to an isolated auditor."""

        return self._layer_runtime[int(layer)]["hidden"]

    def materialize_kv(
        self,
        *,
        layer: int,
        query: Any,
        key: Any,
        value: Any,
    ) -> tuple[Any, Any, Any]:
        layer = int(layer)
        runtime = self._layer_runtime[layer]
        previous = runtime["previous"]
        if runtime["refreshed"]:
            full_query, full_key, full_value = query, key, value
            runtime["past_query"] = None
            runtime["past_key"] = None
        else:
            full_query = previous["query"].clone()
            full_key = previous["key"].clone()
            full_value = previous["value"].clone()
            runtime["past_query"] = previous["query"].index_select(
                2, self.track_positions
            ).clone()
            runtime["past_key"] = full_key.clone()
            full_query.index_copy_(2, self.query_positions, query)
            full_key.index_copy_(2, self.query_positions, key)
            full_value.index_copy_(2, self.query_positions, value)
        runtime["query"] = full_query
        runtime["key"] = full_key
        runtime["value"] = full_value
        return query, full_key, full_value

    def _query_rows(self, absolute_positions: Any, *, refreshed: bool) -> Any:
        import torch

        if refreshed:
            return absolute_positions
        lookup = {int(value): index for index, value in enumerate(self.query_positions.tolist())}
        rows = [lookup[int(value)] for value in absolute_positions.tolist()]
        return torch.tensor(rows, dtype=torch.long, device=absolute_positions.device)

    def observe_attention(self, *, layer: int, attention_weights: Any) -> None:
        import torch
        import torch.nn.functional as functional

        layer = int(layer)
        runtime = self._layer_runtime[layer]
        refreshed = bool(runtime["refreshed"])
        previous = runtime["previous"]
        monitor = None
        monitor_input_positions = self.track_positions.detach().cpu().tolist()
        monitor_triggered_here = False
        if not refreshed and previous is not None and self.track_positions.numel() > 0:
            past_q = runtime["past_query"]
            past_k = runtime["past_key"]
            if past_k.shape[1] != past_q.shape[1]:
                if past_q.shape[1] % past_k.shape[1] != 0:
                    raise ValueError("query and key head counts are incompatible")
                past_k = past_k.repeat_interleave(
                    past_q.shape[1] // past_k.shape[1], dim=1
                )
            past_attention = torch.softmax(
                torch.matmul(past_q, past_k.transpose(-2, -1))
                / math.sqrt(past_q.shape[-1]),
                dim=-1,
            )
            track_rows = self._query_rows(self.track_positions, refreshed=False)
            current_attention = attention_weights.index_select(2, track_rows)
            monitor = float(
                functional.cosine_similarity(
                    past_attention, current_attention, dim=1
                ).mean().item()
            )
            if monitor < self.config.gamma:
                monitor_triggered_here = True
                self.refresh_start_layer = min(self.refresh_start_layer, layer + 1)

        masked_rows = self._query_rows(self.masked_positions, refreshed=refreshed)
        masked_attention = attention_weights.index_select(2, masked_rows)
        importance = masked_attention.sum(dim=(0, 1, 2))
        importance.index_fill_(0, self.all_masked_positions, 0.0)
        count = min(self.config.track_num, int(importance.numel()))
        tracked = importance.topk(k=count, largest=True).indices
        self._next_track_positions.append(tracked)

        last_refresh_step = (
            self.step
            if refreshed or previous is None
            else int(previous["last_refresh_step"])
        )
        drift, visual_attention = self._compact_drift(
            runtime["hidden"], runtime["key"], attention_weights, previous,
            refreshed=refreshed,
        )
        self.layer_cache[layer] = {
            "hidden": runtime["hidden"],
            "query": runtime["query"].detach().clone(),
            "key": runtime["key"].detach().clone(),
            "value": runtime["value"].detach().clone(),
            "last_refresh_step": last_refresh_step,
            "visual_attention": visual_attention,
        }
        record = {
            "step": self.step,
            "block": self.block,
            "layer": layer,
            "elastic_monitor_value": monitor,
            "monitor_input_positions": monitor_input_positions,
            "next_tracked_positions": tracked.detach().cpu().tolist(),
            "gamma": self.config.gamma,
            "monitor_margin_to_gamma": (
                None if monitor is None else monitor - self.config.gamma
            ),
            "confidence_threshold": None,
            "tracked_token": tracked.detach().cpu().tolist(),
            "decision_before_layer": "recompute" if refreshed else "reuse",
            "monitor_triggered_here": monitor_triggered_here,
            "trigger_layer": layer if monitor_triggered_here else None,
            "recompute_start_layer": int(self.refresh_start_layer),
            "step_final_refresh_start_layer": None,
            "refresh": refreshed,
            "refresh_start_layer": int(self.refresh_start_layer),
            "recomputed": refreshed,
            "reused": not refreshed,
            "cache_age": self.step - last_refresh_step,
            "cache_generation": last_refresh_step,
            "active_window": self.masked_positions.detach().cpu().tolist(),
            "queried_masked_positions": self.masked_positions.detach().cpu().tolist(),
            "queried_masked_count": int(self.masked_positions.numel()),
            "remaining_masked_count": int(self.all_masked_positions.numel()),
            "token_role_summary": self._token_role_summary(),
            "token_role_metadata": self._token_role_metadata(),
            "signal_observability": {
                "elastic_monitor_value": "online_observable",
                "response_entropy": "online_observable",
                "response_top1_flip_rate": "online_observable",
                "response_margin": "online_observable",
                "visual_attention_drift": "online_observable",
                "visual_attention_mass": "online_observable",
                "visual_attention_redistribution": "online_observable",
                "visual_k_drift": "cached_state_comparison",
                "visual_hidden_drift": "cached_state_comparison",
            },
            **drift,
        }
        self.decisions.append(record)
        self._step_record_indices.append(len(self.decisions) - 1)
        del self._layer_runtime[layer]

    def observe_logits(self, logits: Any) -> None:
        import torch

        active = logits.index_select(1, self.masked_positions).float()
        if active.numel() == 0:
            entropy = None
            top1_flip_rate = None
            response_margin = None
        else:
            log_normalizer = torch.logsumexp(active, dim=-1)
            probabilities = torch.softmax(active, dim=-1)
            weighted = torch.where(
                torch.isfinite(active), probabilities * active, torch.zeros_like(active)
            )
            entropy = float((log_normalizer - weighted.sum(dim=-1)).mean().item())
            top2 = torch.topk(active, k=2, dim=-1)
            current_top1 = top2.indices[..., 0].reshape(-1).detach().cpu().tolist()
            positions = self.masked_positions.detach().cpu().tolist()
            comparable = [
                int(current) != self._previous_response_top1[int(position)]
                for position, current in zip(positions, current_top1)
                if int(position) in self._previous_response_top1
            ]
            top1_flip_rate = (
                sum(comparable) / len(comparable) if comparable else None
            )
            response_margin = float(
                (top2.values[..., 0] - top2.values[..., 1]).mean().item()
            )
            self._previous_response_top1.update(
                {
                    int(position): int(token)
                    for position, token in zip(positions, current_top1)
                }
            )
        for index in self._step_record_indices:
            self.decisions[index]["response_entropy"] = entropy
            self.decisions[index]["response_top1_flip_rate"] = top1_flip_rate
            self.decisions[index]["response_margin"] = response_margin

    def end_step(self) -> None:
        import torch

        for index in self._step_record_indices:
            self.decisions[index]["step_final_refresh_start_layer"] = int(
                self.refresh_start_layer
            )
        if self._next_track_positions:
            self.track_positions = self._stable_unique(
                torch.cat(self._next_track_positions)
            )
        if self.observer is not None:
            for index in self._step_record_indices:
                self.observer(dict(self.decisions[index]))

    def _compact_drift(
        self,
        hidden: Any,
        key: Any,
        attention_weights: Any,
        previous: dict[str, Any] | None,
        *,
        refreshed: bool,
    ) -> tuple[dict[str, float | None], Any]:
        import torch.nn.functional as functional

        if self.multimodal_layout is None:
            return ({
                "visual_k_drift": None,
                "visual_hidden_drift": None,
                "visual_attention_drift": None,
                "visual_attention_mass": None,
                "visual_attention_redistribution": None,
            }, None)
        visual = [
            index
            for start, end in self.multimodal_layout.get("visual_spans", [])
            for index in range(int(start), int(end))
        ]
        if not visual:
            return ({
                "visual_k_drift": None,
                "visual_hidden_drift": None,
                "visual_attention_drift": None,
                "visual_attention_mass": None,
                "visual_attention_redistribution": None,
            }, None)
        indices = self.query_positions.new_tensor(visual)
        masked_rows = self._query_rows(self.masked_positions, refreshed=refreshed)
        visual_attention_rows = (
            attention_weights.index_select(2, masked_rows)
            .index_select(3, indices)
            .float()
        )
        visual_attention_mass = float(
            visual_attention_rows.sum(dim=-1).mean().item()
        )
        visual_attention = visual_attention_rows.mean(dim=(0, 1, 2))
        visual_attention = visual_attention / visual_attention.sum().clamp_min(1e-12)
        if previous is None:
            return ({
                "visual_k_drift": None,
                "visual_hidden_drift": None,
                "visual_attention_drift": None,
                "visual_attention_mass": visual_attention_mass,
                "visual_attention_redistribution": None,
            }, visual_attention.detach().clone())
        current_hidden = hidden.index_select(1, indices)
        previous_hidden = previous["hidden"].index_select(1, indices)
        current_key = key.index_select(2, indices)
        previous_key = previous["key"].index_select(2, indices)
        previous_attention = previous.get("visual_attention")
        visual_attention_drift = None
        visual_attention_redistribution = None
        if previous_attention is not None:
            midpoint = 0.5 * (visual_attention + previous_attention)
            visual_attention_drift = float(
                0.5
                * (
                    functional.kl_div(
                        midpoint.clamp_min(1e-12).log(),
                        visual_attention,
                        reduction="sum",
                    )
                    + functional.kl_div(
                        midpoint.clamp_min(1e-12).log(),
                        previous_attention,
                        reduction="sum",
                    )
                ).item()
            )
            visual_attention_redistribution = float(
                (0.5 * (visual_attention - previous_attention).abs().sum()).item()
            )
        return ({
            "visual_k_drift": float(
                (1.0 - functional.cosine_similarity(
                    current_key.float(), previous_key.float(), dim=-1
                )).mean().item()
            ),
            "visual_hidden_drift": float(
                (1.0 - functional.cosine_similarity(
                    current_hidden.float(), previous_hidden.float(), dim=-1
                )).mean().item()
            ),
            "visual_attention_drift": visual_attention_drift,
            "visual_attention_mass": visual_attention_mass,
            "visual_attention_redistribution": visual_attention_redistribution,
        }, visual_attention.detach().clone())

    def _token_role_metadata(self) -> dict[str, dict[str, int]]:
        if self.multimodal_layout is None:
            return {"sequence": {}, "queried": {}}
        token_types = list(self.multimodal_layout.get("token_types", []))
        remaining_masked = {
            int(value) for value in self.all_masked_positions.detach().cpu().tolist()
        }
        queried = {
            int(value) for value in self.query_positions.detach().cpu().tolist()
        }

        def role(index: int, token_type: str) -> str:
            if token_type == "visual":
                return "visual"
            if token_type == "prompt_text":
                return "prompt"
            if token_type.startswith("generated"):
                return (
                    "response_masked"
                    if index in remaining_masked
                    else "response_committed"
                )
            return "special"

        names = (
            "visual",
            "prompt",
            "response_committed",
            "response_masked",
            "special",
        )
        sequence_counts = {name: 0 for name in names}
        queried_counts = {name: 0 for name in names}
        for index, token_type in enumerate(token_types):
            name = role(index, token_type)
            sequence_counts[name] += 1
            if index in queried:
                queried_counts[name] += 1
        return {"sequence": sequence_counts, "queried": queried_counts}

    def _token_role_summary(self) -> dict[str, int]:
        if self.multimodal_layout is None:
            return {}
        token_types = list(self.multimodal_layout.get("token_types", []))
        queried = {int(value) for value in self.query_positions.detach().cpu().tolist()}
        counts = {"visual": 0, "prompt": 0, "response": 0, "special": 0}
        for index, role in enumerate(token_types):
            if index in queried:
                continue
            if role == "visual":
                counts["visual"] += 1
            elif role == "prompt_text":
                counts["prompt"] += 1
            elif role.startswith("generated"):
                counts["response"] += 1
            else:
                counts["special"] += 1
        return counts
