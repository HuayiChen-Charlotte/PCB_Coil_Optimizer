"""Math-structured search, caching, ranking, and verification."""

from __future__ import annotations

import math
import heapq
import time
from collections import defaultdict
from typing import Any

from .models import ModelContext, SearchSpace, compact_number
from .physics import (
    H_TO_UH,
    UH_TO_H,
    append_reason,
    build_coil,
    finite_positive,
    load_resistance_ohm,
    mutual_between_coils_h,
)


def context_label(ctx: ModelContext) -> str:
    return (
        f"f={ctx.frequency_hz / 1e6:.6g}MHz "
        f"h={ctx.layer_spacing_mm:.6g}mm "
        f"T={ctx.temperature_c:.6g}C "
        f"Cu={ctx.copper_thickness_oz:.3g}oz"
    )


class ProgressBar:
    """Small stdout progress bar for long PowerShell runs."""

    def __init__(self, label: str, total: int, *, enabled: bool, interval_seconds: float) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.enabled = enabled
        self.interval_seconds = max(float(interval_seconds), 0.1)
        self.started = time.monotonic()
        self.last_print = 0.0
        self.last_completed = -1
        if self.enabled:
            self.update(0, force=True)

    def update(self, completed: int, *, suffix: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        completed = min(max(int(completed), 0), self.total)
        if force and completed == self.last_completed:
            return
        now = time.monotonic()
        if not force and completed < self.total and now - self.last_print < self.interval_seconds:
            return
        self.last_print = now
        self.last_completed = completed
        fraction = completed / self.total
        width = 24
        filled = min(width, int(round(fraction * width)))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = now - self.started
        suffix_text = f" | {suffix}" if suffix else ""
        print(
            f"[{bar}] {fraction * 100:6.2f}% {completed}/{self.total} "
            f"{self.label} | elapsed={elapsed:.1f}s{suffix_text}",
            flush=True,
        )


def build_coil_library(
    side: str,
    turns_values: list[int],
    dout_values: list[float],
    width_values: list[float],
    spacing_values: list[float],
    ctx: ModelContext,
    constraints: dict[str, Any],
    show_progress: bool = False,
    progress_interval_seconds: float = 5.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, int, int, int], int]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    lookup: dict[tuple[int, int, int, int], int] = {}
    total = len(turns_values) * len(dout_values) * len(width_values) * len(spacing_values)
    progress = ProgressBar(
        f"{side.upper()} coil library {context_label(ctx)}",
        total,
        enabled=show_progress,
        interval_seconds=progress_interval_seconds,
    )
    completed = 0
    for n_index, turns in enumerate(turns_values):
        for d_index, dout_mm in enumerate(dout_values):
            for w_index, width_mm in enumerate(width_values):
                for s_index, spacing_mm in enumerate(spacing_values):
                    coil = build_coil(
                        side=side,
                        index_tuple=(n_index, d_index, w_index, s_index),
                        turns=turns,
                        dout_mm=dout_mm,
                        width_mm=width_mm,
                        spacing_mm=spacing_mm,
                        ctx=ctx,
                        constraints=constraints,
                    )
                    if coil["status"] == "ok":
                        lookup[coil["index_tuple"]] = len(valid)
                        valid.append(coil)
                    else:
                        rejected.append(coil)
                    completed += 1
                    progress.update(
                        completed,
                        suffix=f"valid={len(valid)} rejected={len(rejected)}",
                    )
    progress.update(completed, suffix=f"valid={len(valid)} rejected={len(rejected)}", force=True)
    return valid, rejected, lookup


def norm_constraint_reject(tx: dict[str, Any], rx: dict[str, Any], distance_mm: float, constraints: dict[str, Any]) -> str:
    norm_dcoil_percent = distance_mm / math.sqrt(tx["Dout_mm"] * rx["Dout_mm"]) * 100.0
    reject = ""
    norm_min = constraints.get("norm_Dcoil_percent_min")
    norm_max = constraints.get("norm_Dcoil_percent_max")
    if norm_min is not None and norm_dcoil_percent < float(norm_min):
        reject = append_reason(reject, "norm_Dcoil_below_min")
    if norm_max is not None and norm_dcoil_percent > float(norm_max):
        reject = append_reason(reject, "norm_Dcoil_above_max")
    return reject


def pair_metrics(
    tx: dict[str, Any],
    rx: dict[str, Any],
    distance_mm: float,
    distance_index: int,
    ctx: ModelContext,
    constraints: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    norm_dcoil_percent = distance_mm / math.sqrt(tx["Dout_mm"] * rx["Dout_mm"]) * 100.0
    reject = norm_constraint_reject(tx, rx, distance_mm, constraints)
    base = {
        "pair_key": f"{tx['coil_id']}|{rx['coil_id']}|d{compact_number(distance_mm)}",
        "tx_coil_id": tx["coil_id"],
        "rx_coil_id": rx["coil_id"],
        "tx_index_tuple": tx["index_tuple"],
        "rx_index_tuple": rx["index_tuple"],
        "distance_index": distance_index,
        "frequency_MHz": ctx.frequency_hz / 1e6,
        "layer_spacing_mm": ctx.layer_spacing_mm,
        "temperature_C": ctx.temperature_c,
        "copper_thickness_um": ctx.copper_thickness_um,
        "copper_thickness_oz": ctx.copper_thickness_oz,
        "distance_mm": distance_mm,
        "Norm_Dcoil_percent": norm_dcoil_percent,
        "tx_layers": tx["layers"],
        "tx_connection": tx["connection"],
        "tx_N": tx["N"],
        "tx_Dout_mm": tx["Dout_mm"],
        "tx_Din_mm": tx["Din_mm"],
        "tx_Din_webench_fill_mm": tx["Din_webench_fill_mm"],
        "tx_fill_factor_webench": tx["fill_factor_webench"],
        "tx_trace_width_mm": tx["trace_width_mm"],
        "tx_spacing_mm": tx["spacing_mm"],
        "tx_L_total_uH": tx["L_total_uH"],
        "tx_Rdc_ohm": tx["Rdc_ohm"],
        "tx_Rskin_ohm": tx["Rskin_ohm"],
        "tx_qian_phi": tx["qian_phi"],
        "tx_rac_model_version": tx["rac_model_version"],
        "tx_rac_two_layer_extension_version": tx["rac_two_layer_extension_version"],
        "tx_rac_model_domain": tx["rac_model_domain"],
        "tx_Rprox_qian_same_layer_ohm": tx["Rprox_qian_same_layer_ohm"],
        "tx_Rac_qian_same_layer_ohm": tx["Rac_qian_same_layer_ohm"],
        "tx_Q_qian_same_layer": tx["Q_qian_same_layer"],
        "tx_Rprox_qian_other_layer_only_ohm": tx["Rprox_qian_other_layer_only_ohm"],
        "tx_Rprox_qian_cross_term_ohm": tx["Rprox_qian_cross_term_ohm"],
        "tx_Rprox_qian_two_layer_extension_ohm": tx[
            "Rprox_qian_two_layer_extension_ohm"
        ],
        "tx_Rac_qian_two_layer_extension_ohm": tx[
            "Rac_qian_two_layer_extension_ohm"
        ],
        "tx_Q_qian_two_layer_extension": tx["Q_qian_two_layer_extension"],
        "tx_Rprox_legacy_kuhn_ohm": tx["Rprox_legacy_kuhn_ohm"],
        "tx_Rac_legacy_kuhn_ohm": tx["Rac_legacy_kuhn_ohm"],
        "tx_Q_legacy_kuhn": tx["Q_legacy_kuhn"],
        "tx_Rac_model_ohm": tx["Rac_model_ohm"],
        "tx_Q_model": tx["Q_model"],
        "tx_Rprox_ohm": tx["Rprox_ohm"],
        "tx_Rac_theorybook_ohm": tx["Rac_theorybook_ohm"],
        "tx_Q_theorybook": tx["Q_theorybook"],
        "tx_C_total_pF": tx["C_total_pF"],
        "tx_active_cpar_model": tx["active_cpar_model"],
        "tx_Cpar_active_pF": tx["Cpar_active_pF"],
        "tx_C_sensor_activeCpar_pF": tx["C_sensor_activeCpar_pF"],
        "tx_fsrf_activeCpar_MHz": tx["fsrf_activeCpar_MHz"],
        "tx_Cpar_fit_pF": tx["Cpar_fit_pF"],
        "tx_C_sensor_fitCpar_pF": tx["C_sensor_fitCpar_pF"],
        "tx_fsrf_fitCpar_MHz": tx["fsrf_fitCpar_MHz"],
        "tx_Cpar_webench_pF": tx["Cpar_webench_pF"],
        "tx_C_sensor_webenchCpar_pF": tx["C_sensor_webenchCpar_pF"],
        "tx_Cpar_book_pF": tx["Cpar_book_pF"],
        "tx_C_sensor_bookCpar_pF": tx["C_sensor_bookCpar_pF"],
        "tx_fsrf_webenchCpar_MHz": tx["fsrf_webenchCpar_MHz"],
        "tx_fsrf_bookCpar_MHz": tx["fsrf_bookCpar_MHz"],
        "rx_layers": rx["layers"],
        "rx_connection": rx["connection"],
        "rx_N": rx["N"],
        "rx_Dout_mm": rx["Dout_mm"],
        "rx_Din_mm": rx["Din_mm"],
        "rx_Din_webench_fill_mm": rx["Din_webench_fill_mm"],
        "rx_fill_factor_webench": rx["fill_factor_webench"],
        "rx_trace_width_mm": rx["trace_width_mm"],
        "rx_spacing_mm": rx["spacing_mm"],
        "rx_L_total_uH": rx["L_total_uH"],
        "rx_Rdc_ohm": rx["Rdc_ohm"],
        "rx_Rskin_ohm": rx["Rskin_ohm"],
        "rx_qian_phi": rx["qian_phi"],
        "rx_rac_model_version": rx["rac_model_version"],
        "rx_rac_two_layer_extension_version": rx["rac_two_layer_extension_version"],
        "rx_rac_model_domain": rx["rac_model_domain"],
        "rx_Rprox_qian_same_layer_ohm": rx["Rprox_qian_same_layer_ohm"],
        "rx_Rac_qian_same_layer_ohm": rx["Rac_qian_same_layer_ohm"],
        "rx_Q_qian_same_layer": rx["Q_qian_same_layer"],
        "rx_Rprox_qian_other_layer_only_ohm": rx["Rprox_qian_other_layer_only_ohm"],
        "rx_Rprox_qian_cross_term_ohm": rx["Rprox_qian_cross_term_ohm"],
        "rx_Rprox_qian_two_layer_extension_ohm": rx[
            "Rprox_qian_two_layer_extension_ohm"
        ],
        "rx_Rac_qian_two_layer_extension_ohm": rx[
            "Rac_qian_two_layer_extension_ohm"
        ],
        "rx_Q_qian_two_layer_extension": rx["Q_qian_two_layer_extension"],
        "rx_Rprox_legacy_kuhn_ohm": rx["Rprox_legacy_kuhn_ohm"],
        "rx_Rac_legacy_kuhn_ohm": rx["Rac_legacy_kuhn_ohm"],
        "rx_Q_legacy_kuhn": rx["Q_legacy_kuhn"],
        "rx_Rac_model_ohm": rx["Rac_model_ohm"],
        "rx_Q_model": rx["Q_model"],
        "rx_Rprox_ohm": rx["Rprox_ohm"],
        "rx_Rac_theorybook_ohm": rx["Rac_theorybook_ohm"],
        "rx_Q_theorybook": rx["Q_theorybook"],
        "rx_C_total_pF": rx["C_total_pF"],
        "rx_active_cpar_model": rx["active_cpar_model"],
        "rx_Cpar_active_pF": rx["Cpar_active_pF"],
        "rx_C_sensor_activeCpar_pF": rx["C_sensor_activeCpar_pF"],
        "rx_fsrf_activeCpar_MHz": rx["fsrf_activeCpar_MHz"],
        "rx_Cpar_fit_pF": rx["Cpar_fit_pF"],
        "rx_C_sensor_fitCpar_pF": rx["C_sensor_fitCpar_pF"],
        "rx_fsrf_fitCpar_MHz": rx["fsrf_fitCpar_MHz"],
        "rx_Cpar_webench_pF": rx["Cpar_webench_pF"],
        "rx_C_sensor_webenchCpar_pF": rx["C_sensor_webenchCpar_pF"],
        "rx_Cpar_book_pF": rx["Cpar_book_pF"],
        "rx_C_sensor_bookCpar_pF": rx["C_sensor_bookCpar_pF"],
        "rx_fsrf_webenchCpar_MHz": rx["fsrf_webenchCpar_MHz"],
        "rx_fsrf_bookCpar_MHz": rx["fsrf_bookCpar_MHz"],
        "status": "ok" if not reject else "rejected",
        "reject_reason": reject,
        "fasthenry_status": "not_requested",
    }
    if reject:
        return base, reject

    mutual_h = mutual_between_coils_h(tx, rx, distance_mm, ctx)
    ltx_h = tx["L_total_uH"] * UH_TO_H
    lrx_h = rx["L_total_uH"] * UH_TO_H
    k = mutual_h / math.sqrt(ltx_h * lrx_h)
    k_min = float(constraints.get("k_min", 0.0))
    k_max = float(constraints.get("k_max", 1.0))
    if not (k_min <= k < k_max):
        reject = append_reason(reject, "k_out_of_range")
    rtx = tx["Rac_model_ohm"]
    rrx = rx["Rac_model_ohm"]
    qtx = tx["Q_model"]
    qrx = rx["Q_model"]
    chi = k * k * qtx * qrx
    chi_math = (ctx.omega * mutual_h) ** 2 / (rtx * rrx)
    eta_max = chi / (1.0 + math.sqrt(1.0 + chi)) ** 2
    rload_opt = rrx * math.sqrt(1.0 + chi)
    row = {
        **base,
        "M_uH": mutual_h * H_TO_UH,
        "k": k,
        "chi_k2Q1Q2": chi,
        "chi_math": chi_math,
        "chi_identity_error": abs(chi - chi_math),
        "eta_max": eta_max,
        "Rload_opt_ohm": rload_opt,
        "Q_min": min(qtx, qrx),
        "Q_geom": math.sqrt(qtx * qrx),
        "status": "ok" if not reject else "rejected",
        "reject_reason": reject,
    }
    if not (0.0 <= eta_max <= 1.0):
        row["status"] = "rejected"
        row["reject_reason"] = append_reason(row["reject_reason"], "eta_max_out_of_range")
    return row, row["reject_reason"] or None


def scenario_id(pair: dict[str, Any], load_voltage_v: float, load_power_mw: float) -> str:
    return (
        f"f{compact_number(pair['frequency_MHz'])}MHz_"
        f"h{compact_number(pair['layer_spacing_mm'])}mm_"
        f"T{compact_number(pair['temperature_C'])}C_"
        f"rxD{compact_number(pair['rx_Dout_mm'])}mm_"
        f"d{compact_number(pair['distance_mm'])}mm_"
        f"v{compact_number(load_voltage_v)}V_"
        f"p{compact_number(load_power_mw)}mW_"
        f"cu{compact_number(pair['copper_thickness_oz'])}oz"
    )


def loaded_efficiency_value(pair: dict[str, Any], load_voltage_v: float, load_power_mw: float, ctx: ModelContext) -> float:
    load = load_resistance_ohm(load_voltage_v, load_power_mw)
    mutual_h = pair["M_uH"] * UH_TO_H
    rtx = pair["tx_Rac_model_ohm"]
    rrx = pair["rx_Rac_model_ohm"]
    a = (ctx.omega * mutual_h) ** 2
    denominator = rtx * (rrx + load) ** 2 + a * (rrx + load)
    return a * load / denominator


def loaded_candidate_row(pair: dict[str, Any], load_voltage_v: float, load_power_mw: float, ctx: ModelContext) -> dict[str, Any]:
    load = load_resistance_ohm(load_voltage_v, load_power_mw)
    loaded_eta = loaded_efficiency_value(pair, load_voltage_v, load_power_mw, ctx)
    row = {
        **pair,
        "scenario_id": scenario_id(pair, load_voltage_v, load_power_mw),
        "load_voltage_V": load_voltage_v,
        "target_load_power_mW": load_power_mw,
        "load_resistance_ohm": load,
        "Rload_over_Rload_opt": load / pair["Rload_opt_ohm"],
        "loaded_efficiency": loaded_eta,
        "loaded_efficiency_rank": "",
        "eta_max_rank": "",
    }
    row["candidate_key"] = (
        f"{row['pair_key']}|v{compact_number(load_voltage_v)}|p{compact_number(load_power_mw)}"
    )
    if not (0.0 <= loaded_eta <= 1.0):
        row["status"] = "rejected"
        row["reject_reason"] = append_reason(row["reject_reason"], "loaded_efficiency_out_of_range")
    return row


def coarse_subset(coils: list[dict[str, Any]], dimension_lengths: tuple[int, int, int, int], stride: int) -> list[int]:
    if stride <= 1:
        return list(range(len(coils)))
    max_indexes = tuple(length - 1 for length in dimension_lengths)
    subset = []
    for index, coil in enumerate(coils):
        indexes = coil["index_tuple"]
        if all((item % stride == 0) or (item == max_indexes[pos]) for pos, item in enumerate(indexes)):
            subset.append(index)
    return subset or list(range(len(coils)))


def neighbor_tuples(center: tuple[int, int, int, int], radius: int) -> list[tuple[int, int, int, int]]:
    result = []
    for dn in range(-radius, radius + 1):
        for dd in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                for ds in range(-radius, radius + 1):
                    result.append((center[0] + dn, center[1] + dd, center[2] + dw, center[3] + ds))
    return result


def best_loaded_for_pair(pair: dict[str, Any], load_voltages_v: list[float], load_powers_mw: list[float], ctx: ModelContext) -> float:
    return max(
        loaded_efficiency_value(pair, voltage, power, ctx)
        for voltage in load_voltages_v
        for power in load_powers_mw
    )


def select_seed_rows(
    rows: list[dict[str, Any]],
    space: SearchSpace,
    ctx: ModelContext,
    *,
    eta_count: int,
    loaded_count: int,
) -> list[dict[str, Any]]:
    current_ok = [row for row in rows if row["status"] == "ok"]
    by_eta = sorted(current_ok, key=lambda row: (-row["eta_max"], -row["Q_min"]))
    by_loaded = sorted(
        current_ok,
        key=lambda row: (
            -best_loaded_for_pair(row, space.load_voltages_v, space.load_powers_mw, ctx),
            -row["eta_max"],
        ),
    )
    seed_rows = []
    seen_pairs = set()
    for row in by_eta[:eta_count] + by_loaded[:loaded_count]:
        key = (row["tx_coil_id"], row["rx_coil_id"], row["distance_index"])
        if key not in seen_pairs:
            seed_rows.append(row)
            seen_pairs.add(key)
    return seed_rows


def evaluate_pair_set(
    tx_valid: list[dict[str, Any]],
    rx_valid: list[dict[str, Any]],
    distances_mm: list[float],
    distance_indexes: list[int],
    tx_indexes: list[int],
    rx_indexes: list[int],
    ctx: ModelContext,
    constraints: dict[str, Any],
    evaluated: dict[tuple[int, int, int], dict[str, Any]],
    rejected: list[dict[str, Any]],
    show_progress: bool = False,
    progress_interval_seconds: float = 5.0,
    progress_label: str = "",
) -> None:
    total = len(distance_indexes) * len(tx_indexes) * len(rx_indexes)
    progress = ProgressBar(
        progress_label or "pair-distance evaluation",
        total,
        enabled=show_progress,
        interval_seconds=progress_interval_seconds,
    )
    completed = 0
    before_evaluated = len(evaluated)
    before_rejected = len(rejected)
    for distance_index in distance_indexes:
        distance_mm = distances_mm[distance_index]
        for tx_index in tx_indexes:
            tx = tx_valid[tx_index]
            for rx_index in rx_indexes:
                key = (tx_index, rx_index, distance_index)
                if key not in evaluated:
                    row, reason = pair_metrics(
                        tx,
                        rx_valid[rx_index],
                        distance_mm,
                        distance_index,
                        ctx,
                        constraints,
                    )
                    evaluated[key] = row
                    if reason:
                        rejected.append(row)
                completed += 1
                progress.update(
                    completed,
                    suffix=(
                        f"new={len(evaluated) - before_evaluated} "
                        f"rejected+={len(rejected) - before_rejected}"
                    ),
                )
    progress.update(
        completed,
        suffix=f"new={len(evaluated) - before_evaluated} rejected+={len(rejected) - before_rejected}",
        force=True,
    )


def pair_eta_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (-row["eta_max"], -row["Q_min"], row["distance_mm"], row["pair_key"])


def pair_loaded_heap_key(
    row: dict[str, Any],
    space: SearchSpace,
    ctx: ModelContext,
) -> tuple[float, float, float, float]:
    return (
        best_loaded_for_pair(row, space.load_voltages_v, space.load_powers_mw, ctx),
        float(row["eta_max"]),
        float(row["Q_min"]),
        -float(row["distance_mm"]),
    )


def prune_pair_rows(
    pair_rows: list[dict[str, Any]],
    space: SearchSpace,
    ctx: ModelContext,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pruning = config.get("search", {}).get("pair_pruning", {})
    enabled = bool(pruning.get("enabled", False))
    limit_value = pruning.get("max_pair_rows_per_context")
    before_count = len(pair_rows)
    metadata: dict[str, Any] = {
        "pair_pruning_enabled": enabled,
        "pair_pruning_before_count": before_count,
        "pair_pruning_after_count": before_count,
        "pair_pruning_limit": limit_value,
        "pair_pruning_loaded_fraction": pruning.get("loaded_fraction", None),
        "pair_pruning_removed_count": 0,
    }
    if not enabled or limit_value is None:
        return pair_rows, metadata
    limit = int(limit_value)
    if limit <= 0 or before_count <= limit:
        metadata["pair_pruning_limit"] = limit
        return pair_rows, metadata

    loaded_fraction = float(pruning.get("loaded_fraction", 0.7))
    loaded_fraction = min(max(loaded_fraction, 0.0), 1.0)
    loaded_quota = min(limit, max(0, int(round(limit * loaded_fraction))))
    eta_quota = max(0, limit - loaded_quota)
    metadata["pair_pruning_limit"] = limit
    metadata["pair_pruning_loaded_quota"] = loaded_quota
    metadata["pair_pruning_eta_quota"] = eta_quota

    show_progress = bool(config.get("search", {}).get("show_progress", True))
    progress_interval_seconds = float(config.get("search", {}).get("progress_interval_seconds", 5.0))
    if show_progress:
        print(
            f"Pair pruning for {context_label(ctx)}: "
            f"before={before_count}, limit={limit}, "
            f"loaded_quota={loaded_quota}, eta_quota={eta_quota}",
            flush=True,
        )
    progress = ProgressBar(
        f"pair pruning {context_label(ctx)}",
        before_count,
        enabled=show_progress,
        interval_seconds=progress_interval_seconds,
    )
    selected: dict[str, dict[str, Any]] = {}
    for row in pair_rows[:eta_quota]:
        selected[row["pair_key"]] = row

    loaded_heap: list[tuple[tuple[float, float, float, float], int, dict[str, Any]]] = []
    for index, row in enumerate(pair_rows):
        if loaded_quota:
            entry = (pair_loaded_heap_key(row, space, ctx), index, row)
            if len(loaded_heap) < loaded_quota:
                heapq.heappush(loaded_heap, entry)
            elif entry[0] > loaded_heap[0][0]:
                heapq.heapreplace(loaded_heap, entry)
        progress.update(index + 1, suffix=f"selected_eta={len(selected)} loaded_heap={len(loaded_heap)}")
    progress.update(before_count, suffix=f"selected_eta={len(selected)} loaded_heap={len(loaded_heap)}", force=True)
    for _, _, row in loaded_heap:
        selected[row["pair_key"]] = row

    if len(selected) < limit:
        for row in pair_rows:
            selected[row["pair_key"]] = row
            if len(selected) >= limit:
                break

    pruned_rows = list(selected.values())
    pruned_rows.sort(key=pair_eta_key)
    metadata["pair_pruning_after_count"] = len(pruned_rows)
    metadata["pair_pruning_removed_count"] = before_count - len(pruned_rows)
    if show_progress:
        print(
            f"  pair pruning kept {metadata['pair_pruning_after_count']} rows "
            f"(removed {metadata['pair_pruning_removed_count']})",
            flush=True,
        )
    return pruned_rows, metadata


def bounded_float_values(
    center: float,
    values: list[float],
    *,
    radius: float,
    step: float,
) -> list[float]:
    if not values:
        return [round(center, 10)]
    minimum = min(values)
    maximum = max(values)
    if radius <= 0.0 or step <= 0.0:
        return [round(center, 10)]
    start = max(minimum, center - radius)
    stop = min(maximum, center + radius)
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    result = {round(start + index * step, 10) for index in range(count)}
    result.add(round(center, 10))
    return sorted(value for value in result if minimum - 1e-12 <= value <= maximum + 1e-12)


def bounded_int_values(center: int, values: list[int], *, radius: int) -> list[int]:
    if not values:
        return [center]
    minimum = min(values)
    maximum = max(values)
    result = range(max(minimum, center - radius), min(maximum, center + radius) + 1)
    return list(result)


def fine_index_tuple(turns: int, dout_mm: float, width_mm: float, spacing_mm: float) -> tuple[int, int, int, int]:
    return (
        int(turns),
        int(round(dout_mm * 10000.0)),
        int(round(width_mm * 10000000.0)),
        int(round(spacing_mm * 10000000.0)),
    )


def local_fine_expansion(
    seed_rows: list[dict[str, Any]],
    space: SearchSpace,
    ctx: ModelContext,
    constraints: dict[str, Any],
    config: dict[str, Any],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fine = config.get("search", {}).get("local_fine_expansion", {})
    if not bool(fine.get("enabled", False)) or not seed_rows:
        return []

    n_radius = int(fine.get("N_radius", 1))
    dout_radius_mm = float(fine.get("Dout_radius_mm", 0.5))
    dout_step_mm = float(fine.get("Dout_step_mm", 0.25))
    trace_radius_mm = float(fine.get("trace_width_radius_mm", 2.0 * 0.0254))
    trace_step_mm = float(fine.get("trace_width_step_mm", 1.0 * 0.0254))
    spacing_radius_mm = float(fine.get("spacing_radius_mm", 2.0 * 0.0254))
    spacing_step_mm = float(fine.get("spacing_step_mm", 1.0 * 0.0254))
    distance_radius_mm = float(fine.get("distance_radius_mm", 0.0))
    distance_step_mm = float(fine.get("distance_step_mm", 1.0))

    tx_cache: dict[tuple[int, float, float, float], dict[str, Any]] = {}
    rx_cache: dict[tuple[int, float, float, float], dict[str, Any]] = {}
    fine_rows: dict[str, dict[str, Any]] = {}
    seed_specs = []

    def cached_coil(side: str, key: tuple[int, float, float, float]) -> dict[str, Any]:
        cache = tx_cache if side == "tx" else rx_cache
        if key not in cache:
            turns, dout_mm, width_mm, spacing_mm = key
            cache[key] = build_coil(
                side=side,
                index_tuple=fine_index_tuple(turns, dout_mm, width_mm, spacing_mm),
                turns=turns,
                dout_mm=dout_mm,
                width_mm=width_mm,
                spacing_mm=spacing_mm,
                ctx=ctx,
                constraints=constraints,
            )
        return cache[key]

    for seed in seed_rows:
        tx_n_values = bounded_int_values(int(seed["tx_N"]), space.tx_n, radius=n_radius)
        rx_n_values = bounded_int_values(int(seed["rx_N"]), space.rx_n, radius=n_radius)
        tx_d_values = bounded_float_values(
            float(seed["tx_Dout_mm"]), space.tx_dout_mm, radius=dout_radius_mm, step=dout_step_mm
        )
        rx_d_values = bounded_float_values(
            float(seed["rx_Dout_mm"]), space.rx_dout_mm, radius=dout_radius_mm, step=dout_step_mm
        )
        tx_w_values = bounded_float_values(
            float(seed["tx_trace_width_mm"]), space.tx_width_mm, radius=trace_radius_mm, step=trace_step_mm
        )
        rx_w_values = bounded_float_values(
            float(seed["rx_trace_width_mm"]), space.rx_width_mm, radius=trace_radius_mm, step=trace_step_mm
        )
        tx_s_values = bounded_float_values(
            float(seed["tx_spacing_mm"]), space.tx_spacing_mm, radius=spacing_radius_mm, step=spacing_step_mm
        )
        rx_s_values = bounded_float_values(
            float(seed["rx_spacing_mm"]), space.rx_spacing_mm, radius=spacing_radius_mm, step=spacing_step_mm
        )
        d_values = bounded_float_values(
            float(seed["distance_mm"]), space.distances_mm, radius=distance_radius_mm, step=distance_step_mm
        )
        tx_option_count = len(tx_n_values) * len(tx_d_values) * len(tx_w_values) * len(tx_s_values)
        rx_option_count = len(rx_n_values) * len(rx_d_values) * len(rx_w_values) * len(rx_s_values)
        pair_attempt_count = tx_option_count * rx_option_count * len(d_values)
        seed_specs.append(
            {
                "seed": seed,
                "tx_n_values": tx_n_values,
                "rx_n_values": rx_n_values,
                "tx_d_values": tx_d_values,
                "rx_d_values": rx_d_values,
                "tx_w_values": tx_w_values,
                "rx_w_values": rx_w_values,
                "tx_s_values": tx_s_values,
                "rx_s_values": rx_s_values,
                "d_values": d_values,
                "tx_option_count": tx_option_count,
                "rx_option_count": rx_option_count,
                "pair_attempt_count": pair_attempt_count,
            }
        )

    show_progress = bool(config.get("search", {}).get("show_progress", True))
    progress_interval_seconds = float(config.get("search", {}).get("progress_interval_seconds", 5.0))
    total_pair_attempts = sum(item["pair_attempt_count"] for item in seed_specs)
    if show_progress:
        print(
            f"Local fine expansion for {context_label(ctx)}: "
            f"seeds={len(seed_specs)}, estimated_pair_attempts={total_pair_attempts}",
            flush=True,
        )
    progress = ProgressBar(
        f"local fine expansion {context_label(ctx)}",
        total_pair_attempts,
        enabled=show_progress,
        interval_seconds=progress_interval_seconds,
    )
    completed_attempts = 0
    last_progress_attempts = 0
    rejected_before_all = len(rejected)

    def update_fine_progress(seed_index: int, *, force: bool = False) -> None:
        nonlocal last_progress_attempts
        if not force and completed_attempts - last_progress_attempts < 4096:
            return
        progress.update(
            completed_attempts,
            suffix=(
                f"seed={seed_index}/{len(seed_specs)} "
                f"valid_pairs={len(fine_rows)} "
                f"rejected+={len(rejected) - rejected_before_all}"
            ),
            force=force,
        )
        last_progress_attempts = completed_attempts

    for seed_index, spec in enumerate(seed_specs, start=1):
        seed = spec["seed"]
        if show_progress:
            print(
                "  fine seed "
                f"{seed_index}/{len(seed_specs)}: "
                f"tx_options={spec['tx_option_count']}, "
                f"rx_options={spec['rx_option_count']}, "
                f"distance_options={len(spec['d_values'])}, "
                f"pair_attempts={spec['pair_attempt_count']}, "
                f"seed_eta={seed['eta_max']:.6g}, "
                f"seed_best_loaded={best_loaded_for_pair(seed, space.load_voltages_v, space.load_powers_mw, ctx):.6g}",
                flush=True,
            )
        seed_valid_before = len(fine_rows)
        seed_rejected_before = len(rejected)
        for tx_n in spec["tx_n_values"]:
            for tx_dout in spec["tx_d_values"]:
                for tx_width in spec["tx_w_values"]:
                    for tx_spacing in spec["tx_s_values"]:
                        tx_key = (tx_n, tx_dout, tx_width, tx_spacing)
                        tx = cached_coil("tx", tx_key)
                        if tx["status"] != "ok":
                            completed_attempts += spec["rx_option_count"] * len(spec["d_values"])
                            update_fine_progress(seed_index)
                            continue
                        for rx_n in spec["rx_n_values"]:
                            for rx_dout in spec["rx_d_values"]:
                                for rx_width in spec["rx_w_values"]:
                                    for rx_spacing in spec["rx_s_values"]:
                                        rx_key = (rx_n, rx_dout, rx_width, rx_spacing)
                                        rx = cached_coil("rx", rx_key)
                                        if rx["status"] != "ok":
                                            completed_attempts += len(spec["d_values"])
                                            update_fine_progress(seed_index)
                                            continue
                                        for distance_mm in spec["d_values"]:
                                            distance_index = (
                                                space.distances_mm.index(distance_mm)
                                                if distance_mm in space.distances_mm
                                                else -1
                                            )
                                            row, reason = pair_metrics(
                                                tx,
                                                rx,
                                                distance_mm,
                                                distance_index,
                                                ctx,
                                                constraints,
                                            )
                                            if reason:
                                                rejected.append(row)
                                            elif row["pair_key"] not in fine_rows:
                                                fine_rows[row["pair_key"]] = row
                                            completed_attempts += 1
                                            update_fine_progress(seed_index)
        if show_progress:
            print(
                "  fine seed "
                f"{seed_index}/{len(seed_specs)} done: "
                f"valid_pairs+={len(fine_rows) - seed_valid_before}, "
                f"rejected+={len(rejected) - seed_rejected_before}",
                flush=True,
            )
        update_fine_progress(seed_index, force=True)
    progress.update(
        completed_attempts,
        suffix=f"valid_pairs={len(fine_rows)} rejected+={len(rejected) - rejected_before_all}",
        force=True,
    )
    return list(fine_rows.values())


def evaluate_pairs_for_copper(
    tx_valid: list[dict[str, Any]],
    rx_valid: list[dict[str, Any]],
    tx_lookup: dict[tuple[int, int, int, int], int],
    rx_lookup: dict[tuple[int, int, int, int], int],
    space: SearchSpace,
    ctx: ModelContext,
    constraints: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    search = config["search"]
    total_pair_distance = len(tx_valid) * len(rx_valid) * len(space.distances_mm)
    mode = str(search.get("mode", "auto"))
    if mode == "auto":
        limit = int(search.get("exhaustive_pair_distance_limit", 250000))
        mode = "exhaustive" if total_pair_distance <= limit else "grid_refine"
    if mode == "hybrid":
        mode = "grid_refine"
    show_progress = bool(search.get("show_progress", True))
    progress_interval_seconds = float(search.get("progress_interval_seconds", 5.0))
    started = time.monotonic()
    evaluated: dict[tuple[int, int, int], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []

    if show_progress:
        print(
            f"Pair-distance search for {context_label(ctx)}: "
            f"mode={mode}, possible={total_pair_distance}",
            flush=True,
        )

    all_distance_indexes = list(range(len(space.distances_mm)))
    if mode == "exhaustive":
        evaluate_pair_set(
            tx_valid,
            rx_valid,
            space.distances_mm,
            all_distance_indexes,
            list(range(len(tx_valid))),
            list(range(len(rx_valid))),
            ctx,
            constraints,
            evaluated,
            rejected,
            show_progress=show_progress,
            progress_interval_seconds=progress_interval_seconds,
            progress_label=f"exhaustive pair-distance {context_label(ctx)}",
        )
        pre_fine_seed_rows: list[dict[str, Any]] = []
    elif mode == "grid_refine":
        stride = int(search.get("coarse_stride", 3))
        tx_dims = (len(space.tx_n), len(space.tx_dout_mm), len(space.tx_width_mm), len(space.tx_spacing_mm))
        rx_dims = (len(space.rx_n), len(space.rx_dout_mm), len(space.rx_width_mm), len(space.rx_spacing_mm))
        tx_coarse = coarse_subset(tx_valid, tx_dims, stride)
        rx_coarse = coarse_subset(rx_valid, rx_dims, stride)
        evaluate_pair_set(
            tx_valid,
            rx_valid,
            space.distances_mm,
            all_distance_indexes,
            tx_coarse,
            rx_coarse,
            ctx,
            constraints,
            evaluated,
            rejected,
            show_progress=show_progress,
            progress_interval_seconds=progress_interval_seconds,
            progress_label=f"coarse pair-distance {context_label(ctx)}",
        )
        seed_rows: list[dict[str, Any]] = []
        for round_index in range(int(search.get("local_refine_rounds", 2))):
            current_ok = [row for row in evaluated.values() if row["status"] == "ok"]
            seed_rows = select_seed_rows(
                current_ok,
                space,
                ctx,
                eta_count=int(search.get("coarse_top_pairs", 60)),
                loaded_count=int(search.get("refine_top_pairs", 20)),
            )
            before = len(evaluated)
            radius = int(search.get("local_radius", 1))
            for row in seed_rows:
                tx_neighbors = [
                    tx_lookup[item] for item in neighbor_tuples(row["tx_index_tuple"], radius) if item in tx_lookup
                ]
                rx_neighbors = [
                    rx_lookup[item] for item in neighbor_tuples(row["rx_index_tuple"], radius) if item in rx_lookup
                ]
                d0 = int(row["distance_index"])
                distance_neighbors = [
                    index for index in range(d0 - radius, d0 + radius + 1) if 0 <= index < len(space.distances_mm)
                ]
                evaluate_pair_set(
                    tx_valid,
                    rx_valid,
                    space.distances_mm,
                    distance_neighbors,
                    tx_neighbors,
                    rx_neighbors,
                    ctx,
                    constraints,
                    evaluated,
                    rejected,
                    show_progress=False,
                )
            if show_progress:
                print(
                    f"  refine round {round_index + 1}: added {len(evaluated) - before} pair-distance rows",
                    flush=True,
                )
        current_ok = [row for row in evaluated.values() if row["status"] == "ok"]
        fine = search.get("local_fine_expansion", {})
        pre_fine_seed_rows = select_seed_rows(
            current_ok,
            space,
            ctx,
            eta_count=int(fine.get("seed_top_eta_pairs", search.get("coarse_top_pairs", 60))),
            loaded_count=int(fine.get("seed_top_loaded_pairs", search.get("refine_top_pairs", 20))),
        )
        fine_rows = local_fine_expansion(pre_fine_seed_rows, space, ctx, constraints, config, rejected)
        if fine_rows:
            before = len(evaluated)
            existing_pair_keys = {row["pair_key"] for row in evaluated.values()}
            for fine_index, row in enumerate(fine_rows):
                if row["pair_key"] in existing_pair_keys:
                    continue
                evaluated[(-(fine_index + 1), -1, -1)] = row
                existing_pair_keys.add(row["pair_key"])
            if show_progress:
                print(
                    f"  local fine expansion: added {len(evaluated) - before} pair-distance rows",
                    flush=True,
                )
    else:
        raise ValueError(f"Unsupported search mode: {mode}")

    pair_rows = [row for row in evaluated.values() if row["status"] == "ok"]
    pair_rows.sort(key=pair_eta_key)
    pair_rows, pruning_metadata = prune_pair_rows(pair_rows, space, ctx, config)
    metadata = {
        "frequency_MHz": ctx.frequency_hz / 1e6,
        "layer_spacing_mm": ctx.layer_spacing_mm,
        "temperature_C": ctx.temperature_c,
        "copper_thickness_um": ctx.copper_thickness_um,
        "copper_thickness_oz": ctx.copper_thickness_oz,
        "search_mode_used": mode,
        "possible_pair_distance_count": total_pair_distance,
        "evaluated_pair_distance_count": len(evaluated),
        "valid_pair_distance_count": len(pair_rows),
        "pre_local_fine_seed_pair_count": len(pre_fine_seed_rows),
        "rejected_pair_distance_count": len(rejected),
        "elapsed_seconds": time.monotonic() - started,
        **pruning_metadata,
    }
    return pair_rows, rejected, metadata, pre_fine_seed_rows


def expand_load_candidates(
    pair_rows_by_context: list[tuple[ModelContext, list[dict[str, Any]]]],
    space: SearchSpace,
    *,
    show_progress: bool = False,
    progress_interval_seconds: float = 5.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_pairs = sum(len(pair_rows) for _, pair_rows in pair_rows_by_context)
    total = total_pairs * len(space.load_voltages_v) * len(space.load_powers_mw)
    progress = ProgressBar(
        "load expansion",
        total,
        enabled=show_progress,
        interval_seconds=progress_interval_seconds,
    )
    completed = 0
    for ctx, pair_rows in pair_rows_by_context:
        for pair in pair_rows:
            for voltage in space.load_voltages_v:
                for power_mw in space.load_powers_mw:
                    row = loaded_candidate_row(pair, voltage, power_mw, ctx)
                    if row["status"] == "ok":
                        rows.append(row)
                    completed += 1
                    progress.update(completed, suffix=f"valid_load_rows={len(rows)}")
    progress.update(completed, suffix=f"valid_load_rows={len(rows)}", force=True)
    return rows


def loaded_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -row["loaded_efficiency"],
        -row["eta_max"],
        -row["Q_min"],
        row["distance_mm"],
        row["candidate_key"],
    )


def eta_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -row["eta_max"],
        -row["loaded_efficiency"],
        -row["Q_min"],
        row["distance_mm"],
        row["candidate_key"],
    )


def better_loaded(row: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    return incumbent is None or loaded_sort_key(row) < loaded_sort_key(incumbent)


def better_eta(row: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    return incumbent is None or eta_sort_key(row) < eta_sort_key(incumbent)


class TopRows:
    """Keep only the best N rows without retaining every evaluated load row."""

    def __init__(self, limit: int, objective: str) -> None:
        self.limit = max(int(limit), 1)
        self.objective = objective
        self.heap: list[tuple[tuple[float, float, float, float], int, dict[str, Any]]] = []
        self.counter = 0

    def _heap_key(self, row: dict[str, Any]) -> tuple[float, float, float, float]:
        if self.objective == "loaded_efficiency":
            return (
                float(row["loaded_efficiency"]),
                float(row["eta_max"]),
                float(row["Q_min"]),
                -float(row["distance_mm"]),
            )
        if self.objective == "eta_max":
            return (
                float(row["eta_max"]),
                float(row["loaded_efficiency"]),
                float(row["Q_min"]),
                -float(row["distance_mm"]),
            )
        raise ValueError("objective must be loaded_efficiency or eta_max")

    def push(self, row: dict[str, Any]) -> None:
        entry = (self._heap_key(row), self.counter, row)
        self.counter += 1
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, entry)
        elif entry[0] > self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)

    def rows(self) -> list[dict[str, Any]]:
        return [entry[2] for entry in self.heap]


def profile_limit_value(profile: Any, max_candidate_rows: int) -> int:
    if profile is None:
        return max_candidate_rows
    if isinstance(profile, int):
        return int(profile)
    text = str(profile).strip().lower()
    if text == "short":
        return 10
    if text == "fine":
        return 100
    if text == "full":
        return max_candidate_rows
    return int(text)


def streaming_top_limit(config: dict[str, Any]) -> int:
    max_candidate_rows = int(config.get("search", {}).get("max_candidate_rows", 100000))
    candidate_rows = config.get("output", {}).get("candidate_rows", 50000)
    candidate_limit = max_candidate_rows if candidate_rows is None else int(candidate_rows)
    stage_tables = config.get("output", {}).get("stage_tables", {})
    post_limit = profile_limit_value(stage_tables.get("post_local_profile", "fine"), max_candidate_rows)
    return max(1, min(max(candidate_limit, post_limit), max_candidate_rows))


def assign_top_ranks(*row_lists: list[dict[str, Any]]) -> None:
    rows_by_key: dict[str, dict[str, Any]] = {}
    for rows in row_lists:
        for row in rows:
            rows_by_key[row["candidate_key"]] = row
            row["loaded_efficiency_rank"] = ""
            row["eta_max_rank"] = ""

    loaded_rank_by_key = {
        row["candidate_key"]: index + 1 for index, row in enumerate(sorted(row_lists[0], key=loaded_sort_key))
    }
    eta_rank_by_key = {
        row["candidate_key"]: index + 1 for index, row in enumerate(sorted(row_lists[1], key=eta_sort_key))
    }
    for row in rows_by_key.values():
        row["loaded_efficiency_rank"] = loaded_rank_by_key.get(row["candidate_key"], "")
        row["eta_max_rank"] = eta_rank_by_key.get(row["candidate_key"], "")


def rank_load_candidates_streaming(
    pair_rows_by_context: list[tuple[ModelContext, list[dict[str, Any]]]],
    space: SearchSpace,
    config: dict[str, Any],
    *,
    show_progress: bool = False,
    progress_interval_seconds: float = 5.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    top_limit = streaming_top_limit(config)
    loaded_top = TopRows(top_limit, "loaded_efficiency")
    eta_top = TopRows(top_limit, "eta_max")
    per_loaded: dict[str, dict[str, Any]] = {}
    per_eta: dict[str, dict[str, Any]] = {}
    total_pairs = sum(len(pair_rows) for _, pair_rows in pair_rows_by_context)
    total = total_pairs * len(space.load_voltages_v) * len(space.load_powers_mw)
    progress = ProgressBar(
        "streaming load ranking",
        total,
        enabled=show_progress,
        interval_seconds=progress_interval_seconds,
    )
    completed = 0
    valid_rows = 0
    started = time.monotonic()
    for ctx, pair_rows in pair_rows_by_context:
        for pair in pair_rows:
            for voltage in space.load_voltages_v:
                for power_mw in space.load_powers_mw:
                    row = loaded_candidate_row(pair, voltage, power_mw, ctx)
                    if row["status"] == "ok":
                        valid_rows += 1
                        loaded_top.push(row)
                        eta_top.push(row)
                        scenario = row["scenario_id"]
                        if better_loaded(row, per_loaded.get(scenario)):
                            per_loaded[scenario] = row
                        if better_eta(row, per_eta.get(scenario)):
                            per_eta[scenario] = row
                    completed += 1
                    progress.update(
                        completed,
                        suffix=(
                            f"valid_load_rows={valid_rows} "
                            f"top_limit={top_limit} scenarios={len(per_loaded)}"
                        ),
                    )
    progress.update(
        completed,
        suffix=f"valid_load_rows={valid_rows} top_limit={top_limit} scenarios={len(per_loaded)}",
        force=True,
    )

    rows_loaded = sorted(loaded_top.rows(), key=loaded_sort_key)
    rows_eta = sorted(eta_top.rows(), key=eta_sort_key)
    per_loaded_rows = sorted(per_loaded.values(), key=loaded_sort_key)
    per_eta_rows = sorted(per_eta.values(), key=eta_sort_key)
    assign_top_ranks(rows_loaded, rows_eta, per_loaded_rows, per_eta_rows)
    metadata = {
        "mode": "streaming_top_k",
        "top_limit": top_limit,
        "total_load_evaluation_count": total,
        "valid_load_candidate_count": valid_rows,
        "global_loaded_rows_retained": len(rows_loaded),
        "global_eta_rows_retained": len(rows_eta),
        "per_scenario_loaded_rows": len(per_loaded_rows),
        "per_scenario_eta_rows": len(per_eta_rows),
        "elapsed_seconds": time.monotonic() - started,
    }
    return rows_loaded, rows_eta, per_loaded_rows, per_eta_rows, metadata


def assign_ranks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_loaded = sorted(rows, key=loaded_sort_key)
    by_eta = sorted(rows, key=eta_sort_key)
    loaded_rank_by_key = {row["candidate_key"]: index + 1 for index, row in enumerate(by_loaded)}
    eta_rank_by_key = {row["candidate_key"]: index + 1 for index, row in enumerate(by_eta)}
    for row in rows:
        row["loaded_efficiency_rank"] = loaded_rank_by_key[row["candidate_key"]]
        row["eta_max_rank"] = eta_rank_by_key[row["candidate_key"]]
    return by_loaded, by_eta


def per_scenario_best(rows: list[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["scenario_id"]].append(row)
    if objective == "loaded_efficiency":
        key = lambda row: (-row["loaded_efficiency"], -row["eta_max"], -row["Q_min"])
    elif objective == "eta_max":
        key = lambda row: (-row["eta_max"], -row["loaded_efficiency"], -row["Q_min"])
    else:
        raise ValueError("objective must be loaded_efficiency or eta_max")
    best = [sorted(group_rows, key=key)[0] for group_rows in groups.values()]
    best.sort(key=key)
    return best


def verify_candidates(rows_loaded: list[dict[str, Any]], rows_eta: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    sorted_loaded = all(
        rows_loaded[index]["loaded_efficiency"] >= rows_loaded[index + 1]["loaded_efficiency"]
        for index in range(len(rows_loaded) - 1)
    )
    sorted_eta = all(
        rows_eta[index]["eta_max"] >= rows_eta[index + 1]["eta_max"]
        for index in range(len(rows_eta) - 1)
    )
    numeric = all(
        finite_positive(row["M_uH"])
        and 0.0 <= row["k"] < 1.0
        and 0.0 <= row["eta_max"] <= 1.0
        and 0.0 <= row["loaded_efficiency"] <= 1.0
        and row["loaded_efficiency"] <= row["eta_max"] + 1e-12
        for row in rows_loaded
    )
    spot = {}
    if rows_loaded:
        row = rows_loaded[0]
        omega = float(row["frequency_MHz"]) * 1e6 * 2.0 * math.pi
        mutual_h = row["M_uH"] * UH_TO_H
        chi_kq = row["k"] ** 2 * row["tx_Q_model"] * row["rx_Q_model"]
        chi_math = (omega * mutual_h) ** 2 / (
            row["tx_Rac_model_ohm"] * row["rx_Rac_model_ohm"]
        )
        eta = chi_kq / (1.0 + math.sqrt(1.0 + chi_kq)) ** 2
        rload = row["load_voltage_V"] ** 2 / (row["target_load_power_mW"] / 1000.0)
        rload_opt = row["rx_Rac_model_ohm"] * math.sqrt(1.0 + chi_kq)
        spot = {
            "top_loaded_chi_kq_error": abs(chi_kq - row["chi_k2Q1Q2"]),
            "top_loaded_chi_math_error": abs(chi_math - row["chi_math"]),
            "top_loaded_eta_error": abs(eta - row["eta_max"]),
            "top_loaded_Rload_error_ohm": abs(rload - row["load_resistance_ohm"]),
            "top_loaded_Rload_opt_error_ohm": abs(rload_opt - row["Rload_opt_ohm"]),
        }
    return {
        "sorted_by_loaded_efficiency_desc": sorted_loaded,
        "sorted_by_eta_max_desc": sorted_eta,
        "candidate_numeric_checks_pass": numeric,
        "loaded_efficiency_not_above_eta_max": all(
            row["loaded_efficiency"] <= row["eta_max"] + 1e-12 for row in rows_loaded
        ),
        "spot_checks": spot,
    }
