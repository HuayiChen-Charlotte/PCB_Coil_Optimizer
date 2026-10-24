"""Config loading, output writing, and run orchestration."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .models import (
    build_model_context,
    build_search_space,
    deep_update,
    default_config,
)
from .optimizer import (
    assign_ranks,
    build_coil_library,
    evaluate_pairs_for_copper,
    expand_load_candidates,
    rank_load_candidates_streaming,
    verify_candidates,
)


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """Resolve relative output paths from the caller's working directory."""
    output_dir = Path(str(config["output_dir"])).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    return output_dir.resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    merged = default_config()
    deep_update(merged, config)
    return merged


def format_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        if abs(value) >= 1000.0:
            return f"{value:.10g}"
        return f"{value:.12g}"
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return value


def serializable(row: dict[str, Any]) -> dict[str, Any]:
    skip = {"radii_m", "index_tuple"}
    return {key: value for key, value in row.items() if key not in skip}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write {path}. Close the file if it is open, or choose a different output_dir."
        ) from exc


def candidate_fieldnames() -> list[str]:
    return [
        "candidate_key",
        "pair_key",
        "scenario_id",
        "loaded_efficiency_rank",
        "eta_max_rank",
        "frequency_MHz",
        "layer_spacing_mm",
        "temperature_C",
        "rx_Dout_mm",
        "distance_mm",
        "Norm_Dcoil_percent",
        "load_voltage_V",
        "target_load_power_mW",
        "load_resistance_ohm",
        "copper_thickness_um",
        "copper_thickness_oz",
        "tx_coil_id",
        "tx_layers",
        "tx_connection",
        "tx_N",
        "tx_Dout_mm",
        "tx_Din_mm",
        "tx_Din_webench_fill_mm",
        "tx_fill_factor_webench",
        "tx_trace_width_mm",
        "tx_spacing_mm",
        "tx_L_total_uH",
        "tx_Rdc_ohm",
        "tx_Rskin_ohm",
        "tx_qian_phi",
        "tx_rac_model_version",
        "tx_rac_two_layer_extension_version",
        "tx_rac_model_domain",
        "tx_Rprox_qian_same_layer_ohm",
        "tx_Rac_qian_same_layer_ohm",
        "tx_Q_qian_same_layer",
        "tx_Rprox_qian_other_layer_only_ohm",
        "tx_Rprox_qian_cross_term_ohm",
        "tx_Rprox_qian_two_layer_extension_ohm",
        "tx_Rac_qian_two_layer_extension_ohm",
        "tx_Q_qian_two_layer_extension",
        "tx_Rprox_legacy_kuhn_ohm",
        "tx_Rac_legacy_kuhn_ohm",
        "tx_Q_legacy_kuhn",
        "tx_Rac_model_ohm",
        "tx_Q_model",
        "tx_Rprox_ohm",
        "tx_Rac_theorybook_ohm",
        "tx_Q_theorybook",
        "tx_C_total_pF",
        "tx_active_cpar_model",
        "tx_Cpar_active_pF",
        "tx_C_sensor_activeCpar_pF",
        "tx_fsrf_activeCpar_MHz",
        "tx_Cpar_fit_pF",
        "tx_C_sensor_fitCpar_pF",
        "tx_fsrf_fitCpar_MHz",
        "tx_Cpar_webench_pF",
        "tx_C_sensor_webenchCpar_pF",
        "tx_Cpar_book_pF",
        "tx_C_sensor_bookCpar_pF",
        "tx_fsrf_webenchCpar_MHz",
        "tx_fsrf_bookCpar_MHz",
        "rx_coil_id",
        "rx_layers",
        "rx_connection",
        "rx_N",
        "rx_Din_mm",
        "rx_Din_webench_fill_mm",
        "rx_fill_factor_webench",
        "rx_trace_width_mm",
        "rx_spacing_mm",
        "rx_L_total_uH",
        "rx_Rdc_ohm",
        "rx_Rskin_ohm",
        "rx_qian_phi",
        "rx_rac_model_version",
        "rx_rac_two_layer_extension_version",
        "rx_rac_model_domain",
        "rx_Rprox_qian_same_layer_ohm",
        "rx_Rac_qian_same_layer_ohm",
        "rx_Q_qian_same_layer",
        "rx_Rprox_qian_other_layer_only_ohm",
        "rx_Rprox_qian_cross_term_ohm",
        "rx_Rprox_qian_two_layer_extension_ohm",
        "rx_Rac_qian_two_layer_extension_ohm",
        "rx_Q_qian_two_layer_extension",
        "rx_Rprox_legacy_kuhn_ohm",
        "rx_Rac_legacy_kuhn_ohm",
        "rx_Q_legacy_kuhn",
        "rx_Rac_model_ohm",
        "rx_Q_model",
        "rx_Rprox_ohm",
        "rx_Rac_theorybook_ohm",
        "rx_Q_theorybook",
        "rx_C_total_pF",
        "rx_active_cpar_model",
        "rx_Cpar_active_pF",
        "rx_C_sensor_activeCpar_pF",
        "rx_fsrf_activeCpar_MHz",
        "rx_Cpar_fit_pF",
        "rx_C_sensor_fitCpar_pF",
        "rx_fsrf_fitCpar_MHz",
        "rx_Cpar_webench_pF",
        "rx_C_sensor_webenchCpar_pF",
        "rx_Cpar_book_pF",
        "rx_C_sensor_bookCpar_pF",
        "rx_fsrf_webenchCpar_MHz",
        "rx_fsrf_bookCpar_MHz",
        "M_uH",
        "k",
        "chi_k2Q1Q2",
        "chi_math",
        "chi_identity_error",
        "eta_max",
        "Rload_opt_ohm",
        "Rload_over_Rload_opt",
        "loaded_efficiency",
        "Q_min",
        "Q_geom",
        "status",
        "reject_reason",
        "fasthenry_status",
    ]


def rejected_fieldnames() -> list[str]:
    return list(
        dict.fromkeys(
            [
                "type",
                "coil_id",
                "pair_key",
                "side",
                "status",
                "reject_reason",
                "copper_thickness_um",
                "copper_thickness_oz",
                "N",
                "Dout_mm",
                "Din_mm",
                "Din_webench_fill_mm",
                "fill_factor_webench",
                "trace_width_mm",
                "spacing_mm",
            ]
            + candidate_fieldnames()
        )
    )


def build_rejected_rows(
    tx_rejected: list[dict[str, Any]],
    rx_rejected: list[dict[str, Any]],
    pair_rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for coil in tx_rejected + rx_rejected:
        row = serializable(coil)
        row["type"] = "coil"
        rows.append(row)
    for pair in pair_rejected:
        row = serializable(pair)
        row["type"] = "pair_distance"
        rows.append(row)
    return rows


def limited(rows: list[dict[str, Any]], value: Any) -> list[dict[str, Any]]:
    if value is None:
        return rows
    return rows[: int(value)]


def profile_limit(profile: Any) -> int | None:
    if profile is None:
        return None
    if isinstance(profile, int):
        return profile
    text = str(profile).strip().lower()
    if text == "full":
        return None
    if text == "short":
        return 10
    if text == "fine":
        return 100
    return int(text)


def write_outputs(
    config: dict[str, Any],
    rows_loaded: list[dict[str, Any]],
    rows_eta: list[dict[str, Any]],
    pre_local_rows_loaded: list[dict[str, Any]],
    pre_local_rows_eta: list[dict[str, Any]],
    per_loaded: list[dict[str, Any]],
    per_eta: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(config.get("search", {}).get("show_progress", True)):
        print(f"Writing coil optimizer outputs to {output_dir}", flush=True)
    candidate_rows = config.get("output", {}).get("candidate_rows", 50000)
    write_per_scenario = bool(config.get("output", {}).get("write_per_scenario", False))
    per_scenario_rows = config.get("output", {}).get("per_scenario_rows", None)
    rejected_limit = config.get("output", {}).get("rejected_rows", 50000)
    write_rejected = bool(config.get("output", {}).get("write_rejected", True))
    stage_tables = config.get("output", {}).get("stage_tables", {})
    write_stage_tables = bool(stage_tables.get("enabled", True))
    pre_local_limit = profile_limit(stage_tables.get("pre_local_profile", "fine"))
    post_local_limit = profile_limit(stage_tables.get("post_local_profile", "fine"))

    paths = {
        "config_used": output_dir / "config_used.json",
        "candidates_loaded_efficiency_csv": output_dir / "candidates_loaded_efficiency.csv",
        "candidates_eta_max_csv": output_dir / "candidates_eta_max.csv",
        "pre_local_fine_loaded_efficiency_csv": output_dir / "pre_local_fine_loaded_efficiency.csv",
        "pre_local_fine_eta_max_csv": output_dir / "pre_local_fine_eta_max.csv",
        "post_local_fine_loaded_efficiency_csv": output_dir / "post_local_fine_loaded_efficiency.csv",
        "post_local_fine_eta_max_csv": output_dir / "post_local_fine_eta_max.csv",
        "per_scenario_best_loaded_efficiency_csv": output_dir / "per_scenario_best_loaded_efficiency.csv",
        "per_scenario_best_eta_max_csv": output_dir / "per_scenario_best_eta_max.csv",
        "rejected_csv": output_dir / "rejected.csv",
        "summary_json": output_dir / "summary.json",
    }
    fields = candidate_fieldnames()
    write_csv(paths["candidates_loaded_efficiency_csv"], limited(rows_loaded, candidate_rows), fields)
    write_csv(paths["candidates_eta_max_csv"], limited(rows_eta, candidate_rows), fields)
    write_csv(
        paths["pre_local_fine_loaded_efficiency_csv"],
        limited(pre_local_rows_loaded, pre_local_limit) if write_stage_tables else [],
        fields,
    )
    write_csv(
        paths["pre_local_fine_eta_max_csv"],
        limited(pre_local_rows_eta, pre_local_limit) if write_stage_tables else [],
        fields,
    )
    write_csv(
        paths["post_local_fine_loaded_efficiency_csv"],
        limited(rows_loaded, post_local_limit) if write_stage_tables else [],
        fields,
    )
    write_csv(
        paths["post_local_fine_eta_max_csv"],
        limited(rows_eta, post_local_limit) if write_stage_tables else [],
        fields,
    )
    write_csv(
        paths["per_scenario_best_loaded_efficiency_csv"],
        limited(per_loaded, per_scenario_rows) if write_per_scenario else [],
        fields,
    )
    write_csv(
        paths["per_scenario_best_eta_max_csv"],
        limited(per_eta, per_scenario_rows) if write_per_scenario else [],
        fields,
    )
    write_csv(
        paths["rejected_csv"],
        limited(rejected_rows, rejected_limit) if write_rejected else [],
        rejected_fieldnames(),
    )

    config_copy = json.loads(json.dumps(config))
    config_copy["derived"] = summary["derived"]
    paths["config_used"].write_text(json.dumps(config_copy, indent=2), encoding="utf-8")
    summary["paths"] = {key: str(value) for key, value in paths.items()}
    summary["top_loaded_efficiency_candidate"] = rows_loaded[0] if rows_loaded else None
    summary["top_eta_max_candidate"] = rows_eta[0] if rows_eta else None
    summary["written_counts"] = {
        "candidates_loaded_efficiency": min(len(rows_loaded), candidate_rows)
        if candidate_rows is not None
        else len(rows_loaded),
        "candidates_eta_max": min(len(rows_eta), candidate_rows)
        if candidate_rows is not None
        else len(rows_eta),
        "pre_local_fine_loaded_efficiency": len(limited(pre_local_rows_loaded, pre_local_limit))
        if write_stage_tables
        else 0,
        "pre_local_fine_eta_max": len(limited(pre_local_rows_eta, pre_local_limit))
        if write_stage_tables
        else 0,
        "post_local_fine_loaded_efficiency": len(limited(rows_loaded, post_local_limit))
        if write_stage_tables
        else 0,
        "post_local_fine_eta_max": len(limited(rows_eta, post_local_limit)) if write_stage_tables else 0,
        "per_scenario_best_loaded_efficiency": len(limited(per_loaded, per_scenario_rows))
        if write_per_scenario
        else 0,
        "per_scenario_best_eta_max": len(limited(per_eta, per_scenario_rows)) if write_per_scenario else 0,
        "rejected": len(limited(rejected_rows, rejected_limit)) if write_rejected else 0,
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_pre_local_checkpoint(
    config: dict[str, Any],
    pre_local_rows_loaded: list[dict[str, Any]],
    pre_local_rows_eta: list[dict[str, Any]],
    checkpoint_summary: dict[str, Any],
) -> None:
    stage_tables = config.get("output", {}).get("stage_tables", {})
    if not bool(stage_tables.get("enabled", True)):
        return
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(config.get("search", {}).get("show_progress", True)):
        print(f"Writing pre-local seed checkpoint to {output_dir}", flush=True)

    paths = {
        "config_used": output_dir / "config_used.json",
        "pre_local_fine_loaded_efficiency_csv": output_dir / "pre_local_fine_loaded_efficiency.csv",
        "pre_local_fine_eta_max_csv": output_dir / "pre_local_fine_eta_max.csv",
        "pre_local_checkpoint_summary_json": output_dir / "pre_local_checkpoint_summary.json",
    }
    pre_local_limit = profile_limit(stage_tables.get("pre_local_profile", "fine"))
    fields = candidate_fieldnames()
    write_csv(paths["pre_local_fine_loaded_efficiency_csv"], limited(pre_local_rows_loaded, pre_local_limit), fields)
    write_csv(paths["pre_local_fine_eta_max_csv"], limited(pre_local_rows_eta, pre_local_limit), fields)
    config_copy = json.loads(json.dumps(config))
    config_copy["checkpoint_status"] = "pre_local_seed_tables_written_before_streaming_load_ranking"
    paths["config_used"].write_text(json.dumps(config_copy, indent=2), encoding="utf-8")
    checkpoint = {
        "status": "partial_checkpoint",
        "note": "Pre-local seed tables were written before streaming load ranking. Final outputs overwrite config_used.json when complete.",
        "paths": {key: str(value) for key, value in paths.items()},
        **checkpoint_summary,
    }
    paths["pre_local_checkpoint_summary_json"].write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def run(config: dict[str, Any]) -> dict[str, Any]:
    space = build_search_space(config)
    constraints = config["constraints"]
    show_progress = bool(config.get("search", {}).get("show_progress", True))
    progress_interval_seconds = float(
        config.get("search", {}).get("progress_interval_seconds", 5.0)
    )
    all_tx_rejected: list[dict[str, Any]] = []
    all_rx_rejected: list[dict[str, Any]] = []
    all_pair_rejected: list[dict[str, Any]] = []
    pair_rows_by_context = []
    pre_local_seed_pairs_by_context = []
    search_metadata = []
    coil_metadata = []

    for frequency_mhz in space.frequencies_mhz:
        for layer_spacing_mm in space.layer_spacings_mm:
            for temperature_c in space.temperatures_c:
                for copper_um in space.copper_thicknesses_um:
                    ctx = build_model_context(
                        config,
                        copper_um,
                        frequency_mhz=frequency_mhz,
                        layer_spacing_mm=layer_spacing_mm,
                        temperature_c=temperature_c,
                    )
                    print(
                        f"Building coil libraries for f={ctx.frequency_hz / 1e6:.6g} MHz, "
                        f"layers={ctx.layers}, "
                        f"h={ctx.layer_spacing_mm:.6g} mm, T={ctx.temperature_c:.6g} C, "
                        f"Cu={ctx.copper_thickness_oz:.3g} oz ({ctx.copper_thickness_um:.3g} um)",
                        flush=True,
                    )
                    tx_valid, tx_rejected, tx_lookup = build_coil_library(
                        "tx",
                        space.tx_n,
                        space.tx_dout_mm,
                        space.tx_width_mm,
                        space.tx_spacing_mm,
                        ctx,
                        constraints,
                        show_progress=show_progress,
                        progress_interval_seconds=progress_interval_seconds,
                    )
                    rx_valid, rx_rejected, rx_lookup = build_coil_library(
                        "rx",
                        space.rx_n,
                        space.rx_dout_mm,
                        space.rx_width_mm,
                        space.rx_spacing_mm,
                        ctx,
                        constraints,
                        show_progress=show_progress,
                        progress_interval_seconds=progress_interval_seconds,
                    )
                    all_tx_rejected.extend(tx_rejected)
                    all_rx_rejected.extend(rx_rejected)
                    coil_metadata.append(
                        {
                            "frequency_MHz": ctx.frequency_hz / 1e6,
                            "layers": ctx.layers,
                            "layer_spacing_mm": ctx.layer_spacing_mm,
                            "temperature_C": ctx.temperature_c,
                            "copper_thickness_um": ctx.copper_thickness_um,
                            "copper_thickness_oz": ctx.copper_thickness_oz,
                            "tx_valid_coils": len(tx_valid),
                            "tx_rejected_coils": len(tx_rejected),
                            "rx_valid_coils": len(rx_valid),
                            "rx_rejected_coils": len(rx_rejected),
                        }
                    )
                    if not tx_valid or not rx_valid:
                        continue
                    pair_rows, pair_rejected, meta, pre_local_seed_rows = evaluate_pairs_for_copper(
                        tx_valid,
                        rx_valid,
                        tx_lookup,
                        rx_lookup,
                        space,
                        ctx,
                        constraints,
                        config,
                    )
                    pair_rows_by_context.append((ctx, pair_rows))
                    pre_local_seed_pairs_by_context.append((ctx, pre_local_seed_rows))
                    all_pair_rejected.extend(pair_rejected)
                    search_metadata.append(meta)

    pre_local_load_rows = expand_load_candidates(
        pre_local_seed_pairs_by_context,
        space,
        show_progress=False,
        progress_interval_seconds=progress_interval_seconds,
    )
    pre_local_rows_loaded, pre_local_rows_eta = assign_ranks(pre_local_load_rows) if pre_local_load_rows else ([], [])
    write_pre_local_checkpoint(
        config,
        pre_local_rows_loaded,
        pre_local_rows_eta,
        {
            "pre_local_seed_pair_count": sum(len(rows) for _, rows in pre_local_seed_pairs_by_context),
            "pre_local_loaded_candidate_count": len(pre_local_load_rows),
            "pre_local_loaded_rows_written_available": len(pre_local_rows_loaded),
            "pre_local_eta_rows_written_available": len(pre_local_rows_eta),
            "search_context_count": len(pair_rows_by_context),
        },
    )

    rows_loaded, rows_eta, per_loaded, per_eta, load_ranking_metadata = rank_load_candidates_streaming(
        pair_rows_by_context,
        space,
        config,
        show_progress=show_progress,
        progress_interval_seconds=progress_interval_seconds,
    )
    if not rows_loaded:
        raise RuntimeError("No valid coil optimizer candidates after constraints.")

    rejected_rows = build_rejected_rows(all_tx_rejected, all_rx_rejected, all_pair_rejected)
    checks = verify_candidates(rows_loaded, rows_eta, config)
    scenario_count = (
        len(space.frequencies_mhz)
        * len(space.rx_dout_mm)
        * len(space.distances_mm)
        * len(space.load_voltages_v)
        * len(space.load_powers_mw)
        * len(space.copper_thicknesses_um)
        * len(space.layer_spacings_mm)
        * len(space.temperatures_c)
    )
    layers = int(config.get("layers", 2))
    if layers == 1:
        model_description = "math-structured single-layer analytic PCB coil optimizer"
        connection = "single_layer"
        l_total_formula = "L_self"
        m_formula = "M(z)"
        active_cpar = "webench"
    elif layers == 2:
        model_description = "math-structured self-consistent 2-layer series-aiding analytic PCB coil optimizer"
        connection = "series_aiding"
        l_total_formula = "2*L_self + 2*M_interlayer"
        m_formula = "M(z) + 2*M(z+h_layer) + M(z+2*h_layer)"
        active_cpar = "fit"
    else:
        raise ValueError("layers must be 1 or 2.")
    summary = {
        "case_name": config["case_name"],
        "model": {
            "description": model_description,
            "shape": "circular",
            "layers": layers,
            "connection": connection,
            "frequency_regime": "low-frequency/quasi-static",
            "L_total": l_total_formula,
            "M_TX_RX": m_formula,
            "Rac": "Rac_model = Rskin_Jow + Rprox_Qian_same_layer",
            "Rac_default_model": "qian_2018_circular_same_layer_v1",
            "active_Cpar_model": active_cpar,
            "active_Cpar_policy": "layers=1 uses WEBENCH Cpar; layers=2 uses fitted measured-SRF Cpar",
            "Rac_default_scope": (
                "Each physical copper layer is evaluated independently with exact circular-loop "
                "normal H fields and 5-point trace-width averaging. Interlayer proximity is excluded "
                "from ranking and efficiency."
            ),
            "Rac_two_layer_diagnostic": (
                "qian_2018_circular_two_layer_extension_v1 includes the aligned series-aiding "
                "layer field and is output only as an uncalibrated diagnostic."
            ),
            "Rac_reference": (
                "G. Qian, Y. Cheng, G. Chen, and G. Wang, New AC Resistance Calculation of "
                "Printed Spiral Coils for Wireless Power Transfer, ISQED 2018."
            ),
            "Rac_compatibility": (
                "Rprox_ohm, Rac_theorybook_ohm, and Q_theorybook are aliases of the selected "
                "Qian same-layer model; legacy Kuhn results use explicit legacy_kuhn names."
            ),
            "Q": "Q_model = omega*L/Rac_model",
            "chi_identity": "k^2*Q_TX*Q_RX = (omega*M)^2/(R_TX*R_RX)",
            "eta_max": "chi/(1+sqrt(1+chi))^2",
            "Rload": "Vload^2/Pload",
            "Rload_opt": "R_RX*sqrt(1+chi)",
            "loaded_efficiency": "(omega*M)^2*Rload/[R_TX*(R_RX+Rload)^2+(omega*M)^2*(R_RX+Rload)]",
            "distance_reference": "distance_mm is closest copper-current-plane to copper-current-plane separation",
            "excluded_losses": [
                "tissue loss",
                "capacitor ESR",
                "source resistance",
                "matching network",
                "rectifier/regulator",
                "offset/tilt",
            ],
        },
        "inputs": {
            "frequency_MHz": config["frequency_MHz"],
            "layer_spacing_mm": config["layer_spacing_mm"],
            "temperature_C": config["temperature_C"],
            "rx_Dout_mm": config["rx_Dout_mm"],
            "tx_Dout_mm": config["tx_Dout_mm"],
            "distance_mm": config["distance_mm"],
            "rx_load": config["rx_load"],
            "copper_thickness_um": config["copper_thickness_um"],
            "constraints": config["constraints"],
        },
        "derived": {
            "scenario_count": scenario_count,
            "load_values": [
                {
                    "voltage_V": voltage,
                    "power_mW": power,
                    "load_resistance_ohm": voltage**2 / (power / 1000.0),
                }
                for voltage in space.load_voltages_v
                for power in space.load_powers_mw
            ],
        },
        "counts": {
            "coil_libraries": coil_metadata,
            "search": search_metadata,
            "valid_pair_distance_count": sum(item["valid_pair_distance_count"] for item in search_metadata),
            "evaluated_pair_distance_count": sum(item["evaluated_pair_distance_count"] for item in search_metadata),
            "pre_local_fine_seed_pair_count": sum(
                item.get("pre_local_fine_seed_pair_count", 0) for item in search_metadata
            ),
            "pre_local_fine_loaded_candidate_count": len(pre_local_load_rows),
            "valid_loaded_candidate_count": load_ranking_metadata["valid_load_candidate_count"],
            "total_load_evaluation_count": load_ranking_metadata["total_load_evaluation_count"],
            "retained_global_loaded_candidate_count": len(rows_loaded),
            "retained_global_eta_candidate_count": len(rows_eta),
            "scenario_count": scenario_count,
            "per_scenario_loaded_rows": len(per_loaded),
            "per_scenario_eta_rows": len(per_eta),
            "rejected_rows_available": len(rejected_rows),
        },
        "load_ranking": load_ranking_metadata,
        "cache_structure": {
            "coil_library": "computed once per model context and coil geometry",
            "pair_distance_library": "computed once per model context, TX coil, RX coil, and distance",
            "load_evaluation": "streamed after pair-distance cache using Vload^2/Pload; only top-K and per-scenario best rows are retained",
        },
        "fasthenry": {
            "status": "not_run",
            "note": "FastHenry validation is intentionally separate and can be added as a top-K rerank stage.",
        },
        "checks": checks,
    }
    return write_outputs(
        config,
        rows_loaded,
        rows_eta,
        pre_local_rows_loaded,
        pre_local_rows_eta,
        per_loaded,
        per_eta,
        rejected_rows,
        summary,
    )

