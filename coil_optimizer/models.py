"""Configuration helpers for the math-structured coil optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


RHO_20_OHM_M = 1.68e-8
RHO_TEMP_COEFF_PER_C = 0.00393
OZ_CU_TO_UM = 34.7


def default_config() -> dict[str, Any]:
    return {
        "case_name": "coil_optimizer_example",
        "output_dir": "outputs/coil_optimizer_example",
        "frequency_MHz": 6.78,
        "layers": 2,
        "rx_Dout_mm": {"min": 5.0, "max": 20.0, "step": 0.5},
        "tx_Dout_mm": {"min": 5.0, "max": 40.0, "step": 0.5},
        "distance_mm": {"min": 5.0, "max": 10.0, "step": 1.0},
        "rx_load": {
            "voltage_V": 1.8,
            "power_mW": {"min": 20.0, "max": 80.0, "step": 10.0},
        },
        "rx_N": {"min": 1, "max": 35, "step": 1},
        "tx_N": {"min": 1, "max": 70, "step": 1},
        "tx_trace_width_mm": {"min": 0.1016, "max": 0.254, "step": 0.0254},
        "tx_spacing_mm": {"min": 0.1016, "max": 0.254, "step": 0.0254},
        "rx_trace_width_mm": {"min": 0.1016, "max": 0.254, "step": 0.0254},
        "rx_spacing_mm": {"min": 0.1016, "max": 0.254, "step": 0.0254},
        "layer_spacing_mm": 0.6,
        "copper_thickness_um": 34.7,
        "temperature_C": 25.0,
        "constraints": {
            "trace_width_min_mm": 0.1016,
            "spacing_min_mm": 0.1016,
            "fill_factor_min": 0.3,
            "C_sensor_min_pF": 10.0,
            "C_sensor_max_pF": 10000.0,
            "minimum_SRF_ratio": 4.0,
            "k_min": 0.0,
            "k_max": 1.0,
            "norm_Dcoil_percent_min": 40.0,
            "norm_Dcoil_percent_max": 50.0,
        },
        "search": {
            "mode": "grid_refine",
            "exhaustive_pair_distance_limit": 250000,
            "coarse_stride": 6,
            "coarse_top_pairs": 30,
            "refine_top_pairs": 30,
            "local_radius": 1,
            "local_refine_rounds": 1,
            "pair_pruning": {
                "enabled": True,
                "max_pair_rows_per_context": 200000,
                "loaded_fraction": 0.7,
            },
            "local_fine_expansion": {
                "enabled": True,
                "seed_top_eta_pairs": 10,
                "seed_top_loaded_pairs": 15,
                "N_radius": 1,
                "Dout_radius_mm": 0.5,
                "Dout_step_mm": 0.5,
                "trace_width_radius_mm": 0.0254,
                "trace_width_step_mm": 0.0254,
                "spacing_radius_mm": 0.0254,
                "spacing_step_mm": 0.0254,
                "distance_radius_mm": 1.0,
                "distance_step_mm": 0.5,
            },
            "max_candidate_rows": 100000,
            "show_progress": True,
            "progress_interval_seconds": 5.0,
        },
        "output": {
            "candidate_rows": 200,
            "stage_tables": {
                "enabled": True,
                "pre_local_profile": "fine",
                "post_local_profile": "fine",
            },
            "write_per_scenario": True,
            "per_scenario_rows": None,
            "write_rejected": False,
            "rejected_rows": 0,
        },
    }


@dataclass(frozen=True)
class GridSpec:
    minimum: float
    maximum: float
    step: float
    integer: bool = False

    @classmethod
    def from_config(cls, data: dict[str, Any], *, integer: bool = False) -> "GridSpec":
        return cls(float(data["min"]), float(data["max"]), float(data.get("step", 1)), integer)

    def values(self) -> list[int] | list[float]:
        if self.step <= 0:
            raise ValueError("Grid step must be positive.")
        count = int(math.floor((self.maximum - self.minimum) / self.step + 1e-9)) + 1
        values = [self.minimum + index * self.step for index in range(count)]
        if self.integer:
            return [int(round(value)) for value in values]
        return [round(float(value), 10) for value in values]


@dataclass(frozen=True)
class ModelContext:
    frequency_hz: float
    omega: float
    layers: int
    layer_spacing_mm: float
    copper_thickness_um: float
    copper_thickness_oz: float
    copper_thickness_m: float
    temperature_c: float
    rho_t_ohm_m: float


@dataclass(frozen=True)
class SearchSpace:
    frequencies_mhz: list[float]
    tx_n: list[int]
    rx_n: list[int]
    tx_dout_mm: list[float]
    rx_dout_mm: list[float]
    distances_mm: list[float]
    load_voltages_v: list[float]
    load_powers_mw: list[float]
    copper_thicknesses_um: list[float]
    tx_width_mm: list[float]
    tx_spacing_mm: list[float]
    rx_width_mm: list[float]
    rx_spacing_mm: list[float]
    layer_spacings_mm: list[float]
    temperatures_c: list[float]


def rho_at_temperature(temperature_c: float) -> float:
    return RHO_20_OHM_M * (1.0 + RHO_TEMP_COEFF_PER_C * (temperature_c - 20.0))


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def values_from_config(data: Any, *, integer: bool = False) -> list[int] | list[float]:
    if isinstance(data, dict):
        if "values" in data:
            values = data["values"]
        else:
            values = GridSpec.from_config(data, integer=integer).values()
    elif isinstance(data, list):
        values = data
    else:
        values = [data]
    if integer:
        return [int(round(float(value))) for value in values]
    return [round(float(value), 10) for value in values]


def compact_number(value: float) -> str:
    text = f"{value:.10g}"
    return text.replace("-", "m").replace(".", "p")


def build_search_space(config: dict[str, Any]) -> SearchSpace:
    return SearchSpace(
        frequencies_mhz=list(values_from_config(config["frequency_MHz"])),
        tx_n=list(values_from_config(config["tx_N"], integer=True)),
        rx_n=list(values_from_config(config["rx_N"], integer=True)),
        tx_dout_mm=list(values_from_config(config["tx_Dout_mm"])),
        rx_dout_mm=list(values_from_config(config["rx_Dout_mm"])),
        distances_mm=list(values_from_config(config["distance_mm"])),
        load_voltages_v=list(values_from_config(config["rx_load"]["voltage_V"])),
        load_powers_mw=list(values_from_config(config["rx_load"]["power_mW"])),
        copper_thicknesses_um=list(values_from_config(config["copper_thickness_um"])),
        tx_width_mm=list(values_from_config(config["tx_trace_width_mm"])),
        tx_spacing_mm=list(values_from_config(config["tx_spacing_mm"])),
        rx_width_mm=list(values_from_config(config["rx_trace_width_mm"])),
        rx_spacing_mm=list(values_from_config(config["rx_spacing_mm"])),
        layer_spacings_mm=list(values_from_config(config["layer_spacing_mm"])),
        temperatures_c=list(values_from_config(config["temperature_C"])),
    )


def first_value_from_config(data: Any) -> float:
    return float(values_from_config(data)[0])


def build_model_context(
    config: dict[str, Any],
    copper_thickness_um: float,
    *,
    frequency_mhz: float | None = None,
    layer_spacing_mm: float | None = None,
    temperature_c: float | None = None,
) -> ModelContext:
    if frequency_mhz is None:
        frequency_mhz = first_value_from_config(config["frequency_MHz"])
    if layer_spacing_mm is None:
        layer_spacing_mm = first_value_from_config(config["layer_spacing_mm"])
    if temperature_c is None:
        temperature_c = first_value_from_config(config["temperature_C"])
    frequency_hz = float(frequency_mhz) * 1e6
    layers = int(config.get("layers", 2))
    if layers not in {1, 2}:
        raise ValueError("coil_optimizer currently supports only layers=1 or layers=2.")
    if copper_thickness_um <= 0:
        raise ValueError("copper_thickness_um must be positive.")
    return ModelContext(
        frequency_hz=frequency_hz,
        omega=2.0 * math.pi * frequency_hz,
        layers=layers,
        layer_spacing_mm=float(layer_spacing_mm),
        copper_thickness_um=float(copper_thickness_um),
        copper_thickness_oz=float(copper_thickness_um) / OZ_CU_TO_UM,
        copper_thickness_m=float(copper_thickness_um) * 1e-6,
        temperature_c=float(temperature_c),
        rho_t_ohm_m=rho_at_temperature(float(temperature_c)),
    )

