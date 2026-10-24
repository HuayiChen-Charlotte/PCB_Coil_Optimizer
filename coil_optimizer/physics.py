"""Analytic circular PCB coil model."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import special

from .models import ModelContext, compact_number


MU0 = 4.0 * math.pi * 1e-7
EPS0 = 8.8541878128e-12
MM_TO_M = 1e-3
M_TO_MM = 1000.0
H_TO_UH = 1e6
UH_TO_H = 1e-6
F_TO_PF = 1e12
PF_TO_F = 1e-12
EPS_EFFECTIVE = 0.9 * 1.0 + 0.1 * 4.4
QIAN_MODEL_VERSION = "qian_2018_circular_same_layer_v1"
QIAN_TWO_LAYER_MODEL_VERSION = "qian_2018_circular_two_layer_extension_v1"
QIAN_REFERENCE = (
    "G. Qian, Y. Cheng, G. Chen, and G. Wang, New AC Resistance Calculation of "
    "Printed Spiral Coils for Wireless Power Transfer, ISQED 2018."
)
QIAN_WIDTH_MIN_M = 0.1e-3
QIAN_WIDTH_MAX_M = 10.333e-3
QIAN_FREQUENCY_MIN_HZ = 0.1e6
QIAN_FREQUENCY_MAX_HZ = 1.6e9
QIAN_REFERENCE_COPPER_UM = 35.0
QIAN_WIDTH_QUADRATURE_POINTS = 5
_QIAN_GAUSS_X, _QIAN_GAUSS_W = np.polynomial.legendre.leggauss(QIAN_WIDTH_QUADRATURE_POINTS)
_QIAN_WIDTH_AVERAGE_WEIGHTS = _QIAN_GAUSS_W / 2.0


def loop_mutual_inductance_h(a_m: Any, b_m: Any, z_m: float) -> Any:
    a = np.asarray(a_m, dtype=float)
    b = np.asarray(b_m, dtype=float)
    m = 4.0 * a * b / ((a + b) ** 2 + z_m**2)
    m = np.clip(m, 0.0, np.nextafter(1.0, 0.0))
    k = np.sqrt(m)
    k_safe = np.where(k == 0.0, np.finfo(float).tiny, k)
    return MU0 * np.sqrt(a * b) * (
        (((2.0 / k_safe) - k_safe) * special.ellipk(m))
        - ((2.0 / k_safe) * special.ellipe(m))
    )


def circular_loop_normal_field_a_per_m(
    source_radius_m: Any,
    evaluation_radius_m: Any,
    z_m: Any,
) -> Any:
    """Normal H field from a 1 A circular filament at a cylindrical field point."""
    source_radius = np.asarray(source_radius_m, dtype=float)
    evaluation_radius = np.asarray(evaluation_radius_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    radial_sum_squared = (source_radius + evaluation_radius) ** 2 + z**2
    radial_difference_squared = (source_radius - evaluation_radius) ** 2 + z**2
    m = 4.0 * source_radius * evaluation_radius / radial_sum_squared
    m = np.clip(m, 0.0, np.nextafter(1.0, 0.0))
    return (
        special.ellipk(m)
        + (
            (source_radius**2 - evaluation_radius**2 - z**2)
            / radial_difference_squared
        )
        * special.ellipe(m)
    ) / (2.0 * math.pi * np.sqrt(radial_sum_squared))


def centerline_radii_m(turns: int, dout_m: float, width_m: float, spacing_m: float) -> np.ndarray:
    outer_center = dout_m / 2.0 - width_m / 2.0
    pitch = width_m + spacing_m
    return np.array([outer_center - i * pitch for i in range(turns)], dtype=float)


def physical_inner_diameter_m(turns: int, dout_m: float, width_m: float, spacing_m: float) -> float:
    return dout_m - (2.0 * turns * width_m) - (2.0 * (turns - 1) * spacing_m)


def webench_fill_inner_diameter_m(turns: int, dout_m: float, width_m: float, spacing_m: float) -> float:
    return dout_m - (2.0 * turns * (width_m + spacing_m))


def circular_self_inductance_h(turns: int, dout_m: float, din_m: float) -> float:
    davg_m = (dout_m + din_m) / 2.0
    phi = (dout_m - din_m) / (dout_m + din_m)
    return MU0 * turns**2 * davg_m / 2.0 * (math.log(2.46 / phi) + 0.2 * phi**2)


def single_layer_length_total_m(turns: int, dout_m: float, din_m: float) -> float:
    davg_m = (dout_m + din_m) / 2.0
    return math.pi * turns * davg_m


def two_layer_series_length_total_m(turns: int, dout_m: float, din_m: float) -> float:
    return 2.0 * single_layer_length_total_m(turns, dout_m, din_m)


def conductor_length_total_m(turns: int, dout_m: float, din_m: float, layers: int) -> float:
    if layers == 1:
        return single_layer_length_total_m(turns, dout_m, din_m)
    if layers == 2:
        return two_layer_series_length_total_m(turns, dout_m, din_m)
    raise ValueError("layers must be 1 or 2.")


def connection_name(layers: int) -> str:
    if layers == 1:
        return "single_layer"
    if layers == 2:
        return "series_aiding"
    raise ValueError("layers must be 1 or 2.")


def gap_length_m(radii_m: np.ndarray) -> float:
    if len(radii_m) < 2:
        return 0.0
    adjacent_mean_radii = (radii_m[:-1] + radii_m[1:]) / 2.0
    return float(np.sum(2.0 * math.pi * adjacent_mean_radii))


def cpar_book_f(radii_m: np.ndarray, copper_thickness_m: float, spacing_m: float) -> float:
    return EPS_EFFECTIVE * EPS0 * (copper_thickness_m / spacing_m) * gap_length_m(radii_m)


def cpar_fit_pf(turns: int, dout_mm: float, width_mm: float, spacing_mm: float) -> float:
    fill_factor = (2.0 * turns * width_mm + 2.0 * (turns - 1) * spacing_mm) / dout_mm
    return -8.51 + 0.6938 * dout_mm + 11.36 * fill_factor


def append_reason(existing: str, reason: str) -> str:
    return reason if not existing else existing + ";" + reason


def qian_phi(width_m: float, copper_thickness_m: float, skin_depth_m: float) -> float:
    """Qian et al. 2018 fitted rectangular-conductor proximity factor."""
    width_over_thickness = width_m / copper_thickness_m
    return (
        -0.998
        + 0.112 * width_over_thickness
        - 0.689 * copper_thickness_m / skin_depth_m
        + 0.543 * width_m / skin_depth_m
        + 0.002 * width_over_thickness**2
    )


def qian_model_domain(ctx: ModelContext, width_m: float) -> str:
    extrapolations = []
    if not (QIAN_WIDTH_MIN_M <= width_m <= QIAN_WIDTH_MAX_M):
        extrapolations.append("trace_width")
    if not (QIAN_FREQUENCY_MIN_HZ <= ctx.frequency_hz <= QIAN_FREQUENCY_MAX_HZ):
        extrapolations.append("frequency")
    if abs(ctx.copper_thickness_um - QIAN_REFERENCE_COPPER_UM) > 1.0:
        extrapolations.append("copper_thickness")
    if not extrapolations:
        return "qian_2018_fit_range_near_35um_copper"
    return "qian_2018_extrapolated:" + ",".join(extrapolations)


def qian_circular_proximity_components(
    ctx: ModelContext,
    *,
    radii_m: np.ndarray,
    width_m: float,
    skin_depth_m: float,
) -> dict[str, float | str]:
    """Compute paper-scoped same-layer and uncalibrated two-layer Qian losses.

    The Qian fit converts the square of the normal external H field into loss.
    Each target trace is averaged across its width.  The same-layer result sums
    each physical layer independently.  The two-layer extension additionally
    includes the normal field from the aligned series-aiding layer.
    """
    radii = np.asarray(radii_m, dtype=float)
    phi = qian_phi(width_m, ctx.copper_thickness_m, skin_depth_m)
    offsets = _QIAN_GAUSS_X * width_m / 2.0
    evaluation_radii = radii[:, None, None] + offsets[None, :, None]
    source_radii = radii[None, None, :]
    target_indexes = np.arange(len(radii))[:, None, None]
    source_indexes = np.arange(len(radii))[None, None, :]
    other_turn_mask = target_indexes != source_indexes

    # Give the excluded self filament a harmless z separation before masking
    # so the circular-filament expression is never evaluated at its singularity.
    same_layer_z = np.where(other_turn_mask, 0.0, 1.0)
    same_layer_fields = circular_loop_normal_field_a_per_m(
        source_radii,
        evaluation_radii,
        same_layer_z,
    )
    h_same = np.sum(np.where(other_turn_mask, same_layer_fields, 0.0), axis=2)
    if ctx.layers == 1:
        h_other = np.zeros_like(h_same)
    else:
        h_other = np.sum(
            circular_loop_normal_field_a_per_m(
                source_radii,
                evaluation_radii,
                ctx.layer_spacing_mm * MM_TO_M,
            ),
            axis=2,
        )

    mean_h_same_squared = np.sum(
        h_same**2 * _QIAN_WIDTH_AVERAGE_WEIGHTS[None, :], axis=1
    )
    mean_h_other_squared = np.sum(
        h_other**2 * _QIAN_WIDTH_AVERAGE_WEIGHTS[None, :], axis=1
    )
    mean_h_cross = np.sum(
        2.0 * h_same * h_other * _QIAN_WIDTH_AVERAGE_WEIGHTS[None, :], axis=1
    )
    turn_lengths_m = 2.0 * math.pi * radii
    conductivity_s_per_m = 1.0 / ctx.rho_t_ohm_m
    loss_factor = ctx.layers * (4.0 * math.pi / conductivity_s_per_m) * phi

    rprox_same = loss_factor * float(np.dot(turn_lengths_m, mean_h_same_squared))
    rprox_other_only = loss_factor * float(np.dot(turn_lengths_m, mean_h_other_squared))
    rprox_cross = loss_factor * float(np.dot(turn_lengths_m, mean_h_cross))
    rprox_two_layer = rprox_same + rprox_other_only + rprox_cross
    return {
        "qian_phi": float(phi),
        "rac_model_version": QIAN_MODEL_VERSION,
        "rac_two_layer_extension_version": QIAN_TWO_LAYER_MODEL_VERSION,
        "rac_model_domain": qian_model_domain(ctx, width_m),
        "Rprox_qian_same_layer_ohm": rprox_same,
        "Rprox_qian_other_layer_only_ohm": rprox_other_only,
        "Rprox_qian_cross_term_ohm": rprox_cross,
        "Rprox_qian_two_layer_extension_ohm": rprox_two_layer,
    }


def ac_resistance_components(
    ctx: ModelContext,
    *,
    length_total_m: float,
    width_m: float,
    spacing_m: float,
) -> tuple[float, float, float, float, float, float]:
    rdc = ctx.rho_t_ohm_m * length_total_m / (width_m * ctx.copper_thickness_m)
    delta = math.sqrt(2.0 * ctx.rho_t_ohm_m / (MU0 * ctx.omega))
    rskin_raw = (
        rdc
        * ctx.copper_thickness_m
        / (delta * (1.0 - math.exp(-ctx.copper_thickness_m / delta)))
        * (1.0 / (1.0 + ctx.copper_thickness_m / width_m))
    )
    rskin = max(rdc, rskin_raw)
    omega_crit = (3.1 / MU0) * ((spacing_m + width_m) / width_m**2) * (
        ctx.rho_t_ohm_m / ctx.copper_thickness_m
    )
    rprox = rdc / 10.0 * (ctx.omega / omega_crit) ** 2
    return rdc, rskin, rprox, rskin + rprox, delta, omega_crit


def build_coil(
    *,
    side: str,
    index_tuple: tuple[int, int, int, int],
    turns: int,
    dout_mm: float,
    width_mm: float,
    spacing_mm: float,
    ctx: ModelContext,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    dout_m = dout_mm * MM_TO_M
    width_m = width_mm * MM_TO_M
    spacing_m = spacing_mm * MM_TO_M
    din_m = physical_inner_diameter_m(turns, dout_m, width_m, spacing_m)
    fill_din_m = webench_fill_inner_diameter_m(turns, dout_m, width_m, spacing_m)
    fill_factor = fill_din_m / dout_m if dout_m > 0.0 else -math.inf
    coil_id = (
        f"{side}_N{turns}_D{compact_number(dout_mm)}_"
        f"W{compact_number(width_mm)}_S{compact_number(spacing_mm)}_"
        f"Cu{compact_number(ctx.copper_thickness_oz)}oz_"
        f"f{compact_number(ctx.frequency_hz / 1e6)}MHz_"
        f"h{compact_number(ctx.layer_spacing_mm)}mm_"
        f"T{compact_number(ctx.temperature_c)}C"
    )
    base = {
        "coil_id": coil_id,
        "side": side,
        "index_tuple": index_tuple,
        "N": turns,
        "Dout_mm": dout_mm,
        "Dout_mils": dout_mm / 0.0254,
        "trace_width_mm": width_mm,
        "trace_width_mils": width_mm / 0.0254,
        "spacing_mm": spacing_mm,
        "spacing_mils": spacing_mm / 0.0254,
        "Din_mm": din_m * M_TO_MM,
        "Din_mils": din_m * M_TO_MM / 0.0254,
        "Din_webench_fill_mm": fill_din_m * M_TO_MM,
        "fill_factor_webench": fill_factor,
        "layers": ctx.layers,
        "connection": connection_name(ctx.layers),
        "layer_spacing_mm": ctx.layer_spacing_mm,
        "copper_thickness_um": ctx.copper_thickness_um,
        "copper_thickness_oz": ctx.copper_thickness_oz,
    }
    reject = ""
    if turns <= 0:
        reject = append_reason(reject, "invalid_turn_count")
    if dout_m <= 0.0 or din_m <= 0.0:
        reject = append_reason(reject, "invalid_geometry")
    if fill_factor < float(constraints["fill_factor_min"]):
        reject = append_reason(reject, "fill_factor_below_min")
    if width_mm + 1e-12 < float(constraints.get("trace_width_min_mm", 0.1016)):
        reject = append_reason(reject, "trace_width_below_min")
    if spacing_mm + 1e-12 < float(constraints.get("spacing_min_mm", 0.1016)):
        reject = append_reason(reject, "spacing_below_min")
    if width_m <= 0.0 or spacing_m <= 0.0:
        reject = append_reason(reject, "invalid_width_or_spacing")
    if reject:
        return {**base, "status": "rejected", "reject_reason": reject, "radii_m": np.array([])}

    radii = centerline_radii_m(turns, dout_m, width_m, spacing_m)
    if len(radii) == 0 or np.any(radii <= 0.0):
        return {
            **base,
            "status": "rejected",
            "reject_reason": "invalid_centerline_radii",
            "radii_m": np.array([]),
        }

    l_self_h = circular_self_inductance_h(turns, dout_m, din_m)
    if ctx.layers == 1:
        m_interlayer_h = 0.0
        l_total_h = l_self_h
    elif ctx.layers == 2:
        m_interlayer_h = float(
            loop_mutual_inductance_h(
                radii[:, None],
                radii[None, :],
                ctx.layer_spacing_mm * MM_TO_M,
            ).sum()
        )
        l_total_h = 2.0 * l_self_h + 2.0 * m_interlayer_h
    else:
        raise ValueError("layers must be 1 or 2.")
    length_total_m = conductor_length_total_m(turns, dout_m, din_m, ctx.layers)
    rdc, rskin, rprox_legacy, rac_legacy, skin_depth_m, omega_crit = ac_resistance_components(
        ctx,
        length_total_m=length_total_m,
        width_m=width_m,
        spacing_m=spacing_m,
    )
    qian = qian_circular_proximity_components(
        ctx,
        radii_m=radii,
        width_m=width_m,
        skin_depth_m=skin_depth_m,
    )
    if not math.isfinite(float(qian["qian_phi"])) or float(qian["qian_phi"]) <= 0.0:
        return {
            **base,
            **qian,
            "status": "rejected",
            "reject_reason": "qian_phi_nonpositive",
            "radii_m": radii,
        }
    rprox_same = float(qian["Rprox_qian_same_layer_ohm"])
    rprox_two_layer = float(qian["Rprox_qian_two_layer_extension_ohm"])
    if not all(math.isfinite(value) and value >= 0.0 for value in (rprox_same, rprox_two_layer)):
        return {
            **base,
            **qian,
            "status": "rejected",
            "reject_reason": "qian_proximity_nonfinite_or_negative",
            "radii_m": radii,
        }
    rac_same = rskin + rprox_same
    rac_two_layer = rskin + rprox_two_layer
    q_same = ctx.omega * l_total_h / rac_same
    q_two_layer = ctx.omega * l_total_h / rac_two_layer
    q_legacy = ctx.omega * l_total_h / rac_legacy

    c_total_f = 1.0 / (ctx.omega**2 * l_total_h)
    cpar_webench_f = (dout_mm / 3.0) * PF_TO_F
    cpar_book_value_f = cpar_book_f(radii, ctx.copper_thickness_m, spacing_m)
    cpar_fit_value_f = cpar_fit_pf(turns, dout_mm, width_mm, spacing_mm) * PF_TO_F
    c_sensor_webench_f = c_total_f - cpar_webench_f
    c_sensor_book_f = c_total_f - cpar_book_value_f
    c_sensor_fit_f = c_total_f - cpar_fit_value_f
    fsrf_webench_hz = 1.0 / (2.0 * math.pi * math.sqrt(l_total_h * cpar_webench_f))
    fsrf_book_hz = (
        math.inf
        if cpar_book_value_f <= 0.0
        else 1.0 / (2.0 * math.pi * math.sqrt(l_total_h * cpar_book_value_f))
    )
    fsrf_fit_hz = (
        math.inf
        if cpar_fit_value_f <= 0.0
        else 1.0 / (2.0 * math.pi * math.sqrt(l_total_h * cpar_fit_value_f))
    )
    if ctx.layers == 1:
        active_cpar_model = "webench"
        cpar_active_f = cpar_webench_f
        c_sensor_active_f = c_sensor_webench_f
        fsrf_active_hz = fsrf_webench_hz
    elif ctx.layers == 2:
        active_cpar_model = "fit"
        cpar_active_f = cpar_fit_value_f
        c_sensor_active_f = c_sensor_fit_f
        fsrf_active_hz = fsrf_fit_hz
    else:
        raise ValueError("layers must be 1 or 2.")

    c_min = float(constraints["C_sensor_min_pF"])
    c_max = float(constraints["C_sensor_max_pF"])
    min_srf_hz = float(constraints["minimum_SRF_ratio"]) * ctx.frequency_hz
    if not (c_min <= c_sensor_active_f * F_TO_PF <= c_max):
        reject = append_reason(reject, "C_sensor_activeCpar_out_of_range")
    if fsrf_active_hz < min_srf_hz:
        reject = append_reason(reject, "fsrf_activeCpar_below_min")

    return {
        **base,
        "L_self_layer_uH": l_self_h * H_TO_UH,
        "M_interlayer_uH": m_interlayer_h * H_TO_UH,
        "L_total_uH": l_total_h * H_TO_UH,
        "conductor_length_m": length_total_m,
        "Rdc_ohm": rdc,
        "Rskin_ohm": rskin,
        "Rskin_jow_ohm": rskin,
        **qian,
        "Rac_qian_same_layer_ohm": rac_same,
        "Q_qian_same_layer": q_same,
        "Rac_qian_two_layer_extension_ohm": rac_two_layer,
        "Q_qian_two_layer_extension": q_two_layer,
        "Rprox_legacy_kuhn_ohm": rprox_legacy,
        "Rac_legacy_kuhn_ohm": rac_legacy,
        "Q_legacy_kuhn": q_legacy,
        "Rac_model_ohm": rac_same,
        "Q_model": q_same,
        # Compatibility aliases: existing consumers continue to use the new
        # selected model without silently retaining the former Kuhn result.
        "Rprox_ohm": rprox_same,
        "Rac_theorybook_ohm": rac_same,
        "Q_theorybook": q_same,
        "C_total_pF": c_total_f * F_TO_PF,
        "active_cpar_model": active_cpar_model,
        "Cpar_active_pF": cpar_active_f * F_TO_PF,
        "C_sensor_activeCpar_pF": c_sensor_active_f * F_TO_PF,
        "fsrf_activeCpar_MHz": fsrf_active_hz / 1e6,
        "Cpar_fit_pF": cpar_fit_value_f * F_TO_PF,
        "C_sensor_fitCpar_pF": c_sensor_fit_f * F_TO_PF,
        "fsrf_fitCpar_MHz": fsrf_fit_hz / 1e6,
        "Cpar_webench_pF": cpar_webench_f * F_TO_PF,
        "C_sensor_webenchCpar_pF": c_sensor_webench_f * F_TO_PF,
        "fsrf_webenchCpar_MHz": fsrf_webench_hz / 1e6,
        "Cpar_book_pF": cpar_book_value_f * F_TO_PF,
        "C_sensor_bookCpar_pF": c_sensor_book_f * F_TO_PF,
        "fsrf_bookCpar_MHz": fsrf_book_hz / 1e6,
        "skin_depth_mm": skin_depth_m * M_TO_MM,
        "omega_crit_rad_s": omega_crit,
        "status": "ok" if not reject else "rejected",
        "reject_reason": reject,
        "radii_m": radii,
    }


def mutual_between_coils_h(tx: dict[str, Any], rx: dict[str, Any], distance_mm: float, ctx: ModelContext) -> float:
    radius_tx = tx["radii_m"][:, None]
    radius_rx = rx["radii_m"][None, :]
    z_m = distance_mm * MM_TO_M
    if ctx.layers == 1:
        turn_mutual = loop_mutual_inductance_h(radius_tx, radius_rx, z_m)
    elif ctx.layers == 2:
        h_m = ctx.layer_spacing_mm * MM_TO_M
        turn_mutual = (
            loop_mutual_inductance_h(radius_tx, radius_rx, z_m)
            + 2.0 * loop_mutual_inductance_h(radius_tx, radius_rx, z_m + h_m)
            + loop_mutual_inductance_h(radius_tx, radius_rx, z_m + 2.0 * h_m)
        )
    else:
        raise ValueError("layers must be 1 or 2.")
    return float(turn_mutual.sum())


def finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def load_resistance_ohm(voltage_v: float, power_mw: float) -> float:
    if power_mw <= 0.0:
        raise ValueError("target load power must be positive.")
    return voltage_v**2 / (power_mw / 1000.0)

