import math
import unittest

from coil_optimizer.io import candidate_fieldnames
from coil_optimizer.models import (
    build_model_context,
    default_config,
    values_from_config,
)
from coil_optimizer.optimizer import loaded_efficiency_value, pair_metrics
from coil_optimizer.physics import (
    circular_loop_normal_field_a_per_m,
    build_coil,
    cpar_fit_pf,
    loop_mutual_inductance_h,
    qian_phi,
    single_layer_length_total_m,
)


def relaxed_config(frequency_mhz=40.68, *, layers=2):
    config = default_config()
    config["frequency_MHz"] = frequency_mhz
    config["layers"] = layers
    config["constraints"].update(
        {
            "fill_factor_min": 0.0,
            "C_sensor_min_pF": -1e9,
            "C_sensor_max_pF": 1e9,
            "minimum_SRF_ratio": 0.0,
            "norm_Dcoil_percent_min": None,
            "norm_Dcoil_percent_max": None,
        }
    )
    return config


def tx5_rx3_coils(copper_um=34.7, *, layers=2):
    config = relaxed_config(layers=layers)
    context = build_model_context(config, copper_um)
    tx = build_coil(
        side="tx",
        index_tuple=(0, 0, 0, 0),
        turns=5,
        dout_mm=15.0,
        width_mm=0.82,
        spacing_mm=0.13,
        ctx=context,
        constraints=config["constraints"],
    )
    rx = build_coil(
        side="rx",
        index_tuple=(0, 0, 0, 0),
        turns=3,
        dout_mm=8.0,
        width_mm=0.47,
        spacing_mm=0.11,
        ctx=context,
        constraints=config["constraints"],
    )
    return config, context, tx, rx


class CoilOptimizerRacTests(unittest.TestCase):
    def test_defaults_use_one_ounce_and_srf_ratio_four(self):
        config = default_config()
        self.assertEqual(config["copper_thickness_um"], 34.7)
        self.assertEqual(config["constraints"]["minimum_SRF_ratio"], 4.0)
        self.assertEqual(config["constraints"]["norm_Dcoil_percent_min"], 40.0)
        self.assertEqual(config["constraints"]["norm_Dcoil_percent_max"], 50.0)
        self.assertEqual(
            values_from_config({"min": 34.7, "max": 173.5, "step": 34.7}),
            [34.7, 69.4, 104.1, 138.8, 173.5],
        )
    def test_circular_field_matches_on_axis_closed_form(self):
        radius = 7.0e-3
        z = 0.6e-3
        expected = radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)
        actual = float(circular_loop_normal_field_a_per_m(radius, 0.0, z))
        self.assertAlmostEqual(actual, expected, places=12)

    def test_context_accepts_one_or_two_layers_only(self):
        one_layer = relaxed_config(layers=1)
        two_layer = relaxed_config(layers=2)
        self.assertEqual(build_model_context(one_layer, 34.7).layers, 1)
        self.assertEqual(build_model_context(two_layer, 34.7).layers, 2)
        invalid = relaxed_config(layers=3)
        with self.assertRaisesRegex(ValueError, "layers=1 or layers=2"):
            build_model_context(invalid, 34.7)

    def test_qian_phi_tx5_40p68mhz(self):
        config = relaxed_config()
        context = build_model_context(config, 34.7)
        delta = math.sqrt(
            2.0 * context.rho_t_ohm_m / (4.0 * math.pi * 1e-7 * context.omega)
        )
        self.assertAlmostEqual(
            qian_phi(0.82e-3, context.copper_thickness_m, delta),
            43.5631715328,
            places=8,
        )

    def test_tx5_rx3_qian_regression_values(self):
        _, _, tx, rx = tx5_rx3_coils()
        self.assertEqual(tx["status"], "ok")
        self.assertEqual(rx["status"], "ok")
        self.assertAlmostEqual(tx["Rac_qian_same_layer_ohm"], 0.9512911928, places=8)
        self.assertAlmostEqual(
            tx["Rac_qian_two_layer_extension_ohm"], 1.7018220089, places=8
        )
        self.assertAlmostEqual(rx["Rac_qian_same_layer_ohm"], 0.5035222169, places=8)
        self.assertAlmostEqual(
            rx["Rac_qian_two_layer_extension_ohm"], 0.6615133176, places=8
        )
        for coil in (tx, rx):
            self.assertGreaterEqual(coil["Rskin_ohm"], coil["Rdc_ohm"])
            self.assertAlmostEqual(
                coil["Rprox_qian_two_layer_extension_ohm"],
                coil["Rprox_qian_same_layer_ohm"]
                + coil["Rprox_qian_other_layer_only_ohm"]
                + coil["Rprox_qian_cross_term_ohm"],
                places=12,
            )
            self.assertEqual(coil["Rac_model_ohm"], coil["Rac_qian_same_layer_ohm"])
            self.assertEqual(coil["Q_model"], coil["Q_qian_same_layer"])
            self.assertEqual(coil["Rprox_ohm"], coil["Rprox_qian_same_layer_ohm"])
            self.assertEqual(coil["Rac_theorybook_ohm"], coil["Rac_model_ohm"])
            self.assertEqual(coil["Q_theorybook"], coil["Q_model"])
            self.assertEqual(coil["active_cpar_model"], "fit")
            self.assertEqual(coil["Cpar_active_pF"], coil["Cpar_fit_pF"])
            self.assertEqual(
                coil["C_sensor_activeCpar_pF"],
                coil["C_sensor_fitCpar_pF"],
            )
            self.assertEqual(coil["fsrf_activeCpar_MHz"], coil["fsrf_fitCpar_MHz"])

    def test_one_layer_geometry_uses_single_layer_inductance_length_and_webench_cpar(self):
        config, context, tx, _ = tx5_rx3_coils(layers=1)
        self.assertEqual(tx["status"], "ok")
        self.assertEqual(tx["layers"], 1)
        self.assertEqual(tx["connection"], "single_layer")
        self.assertAlmostEqual(tx["M_interlayer_uH"], 0.0, places=12)
        self.assertAlmostEqual(tx["L_total_uH"], tx["L_self_layer_uH"], places=12)
        expected_length = single_layer_length_total_m(
            tx["N"],
            tx["Dout_mm"] * 1e-3,
            tx["Din_mm"] * 1e-3,
        )
        self.assertAlmostEqual(tx["conductor_length_m"], expected_length, places=12)
        self.assertEqual(tx["active_cpar_model"], "webench")
        self.assertEqual(tx["Cpar_active_pF"], tx["Cpar_webench_pF"])
        self.assertEqual(
            tx["C_sensor_activeCpar_pF"],
            tx["C_sensor_webenchCpar_pF"],
        )
        self.assertEqual(tx["fsrf_activeCpar_MHz"], tx["fsrf_webenchCpar_MHz"])
        self.assertNotEqual(tx["Cpar_active_pF"], tx["Cpar_fit_pF"])
        self.assertEqual(config["layers"], context.layers)

    def test_one_layer_pair_mutual_uses_single_gap_only(self):
        config, context, tx, rx = tx5_rx3_coils(layers=1)
        pair, reject = pair_metrics(tx, rx, 4.75, 0, context, config["constraints"])
        self.assertIsNone(reject)
        expected_m_h = float(
            loop_mutual_inductance_h(
                tx["radii_m"][:, None],
                rx["radii_m"][None, :],
                4.75e-3,
            ).sum()
        )
        self.assertAlmostEqual(pair["M_uH"], expected_m_h * 1e6, places=12)
        self.assertEqual(pair["tx_connection"], "single_layer")
        self.assertEqual(pair["rx_connection"], "single_layer")
        self.assertEqual(pair["tx_active_cpar_model"], "webench")
        self.assertEqual(pair["rx_active_cpar_model"], "webench")

    def test_five_ounce_remains_available_and_is_marked_extrapolated(self):
        _, _, tx, _ = tx5_rx3_coils(173.5)
        self.assertEqual(tx["status"], "ok")
        self.assertIn("copper_thickness", tx["rac_model_domain"])
        self.assertTrue(math.isfinite(tx["Rac_model_ohm"]))
        self.assertGreater(tx["Rac_model_ohm"], 0.0)

    def test_pair_and_loaded_efficiency_use_canonical_model_fields(self):
        config, context, tx, rx = tx5_rx3_coils()
        tx["Rac_theorybook_ohm"] = 999.0
        tx["Q_theorybook"] = 1e-9
        rx["Rac_theorybook_ohm"] = 999.0
        rx["Q_theorybook"] = 1e-9
        pair, reject = pair_metrics(
            tx,
            rx,
            4.75,
            0,
            context,
            config["constraints"],
        )
        self.assertIsNone(reject)
        self.assertEqual(pair["tx_Q_legacy_kuhn"], tx["Q_legacy_kuhn"])
        self.assertEqual(pair["rx_Q_legacy_kuhn"], rx["Q_legacy_kuhn"])
        self.assertAlmostEqual(
            pair["tx_Q_legacy_kuhn"],
            context.omega * pair["tx_L_total_uH"] * 1e-6 / pair["tx_Rac_legacy_kuhn_ohm"],
            places=12,
        )
        self.assertAlmostEqual(
            pair["rx_Q_legacy_kuhn"],
            context.omega * pair["rx_L_total_uH"] * 1e-6 / pair["rx_Rac_legacy_kuhn_ohm"],
            places=12,
        )
        fields = candidate_fieldnames()
        self.assertIn("tx_Q_legacy_kuhn", fields)
        self.assertIn("rx_Q_legacy_kuhn", fields)
        self.assertIn("tx_connection", fields)
        self.assertIn("rx_connection", fields)
        self.assertIn("tx_C_total_pF", fields)
        self.assertIn("rx_C_total_pF", fields)
        for prefix in ("tx", "rx"):
            self.assertIn(f"{prefix}_active_cpar_model", fields)
            self.assertIn(f"{prefix}_Cpar_active_pF", fields)
            self.assertIn(f"{prefix}_C_sensor_activeCpar_pF", fields)
            self.assertIn(f"{prefix}_fsrf_activeCpar_MHz", fields)
            self.assertIn(f"{prefix}_Cpar_fit_pF", fields)
            self.assertIn(f"{prefix}_C_sensor_fitCpar_pF", fields)
            self.assertIn(f"{prefix}_fsrf_fitCpar_MHz", fields)
            self.assertIn(f"{prefix}_Cpar_webench_pF", fields)
            self.assertIn(f"{prefix}_Cpar_book_pF", fields)
        self.assertEqual(pair["tx_C_total_pF"], tx["C_total_pF"])
        self.assertEqual(pair["rx_C_total_pF"], rx["C_total_pF"])
        self.assertEqual(pair["tx_Cpar_fit_pF"], tx["Cpar_fit_pF"])
        self.assertEqual(pair["rx_Cpar_fit_pF"], rx["Cpar_fit_pF"])
        self.assertEqual(pair["tx_fsrf_fitCpar_MHz"], tx["fsrf_fitCpar_MHz"])
        self.assertEqual(pair["rx_fsrf_fitCpar_MHz"], rx["fsrf_fitCpar_MHz"])
        self.assertEqual(pair["tx_Cpar_active_pF"], tx["Cpar_active_pF"])
        self.assertEqual(pair["rx_Cpar_active_pF"], rx["Cpar_active_pF"])
        self.assertEqual(pair["tx_fsrf_activeCpar_MHz"], tx["fsrf_activeCpar_MHz"])
        self.assertEqual(pair["rx_fsrf_activeCpar_MHz"], rx["fsrf_activeCpar_MHz"])
        expected_chi = pair["k"] ** 2 * tx["Q_model"] * rx["Q_model"]
        self.assertAlmostEqual(pair["chi_k2Q1Q2"], expected_chi, places=12)
        efficiency = loaded_efficiency_value(pair, 1.8, 40.0, context)
        self.assertGreater(efficiency, 0.0)
        self.assertLessEqual(efficiency, pair["eta_max"])

    def test_fitted_cpar_and_srf_formula(self):
        _, context, tx, _ = tx5_rx3_coils()
        expected_cpar = cpar_fit_pf(5, 15.0, 0.82, 0.13)
        self.assertAlmostEqual(tx["Cpar_fit_pF"], expected_cpar, places=12)
        expected_fsrf_mhz = (
            1.0
            / (
                2.0
                * math.pi
                * math.sqrt(tx["L_total_uH"] * 1e-6 * tx["Cpar_fit_pF"] * 1e-12)
            )
            / 1e6
        )
        self.assertAlmostEqual(tx["fsrf_fitCpar_MHz"], expected_fsrf_mhz, places=12)
        self.assertAlmostEqual(
            tx["C_sensor_fitCpar_pF"],
            tx["C_total_pF"] - tx["Cpar_fit_pF"],
            places=12,
        )
        self.assertGreater(tx["fsrf_webenchCpar_MHz"], context.frequency_hz / 1e6)
        self.assertGreater(tx["fsrf_bookCpar_MHz"], tx["fsrf_webenchCpar_MHz"])

    def test_srf_rejection_uses_active_cpar_only(self):
        config = relaxed_config()
        config["constraints"]["minimum_SRF_ratio"] = 1.5
        context = build_model_context(config, 34.7)
        tx = build_coil(
            side="tx",
            index_tuple=(0, 0, 0, 0),
            turns=5,
            dout_mm=15.0,
            width_mm=0.82,
            spacing_mm=0.13,
            ctx=context,
            constraints=config["constraints"],
        )
        self.assertEqual(tx["status"], "rejected")
        self.assertIn("fsrf_activeCpar_below_min", tx["reject_reason"])
        self.assertNotIn("fsrf_fitCpar_below_min", tx["reject_reason"])
        self.assertNotIn("fsrf_webenchCpar_below_min", tx["reject_reason"])
        self.assertNotIn("fsrf_bookCpar_below_min", tx["reject_reason"])

    def test_c_sensor_rejection_uses_active_cpar_only(self):
        config = relaxed_config()
        config["constraints"]["C_sensor_min_pF"] = 529.0
        config["constraints"]["C_sensor_max_pF"] = 532.0
        context = build_model_context(config, 34.7)
        tx = build_coil(
            side="tx",
            index_tuple=(0, 0, 0, 0),
            turns=5,
            dout_mm=15.0,
            width_mm=0.82,
            spacing_mm=0.13,
            ctx=context,
            constraints=config["constraints"],
        )
        self.assertEqual(tx["status"], "rejected")
        self.assertIn("C_sensor_activeCpar_out_of_range", tx["reject_reason"])
        self.assertNotIn("C_sensor_fitCpar_out_of_range", tx["reject_reason"])
        self.assertNotIn("C_sensor_webenchCpar_out_of_range", tx["reject_reason"])
        self.assertNotIn("C_sensor_bookCpar_out_of_range", tx["reject_reason"])


if __name__ == "__main__":
    unittest.main()

