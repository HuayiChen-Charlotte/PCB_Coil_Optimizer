# PCB Coil Optimizer

PCB Coil Optimizer is a Python tool for analytical screening and optimization of coaxial circular PCB coil pairs. It searches manufacturing-compatible transmitter and receiver geometries, estimates their electrical parameters, and ranks valid pairs by ideal link efficiency and fixed-load efficiency.

The project is intended for rapid design-space exploration before electromagnetic simulation, PCB layout, and measurement. It is not a replacement for a field solver or hardware validation.

## Features

- Circular single-layer and two-layer series-aiding PCB coils
- Configurable turns, outer diameters, trace widths, spacings, copper thicknesses, frequencies, separation distances, temperatures, and receiver loads
- Analytical estimates of self-inductance, mutual inductance, coupling coefficient, DC resistance, AC resistance, quality factor, parasitic capacitance, and self-resonant frequency
- Ranking by ideal maximum efficiency (`eta_max`) and series-equivalent fixed-load efficiency (`loaded_efficiency`)
- Grid refinement, pair pruning, and local fine expansion for large discrete design spaces
- Reproducible JSON configuration and CSV/JSON output files
- Explicit model-version and validity-domain fields in exported results

## Installation

PCB Coil Optimizer requires Python 3.10 or newer.

```bash
git clone https://github.com/HuayiChen-Charlotte/PCB_Coil_Optimizer.git
cd PCB_Coil_Optimizer
python -m venv .venv
python -m pip install -e .
```

The runtime dependencies are NumPy and SciPy.

## Quick start

Run the small example first. It is designed to complete quickly and verify that the installation works.

```bash
pcb-coil-opt --config examples/smoke_test.json
```

The module form is equivalent:

```bash
python -m coil_optimizer.cli --config examples/smoke_test.json
```

Results are written to `outputs/smoke_test/` relative to the current working directory.

To generate a configuration containing all default options:

```bash
pcb-coil-opt --write-example-config --config optimizer_config.json
```

The larger `examples/full_search.json` file demonstrates a broader search. It can require substantially more time and memory than the smoke test.

## Python API

```python
from pathlib import Path

from coil_optimizer import load_config, run

config = load_config(Path("examples/smoke_test.json"))
config["output_dir"] = "outputs/api_example"
summary = run(config)

best_loaded = summary["top_loaded_efficiency_candidate"]
best_ideal = summary["top_eta_max_candidate"]

print(best_loaded["loaded_efficiency"])
print(best_ideal["eta_max"])
```

## Configuration

Each scalar sweep field accepts one of three forms:

```json
6.78
```

```json
{"min": 5.0, "max": 20.0, "step": 0.5}
```

```json
{"values": [6.78, 13.56, 27.12]}
```

The main inputs are:

| Field | Meaning |
| --- | --- |
| `frequency_MHz` | Operating frequency or frequency sweep |
| `layers` | `1` for a single layer or `2` for aligned series-aiding layers |
| `tx_Dout_mm`, `rx_Dout_mm` | Physical outer diameters |
| `tx_N`, `rx_N` | Turn counts |
| `tx_trace_width_mm`, `rx_trace_width_mm` | Copper trace widths |
| `tx_spacing_mm`, `rx_spacing_mm` | Edge-to-edge trace spacings |
| `distance_mm` | Closest copper-current-plane separation between TX and RX |
| `layer_spacing_mm` | Current-plane spacing between layers of a two-layer coil |
| `copper_thickness_um` | Copper thickness; 34.7 micrometres represents nominal 1 oz copper |
| `temperature_C` | Copper temperature used for resistivity correction |
| `rx_load.voltage_V` | Target receiver load voltage |
| `rx_load.power_mW` | Target receiver load power |

The fixed load is represented by

```text
Rload = Vload^2 / Pload
```

Constraints can reject coils or pairs based on manufacturing dimensions, fill factor, compensation-capacitor range, SRF margin, coupling coefficient, and normalized distance.

## Model overview

The optimizer treats each winding as concentric circular filaments and operates in a low-frequency, quasi-static regime.

Single-layer inductance is estimated with a circular planar-spiral expression. For two aligned identical layers connected in series-aiding form, the total inductance is

```text
Ltotal = 2 Lself + 2 Minterlayer
```

For two two-layer coils, where `z` is the closest current-plane separation and `h` is the layer spacing, pair mutual inductance is

```text
Mpair = M(z) + 2 M(z + h) + M(z + 2h)
```

The coupling and link figure of merit are

```text
k   = M / sqrt(Ltx Lrx)
chi = k^2 Qtx Qrx
```

The ideal maximum-efficiency metric is

```text
eta_max = chi / (1 + sqrt(1 + chi))^2
```

The corresponding optimal series-equivalent receiver load is

```text
Rload_opt = Rrx sqrt(1 + chi)
```

`loaded_efficiency` evaluates the same analytical link using the load selected in the configuration. It is not interchangeable with `eta_max`, rectifier efficiency, or measured power-transfer efficiency.

## AC resistance and SRF

The default ranking resistance is

```text
Rac_model = Rskin + Rprox_Qian_same_layer
```

The proximity-loss term follows the circular-coil implementation of the fitted formulation in:

> G. Qian, Y. Cheng, G. Chen, and G. Wang, "New AC Resistance Calculation of Printed Spiral Coils for Wireless Power Transfer," ISQED, 2018.

The exported results identify whether geometry, frequency, or copper thickness lies outside the fitted model domain. The code also reports an uncalibrated two-layer proximity-loss extension, but that diagnostic is not used for default ranking or efficiency calculations.

The active parasitic-capacitance policy currently uses a WEBENCH-style estimate for single-layer coils and an empirical fitted estimate for two-layer coils. SRF-based filtering should therefore be treated as screening rather than final validation.

## Search strategy

The default `grid_refine` search:

1. Builds reusable TX and RX coil libraries.
2. Evaluates a coarse subset of candidate pairs.
3. Refines neighborhoods around the best coarse pairs.
4. Optionally prunes the retained pair set.
5. Optionally performs a finer local geometry and distance expansion.
6. Ranks candidates independently by `eta_max` and `loaded_efficiency`.

This strategy is intended to make large searches practical. Unless the configured grid is evaluated exhaustively, the result is not proof of a global optimum.

## Outputs

Each run can produce:

- `config_used.json`: resolved input configuration and derived load values
- `candidates_loaded_efficiency.csv`: candidates ranked by fixed-load efficiency
- `candidates_eta_max.csv`: candidates ranked by ideal maximum efficiency
- `pre_local_fine_*.csv` and `post_local_fine_*.csv`: optional search-stage snapshots
- `per_scenario_best_*.csv`: optional best candidates for each input scenario
- `rejected.csv`: optional rejected geometry and pair records
- `summary.json`: model provenance, counts, checks, paths, and top candidates

## Verification

Run the tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover model defaults, circular-loop field behavior, one- and two-layer geometry, Qian-model regression values, mutual inductance, SRF filtering, efficiency identities, exported fields, and a complete smoke optimization.

## Important limitations

The calculations are analytical screening estimates. The current model assumes centered, coaxial, parallel circular coils and does not include:

- lateral offset or angular misalignment
- tissue, ferrite, shielding, substrate, enclosure, or nearby-conductor loss
- capacitor ESR and matching-network loss
- source resistance, inverter loss, rectifier loss, or regulator loss
- pads, vias, feed traces, and connector parasitics
- thermal rise or manufacturing tolerance distributions

Final candidates should be checked with an electromagnetic solver such as FastHenry or Maxwell and then validated by impedance and power-transfer measurements.

## Project layout

```text
coil_optimizer/
  models.py       Configuration and sweep-grid construction
  physics.py      Circular-coil analytical models
  optimizer.py    Search, refinement, pruning, ranking, and checks
  io.py           Run orchestration and CSV/JSON output
  cli.py          Command-line interface
examples/         Small and full JSON configurations
tests/            Physics, efficiency, and end-to-end smoke tests
```

## License

This project is available under the MIT License. See `LICENSE` for details.
