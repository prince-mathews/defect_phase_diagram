# Defect Phase Diagram

A Python tool for constructing and visualising **defect phase diagrams (DPDs)** for binary alloy systems, identifying and predicting thermodynamically of stable and mestastable defct phases across a range of solute chemical potentials.

---

## What is a Defect Phase Diagram?

In computational materials science, a defect phase diagram maps the thermodynamic stability of competing defect configurations (e.g. different substitutional arrangements of a solute element) as a function of the solute chemical potential Δμ. At each value of Δμ, the stable phase is the one with the lowest formation energy per unit area.

---

## Features

- Registry-style API — add phases one at a time with `add_phase()`
- Temperature-dependent energies (e.g. from free-energy considerations)
- Vectorized NumPy formation energy calculation
- Automatic stable-phase envelope tracing via pairwise line intersections
- Clean `matplotlib` output with filled stability regions

---

## Installation

```bash
git clone https://github.com/your-username/defect-phase-diagram.git
cd defect-phase-diagram
pip install -r requirements.txt
```

---

## Quick Start

```python
import numpy as np
from dpd import DefectPhaseDiagram

dpd = DefectPhaseDiagram(
    x_element='Al',
    mu_x=[-2.0, 0.0],
    mu_Mg=-1.52,
    mu_x_bulk=-3.21,
    temperatures=300,
    output_energy_units='meV',
)

dpd.add_phase('GB - 1 Al sub.', n_solute=1, n_total=96,
              e_alloy=-401.12, e_pure=-398.50, area=120.0)
dpd.add_phase('GB - 2 Al sub.', n_solute=2, n_total=96,
              e_alloy=-403.85, e_pure=-398.50, area=120.0)

fig, ax = dpd.plot_at_temperature(300)
```

See [`examples/example_usage.py`](examples/example_usage.py) for a full walkthrough.

---

## API Reference

### `DefectPhaseDiagram(x_element, mu_x, mu_Mg, mu_x_bulk, temperatures, ...)`

| Parameter | Type | Description |
|---|---|---|
| `x_element` | `str` | Solute element symbol (e.g. `'Al'`) |
| `mu_x` | array-like | Chemical potential range [eV]; only endpoints are used |
| `mu_Mg` | float or array | Mg chemical potential [eV]; array for temperature-dependent values |
| `mu_x_bulk` | float or array | Solute bulk reference chemical potential [eV] |
| `temperatures` | float or array | Temperatures [K] matching the energy arrays |
| `output_energy_units` | `'meV'` or `'eV'` | Y-axis units (default `'meV'`) |
| `colormap` | str or list | Seaborn colormap name or list of colours (default `'mako_r'`) |

### `add_phase(label, n_solute, n_total, e_alloy, e_pure, area, is_periodic=False)`

Registers a defect phase. `e_alloy` and `e_pure` accept scalars or arrays (one value per temperature).

| Parameter | Description |
|---|---|
| `label` | Phase name shown in the legend |
| `n_solute` | Number of solute atoms in the supercell |
| `n_total` | Total atoms in the supercell |
| `e_alloy` | DFT energy of the defect supercell [eV] |
| `e_pure` | DFT energy of the pure reference supercell [eV] |
| `area` | Cross-sectional area [Å²] (doubled internally for surface slabs) |
| `is_periodic` | `True` for periodic supercells; `False` (default) for surface slabs |

### `plot_at_temperature(temperature, xlim=None, ylim=None, alpha_fill=0.7, legend=True)`

Returns `(fig, ax)`.

---

## Formation Energy Formula

For each phase, the formation energy per unit area at chemical potential Δμ_x is:

```
E_f = [2·E_alloy − E_pure − (N − 2n)·μ_Mg − 2n·(μ_bulk + Δμ_x)] / (2·A)
```

where `N` = total atoms, `n` = solute atoms, `A` = effective area.

---

## Running Tests

```bash
pip install pytest
pytest tests/
```

---

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.23
- matplotlib ≥ 3.5
- seaborn ≥ 0.12
- pandas ≥ 1.5

---

## License

MIT
