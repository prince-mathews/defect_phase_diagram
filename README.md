# Defect Phase Diagram

A Python tool for constructing and visualising **defect phase diagrams (DPDs)** for binary alloy systems, for analysing the thermodynamic stability of defect phases across a range of solute chemical potentials.

---

## What is a Defect Phase Diagram?

In computational materials science, a defect phase diagram maps the thermodynamic stability of stable and metastable phases as a function of the solute chemical potential Δμ. At each value of Δμ, the stable phase is the one with the lowest formation energy per unit area.

---

## Features

- Registry-style API — add phases one at a time with `add_phase()`
- Temperature-dependent energies (e.g. from free-energy corrections)
- Vectorized NumPy formation energy calculation
- Automatic stable-phase envelope tracing via pairwise line intersections
- Clean `matplotlib` output with filled stability regions

---

## Installation

```bash
git clone https://github.com/prince-mathews/defect_phase_diagram.git
cd defect_phase_diagram
pip install -r requirements.txt
```

---

## Quick Start

```python
import numpy as np
from dpd import DefectPhaseDiagram

dpd = DefectPhaseDiagram(
    solute_element='Al',
    mu_solute=[-2.0, 0.0],
    mu_host=-1.52,
    mu_solute_bulk=-3.21,
    temperatures=300,
    output_energy_units='meV',
)

dpd.add_phase('GB Phase I',  n_solute=1, n_total=96,
              e_alloy=-401.12, e_pure=-398.50, area=120.0)
dpd.add_phase('GB Phase II', n_solute=2, n_total=96,
              e_alloy=-403.85, e_pure=-398.50, area=120.0)

fig, ax = dpd.plot_at_temperature(300)
```

See [`examples/example_usage.py`](examples/example_usage.py) for a full walkthrough.

---

## Formation Energy Formula

For each periodic supercell containing one defect configuration and one pure defect, the formation energy per interface area at chemical potential Δμ_solute is:

```
E_f = [2·E_alloy − E_pure − (N − 2n)·μ_host − 2n·(μ_bulk + Δμ_solute)] / (2·A)
```

where `N` = total atoms, `n` = solute atoms, `A` = cross-sectional area of one interface. The factor of 2 in the denominator normalises per interface rather than per cell. All supercells are assumed to be fully periodic, containing two identical defect interfaces (e.g. grain boundaries) by construction. Formation energies are normalised per interface area.

---

## API Reference

### `DefectPhaseDiagram(solute_element, mu_solute, mu_host, mu_solute_bulk, temperatures, ...)`

| Parameter | Type | Description |
|---|---|---|
| `solute_element` | `str` | Solute element symbol (e.g. `'Al'`) |
| `mu_solute` | array-like | Chemical potential range [eV]; only endpoints are used |
| `mu_host` | float or array | Host element chemical potential [eV]; array for temperature-dependent values |
| `mu_solute_bulk` | float or array | Solute bulk reference chemical potential [eV] |
| `temperatures` | float or array | Temperatures [K] matching the energy arrays |
| `output_energy_units` | `'meV'` or `'eV'` | Y-axis units (default `'meV'`) |
| `colormap` | str or list | Seaborn colormap name or list of colours (default `'mako_r'`) |

### `add_phase(label, n_solute, n_total, e_alloy, e_pure, area)`

Registers a defect phase. `e_alloy` and `e_pure` accept scalars or arrays (one value per temperature).

| Parameter | Description |
|---|---|
| `label` | Phase name shown in the legend |
| `n_solute` | Number of solute atoms in the supercell |
| `n_total` | Total atoms in the supercell |
| `e_alloy` | DFT energy of the defect supercell [eV] |
| `e_pure` | DFT energy of the pure reference supercell [eV] |
| `area` | Cross-sectional area of one defect interface [Å²] |

### `plot_at_temperature(temperature, xlim=None, ylim=None, alpha_fill=0.7, legend=True)`

Returns `(fig, ax)`.

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
