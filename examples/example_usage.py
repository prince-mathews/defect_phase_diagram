"""
Example usage of DefectPhaseDiagram
=====================================
This script demonstrates how to set up a defect phase diagram
for a binary Mg–Al alloy system with grain boundary defect phases.
All supercells are fully periodic, containing two defect interfaces
by construction.
"""

import numpy as np
import matplotlib.pyplot as plt
from dpd import DefectPhaseDiagram

# ── 1. Define the chemical potential range and thermodynamic parameters ──────

mu_solute    = np.linspace(-2.0, 0.0, 100)  # Solute chemical potential range [eV]
mu_host      = -1.52                         # Host element chemical potential [eV]
mu_bulk      = -3.21                         # Solute bulk reference [eV]
temps        = 300                           # Single temperature [K]

# ── 2. Initialise the diagram ────────────────────────────────────────────────

dpd = DefectPhaseDiagram(
    solute_element='Al',
    mu_solute=mu_solute,
    mu_host=mu_host,
    mu_solute_bulk=mu_bulk,
    temperatures=temps,
    output_energy_units='meV',
    colormap='mako_r',
)

# ── 3. Register defect phases ────────────────────────────────────────────────
# add_phase(label, n_solute, n_total, e_alloy, e_pure, area)
# e_alloy: DFT energy of periodic supercell with solute defect [eV]
# e_pure:  DFT energy of pure reference supercell [eV]
# area:    cross-sectional area of one defect interface [Å²]

dpd.add_phase(
    label='GB Phase I',
    n_solute=1,
    n_total=96,
    e_alloy=-401.12,
    e_pure=-398.50,
    area=120.0,
)

dpd.add_phase(
    label='GB Phase II',
    n_solute=2,
    n_total=96,
    e_alloy=-403.85,
    e_pure=-398.50,
    area=120.0,
)

dpd.add_phase(
    label='GB Phase III',
    n_solute=3,
    n_total=96,
    e_alloy=-406.10,
    e_pure=-398.50,
    area=120.0,
)

# ── 4. Plot ──────────────────────────────────────────────────────────────────

fig, ax = dpd.plot_at_temperature(
    temperature=300,
    alpha_fill=0.7,
    legend=True,
)

ax.set_title('Defect Phase Diagram — Mg–Al at 300 K', fontsize=13)
plt.savefig('dpd_example.png', dpi=150, bbox_inches='tight')
plt.show()
