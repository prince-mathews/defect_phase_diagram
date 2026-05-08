"""
Example usage of DefectPhaseDiagram
=====================================
This script demonstrates how to set up a defect phase diagram
for a binary Mg–X alloy surface system.
"""

import numpy as np
import matplotlib.pyplot as plt
from dpd import DefectPhaseDiagram

# ── 1. Define the chemical potential range and thermodynamic parameters ──────

mu_x    = np.linspace(-2.0, 0.0, 100)   # Solute chemical potential range [eV]
mu_Mg   = -1.52                          # Mg chemical potential [eV]
mu_bulk = -3.21                          # Solute bulk reference [eV]
temps   = 300                            # Single temperature [K]

# ── 2. Initialise the diagram ────────────────────────────────────────────────

dpd = DefectPhaseDiagram(
    x_element='Al',
    mu_x=mu_x,
    mu_Mg=mu_Mg,
    mu_x_bulk=mu_bulk,
    temperatures=temps,
    output_energy_units='meV',
    colormap='mako_r',
)

# ── 3. Register defect phases ────────────────────────────────────────────────
# add_phase(label, n_solute, n_total, e_alloy, e_pure, area, is_periodic)

dpd.add_phase(
    label='Substitutional (1Al)',
    n_solute=1,
    n_total=96,
    e_alloy=-401.12,
    e_pure=-398.50,
    area=120.0,
    is_periodic=False,
)

dpd.add_phase(
    label='Substitutional (2Al)',
    n_solute=2,
    n_total=96,
    e_alloy=-403.85,
    e_pure=-398.50,
    area=120.0,
    is_periodic=False,
)

dpd.add_phase(
    label='Periodic reference',
    n_solute=1,
    n_total=96,
    e_alloy=-400.70,
    e_pure=-398.50,
    area=120.0,
    is_periodic=True,
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
