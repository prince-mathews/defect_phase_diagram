import pytest
import numpy as np
import warnings
from dpd import DefectPhaseDiagram


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_dpd():
    """Minimal DefectPhaseDiagram with two phases for reuse across tests."""
    dpd = DefectPhaseDiagram(
        x_element='Al',
        mu_x=[-2.0, 0.0],
        mu_Mg=-1.5,
        mu_x_bulk=-3.2,
        temperatures=300,
        output_energy_units='meV',
    )
    dpd.add_phase('Phase A', n_solute=1, n_total=96,
                  e_alloy=-400.1, e_pure=-398.5, area=120.0)
    dpd.add_phase('Phase B', n_solute=2, n_total=96,
                  e_alloy=-403.5, e_pure=-398.5, area=120.0)
    return dpd


# ── Initialisation ────────────────────────────────────────────────────────────

class TestInit:
    def test_mu_x_endpoints_stored(self):
        dpd = DefectPhaseDiagram('Al', mu_x=[-2, -1, 0],
                                 mu_Mg=-1.5, mu_x_bulk=-3.2, temperatures=300)
        assert dpd.mu_x[0] == -2.0
        assert dpd.mu_x[1] == 0.0

    def test_unit_scale_meV(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300,
                                 output_energy_units='meV')
        assert dpd._unit_scale == 1000

    def test_unit_scale_eV(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300,
                                 output_energy_units='eV')
        assert dpd._unit_scale == 1

    def test_phases_initially_empty(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300)
        assert dpd.phases == []

    def test_energies_cache_initially_none(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300)
        assert dpd._energies is None


# ── add_phase ─────────────────────────────────────────────────────────────────

class TestAddPhase:
    def test_phase_is_registered(self, base_dpd):
        assert len(base_dpd.phases) == 2

    def test_phase_label_stored(self, base_dpd):
        assert base_dpd.phases[0]['label'] == 'Phase A'

    def test_nan_e_alloy_skipped(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dpd.add_phase('Bad', 1, 96, np.nan, -398.5, 120.0)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
        assert len(dpd.phases) == 0

    def test_nan_e_pure_skipped(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dpd.add_phase('Bad', 1, 96, -400.0, np.nan, 120.0)
            assert len(w) == 1
        assert len(dpd.phases) == 0

    def test_add_phase_invalidates_cache(self, base_dpd):
        base_dpd._compute_energies()
        assert base_dpd._energies is not None
        base_dpd.add_phase('Phase C', 1, 96, -399.0, -398.5, 120.0)
        assert base_dpd._energies is None


# ── _compute_energies ─────────────────────────────────────────────────────────

class TestComputeEnergies:
    def test_output_shape(self, base_dpd):
        base_dpd._compute_energies()
        # (n_phases=2, n_temps=1, 2 endpoints)
        assert base_dpd._energies.shape == (2, 1, 2)

    def test_labels_populated(self, base_dpd):
        base_dpd._compute_energies()
        assert base_dpd.labels == ['Phase A', 'Phase B']

    def test_palette_populated(self, base_dpd):
        base_dpd._compute_energies()
        assert len(base_dpd.palette) == 2

    def test_no_phases_raises(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300)
        with pytest.raises(ValueError, match="No phases added"):
            dpd._compute_energies()

    def test_periodic_uses_first_mu_Mg(self):
        """Periodic phases should always use mu_Mg[0], not per-temperature."""
        dpd = DefectPhaseDiagram(
            'Al', [-2, 0],
            mu_Mg=[-1.5, -1.6, -1.7],
            mu_x_bulk=[-3.2, -3.3, -3.4],
            temperatures=[300, 400, 500],
        )
        dpd.add_phase('Periodic', 1, 96,
                      e_alloy=[-400.1, -400.2, -400.3],
                      e_pure=[-398.5, -398.6, -398.7],
                      area=120.0, is_periodic=True)
        dpd._compute_energies()
        assert dpd._energies.shape == (1, 3, 2)


# ── _calculate_intersection ───────────────────────────────────────────────────

class TestCalculateIntersection:
    def test_known_intersection(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300)
        # e1: constant at 1.0; e2: rises from 0 to 2 → crosses at mu=-1
        e1 = np.array([1.0, 1.0])
        e2 = np.array([0.0, 2.0])
        result = dpd._calculate_intersection(e1, e2)
        assert np.isclose(result, -1.0)

    def test_parallel_lines_return_inf(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300)
        e1 = np.array([1.0, 2.0])
        e2 = np.array([0.5, 1.5])
        assert dpd._calculate_intersection(e1, e2) == np.inf


# ── get_stable_phases ─────────────────────────────────────────────────────────

class TestGetStablePhases:
    def test_returns_three_lists(self, base_dpd):
        base_dpd._compute_energies()
        result = base_dpd.get_stable_phases(base_dpd._energies[:, 0])
        assert len(result) == 3

    def test_ordered_mu_bounds(self, base_dpd):
        base_dpd._compute_energies()
        _, ordered_mu, _ = base_dpd.get_stable_phases(base_dpd._energies[:, 0])
        assert ordered_mu[0] == base_dpd.mu_x[0]
        assert ordered_mu[-1] == base_dpd.mu_x[1]

    def test_rejected_plus_order_covers_all(self, base_dpd):
        base_dpd._compute_energies()
        order, _, rejected = base_dpd.get_stable_phases(base_dpd._energies[:, 0])
        all_inds = sorted(order + rejected)
        assert all_inds == list(range(len(base_dpd.labels)))


# ── plot_at_temperature ───────────────────────────────────────────────────────

class TestPlotAtTemperature:
    def test_returns_fig_and_ax(self, base_dpd):
        import matplotlib.pyplot as plt
        fig, ax = base_dpd.plot_at_temperature(300)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_triggers_energy_computation(self):
        dpd = DefectPhaseDiagram('Al', [-2, 0], -1.5, -3.2, 300)
        dpd.add_phase('Phase A', 1, 96, -400.1, -398.5, 120.0)
        assert dpd._energies is None
        import matplotlib.pyplot as plt
        fig, _ = dpd.plot_at_temperature(300)
        assert dpd._energies is not None
        plt.close(fig)
