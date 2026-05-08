import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import warnings


class DefectPhaseDiagram:
    """
    Constructs and plots defect phase diagrams (DPD) for a given chemical
    potential range and defect formation energies.

    The diagram identifies which defect phase is thermodynamically stable
    at each value of the solute chemical potential by finding the lowest
    formation energy envelope and the intersections between competing phases.

    All supercells are assumed to be fully periodic, containing two defects
    (e.g. grain boundaries) by construction. Formation energies are
    normalised per unit area of the defect interface.

    Uses a registry-style API: phases are added one at a time via
    :meth:`add_phase`, and formation energies are computed lazily on the
    first call to a plot method.

    Parameters
    ----------
    solute_element : str
        Symbol of the solute element (used for axis labels, e.g. ``'Al'``).
    mu_solute : array-like, shape (2,) or (N,)
        Chemical potential range of the solute element [eV]. Only the first
        and last values are used to define the range endpoints.
    mu_host : float or array-like
        Chemical potential of the host element [eV]. Supply a scalar for a
        fixed value or an array of length ``len(temperatures)`` for
        temperature-dependent values.
    mu_solute_bulk : float or array-like
        Chemical potential of the solute in its reference bulk phase [eV].
        Same scalar-or-array convention as ``mu_host``.
    temperatures : float or array-like
        Temperatures [K] corresponding to the supplied energies.
    output_energy_units : {'meV', 'eV'}, optional
        Units used on the formation-energy y-axis. Default is ``'meV'``.
    colormap : str or list, optional
        A seaborn colormap name (e.g. ``'mako_r'``) or an explicit list of
        colours, one per phase. Default is ``'mako_r'``.

    Attributes
    ----------
    phases : list of dict
        Registry of phases added via :meth:`add_phase`. Each entry contains
        keys: ``label``, ``n_solute``, ``n_total``, ``e_alloy``, ``e_pure``,
        ``area``.
    labels : list of str
        Phase labels in registration order. Populated after the first call
        to a plot method or :meth:`_compute_energies`.
    palette : list
        Colours assigned to each phase. Populated after the first call to
        a plot method or :meth:`_compute_energies`.

    Examples
    --------
    >>> dpd = DefectPhaseDiagram(
    ...     solute_element='Al',
    ...     mu_solute=[-2.0, 0.0],
    ...     mu_host=-1.5,
    ...     mu_solute_bulk=-3.2,
    ...     temperatures=300,
    ... )
    >>> dpd.add_phase('Phase A', n_solute=1, n_total=96,
    ...               e_alloy=-400.1, e_pure=-398.5, area=120.0)
    >>> dpd.add_phase('Phase B', n_solute=2, n_total=96,
    ...               e_alloy=-401.3, e_pure=-398.5, area=120.0)
    >>> fig, ax = dpd.plot_at_temperature(300)
    >>> plt.show()
    """

    def __init__(
        self,
        solute_element,
        mu_solute,
        mu_host,
        mu_solute_bulk,
        temperatures,
        output_energy_units='meV',
        colormap='mako_r',
    ):
        self.element = solute_element
        self.mu_solute = np.array([mu_solute[0], mu_solute[-1]], dtype=float)
        self.temperatures = np.atleast_1d(temperatures)
        self.mu_host = np.atleast_1d(mu_host)
        self.mu_solute_bulk = np.atleast_1d(mu_solute_bulk)
        self.output_energy_units = output_energy_units
        self.colormap = colormap

        # Internal storage
        self.phases = []
        self._energies = None  # Cached: shape (n_phases, n_temps, 2)

        # Axis labels and unit scale
        self._unit_scale = 1000 if output_energy_units == "meV" else 1
        self.xlabel = r"$\Delta\mu_{\rm " + str(self.element) + r"}$ [eV]"
        self.ylabel = (
            r"E$_{\rm{f}}$ [" + output_energy_units + r"/$\rm{\AA^2}$]"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_phase(self, label, n_solute, n_total, e_alloy, e_pure, area):
        """
        Register a defect phase with the diagram.

        Phases are stored in an internal registry and formation energies are
        recomputed lazily the next time a plot method is called. Calling this
        method invalidates any previously cached energies.

        Parameters
        ----------
        label : str
            Human-readable name for the phase (used in legends and labels).
        n_solute : int
            Number of solute atoms substituted into the supercell.
        n_total : int
            Total number of atoms in the supercell.
        e_alloy : float or array-like
            DFT total energy of the periodic supercell containing the solute
            defect [eV]. The supercell contains two defect interfaces by
            construction. Supply a scalar for a temperature-independent value
            or an array of length ``len(temperatures)`` for
            temperature-dependent energies (e.g. from free-energy
            corrections).
        e_pure : float or array-like
            DFT total energy of the pure reference supercell [eV]. Same
            scalar-or-array convention as ``e_alloy``.
        area : float
            Cross-sectional area of the defect interface [Å²] used for
            energy normalisation. Since the supercell contains two identical
            interfaces, the formation energy is computed as
            ``(2·E_alloy - E_pure - ...) / (2·area)``.

        Warns
        -----
        UserWarning
            If ``e_alloy`` or ``e_pure`` contains any NaN values the phase
            is skipped entirely and a warning is issued.

        Examples
        --------
        Add a phase with scalar (temperature-independent) energies:

        >>> dpd.add_phase('GB-I', n_solute=1, n_total=96,
        ...               e_alloy=-400.1, e_pure=-398.5, area=120.0)

        Add a phase with temperature-dependent energies:

        >>> dpd.add_phase('GB-II', n_solute=2, n_total=96,
        ...               e_alloy=[-401.1, -401.3, -401.6],
        ...               e_pure=[-398.5, -398.7, -399.0],
        ...               area=120.0)
        """
        e_alloy = np.atleast_1d(e_alloy)
        e_pure = np.atleast_1d(e_pure)

        if np.any(np.isnan(e_alloy)) or np.any(np.isnan(e_pure)):
            warnings.warn(
                f"Phase '{label}' contains NaN energies and will be excluded.",
                UserWarning,
                stacklevel=2,
            )
            return

        self.phases.append({
            'label': label,
            'n_solute': n_solute,
            'n_total': n_total,
            'e_alloy': e_alloy,
            'e_pure': e_pure,
            'area': area,
        })
        self._energies = None  # Invalidate cache

    def get_stable_phases(self, energies_at_t):
        """
        Identify the thermodynamically stable phase at each chemical
        potential by tracing the lowest formation energy envelope.

        Starting from ``mu_solute[0]``, the algorithm selects the phase with
        the lowest formation energy, then advances to the next intersection
        where a different phase becomes more stable, repeating until
        ``mu_solute[-1]``.

        Parameters
        ----------
        energies_at_t : np.ndarray, shape (n_phases, 2)
            Formation energies at ``mu_solute[0]`` and ``mu_solute[-1]`` for
            each phase at a single temperature, i.e.
            ``self._energies[:, t_idx, :]``.

        Returns
        -------
        order : list of int
            Phase indices in the order they become stable, from left
            (``mu_solute[0]``) to right (``mu_solute[-1]``).
        ordered_mu : list of float
            Boundary chemical potentials [eV] delimiting each stability
            window. Always satisfies ``len(ordered_mu) == len(order) + 1``,
            ``ordered_mu[0] == mu_solute[0]``, and
            ``ordered_mu[-1] == mu_solute[-1]``.
        rejected : list of int
            Sorted indices of phases that are never the lowest-energy phase
            anywhere in the ``mu_solute`` range.

        Notes
        -----
        Phases with identical slopes (parallel lines) are handled by
        :meth:`_calculate_intersection`, which returns ``np.inf`` for such
        pairs, effectively excluding them from the envelope tracing.
        """
        n_p = len(self.labels)
        intersections = np.array([
            [
                self._calculate_intersection(
                    energies_at_t[i], energies_at_t[j]
                )
                for j in range(n_p)
            ]
            for i in range(n_p)
        ])

        cur_mu = self.mu_solute[0]
        active = int(np.argmin(energies_at_t[:, 0]))
        order, ordered_mu = [active], [cur_mu]

        while cur_mu < self.mu_solute[1] - 1e-9:
            valid_mask = (
                (intersections[active] > cur_mu + 1e-9)
                & (intersections[active] <= self.mu_solute[1])
            )
            valid_ints = intersections[active, valid_mask]
            if not len(valid_ints):
                break

            next_mu = np.min(valid_ints)
            next_phase = int(
                np.where(np.isclose(intersections[active], next_mu))[0][0]
            )
            order.append(next_phase)
            ordered_mu.append(next_mu)
            active, cur_mu = next_phase, next_mu

        ordered_mu.append(self.mu_solute[1])
        rejected = sorted(set(range(n_p)) - set(order))
        return order, ordered_mu, rejected

    def plot_at_temperature(
        self,
        temperature,
        xlim=None,
        ylim=None,
        alpha_fill=0.7,
        legend=True,
    ):
        """
        Plot the defect phase diagram at a single temperature.

        Stable phases are rendered as filled regions bounded above by their
        formation energy line and below by the minimum formation energy.
        Unstable phases are shown as faint dashed lines for reference.
        Vertical dotted lines mark the chemical potentials at which the
        stable phase changes.

        Parameters
        ----------
        temperature : float
            Target temperature [K]. The nearest value in
            ``self.temperatures`` is selected automatically.
        xlim : tuple of float, optional
            ``(x_min, x_max)`` chemical potential axis limits [eV]. Defaults
            to the full ``mu_solute`` range.
        ylim : tuple of float, optional
            ``(y_min, y_max)`` formation energy axis limits in
            ``output_energy_units``. Computed automatically if not supplied:
            y_min is the global minimum formation energy; y_max is the
            maximum stable formation energy plus a 50-unit margin.
        alpha_fill : float, optional
            Transparency of the filled stability regions, between 0
            (transparent) and 1 (opaque). Default is ``0.7``.
        legend : bool, optional
            Whether to display the phase legend outside the right edge of
            the plot. Default is ``True``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object for further customisation.

        Raises
        ------
        ValueError
            If no phases have been added before calling this method.

        Examples
        --------
        >>> fig, ax = dpd.plot_at_temperature(600, ylim=(-500, 200))
        >>> ax.set_title('DPD at 600 K')
        >>> plt.show()
        """
        if self._energies is None:
            self._compute_energies()

        t_idx = np.argmin(np.abs(self.temperatures - temperature))
        order, ordered_mu, rejected = self.get_stable_phases(
            self._energies[:, t_idx]
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        y_floor = float(np.min(self._energies[:, t_idx]) * self._unit_scale)

        # Unstable phases — faint dashed reference lines
        for r_idx in rejected:
            e = self._energies[r_idx, t_idx] * self._unit_scale
            ax.plot(
                self.mu_solute, e,
                color=self.palette[r_idx],
                ls='--', alpha=0.3, lw=1,
            )

        # Stable phases — filled stability windows
        for i, p_idx in enumerate(order):
            e = self._energies[p_idx, t_idx]
            mu_s, mu_e = ordered_mu[i], ordered_mu[i + 1]
            slope = (e[1] - e[0]) / (self.mu_solute[1] - self.mu_solute[0])
            y_s = self._unit_scale * (e[0] + slope * (mu_s - self.mu_solute[0]))
            y_e = self._unit_scale * (e[0] + slope * (mu_e - self.mu_solute[0]))

            ax.fill_between(
                [mu_s, mu_e], [y_s, y_e], y_floor,
                color=self.palette[p_idx],
                alpha=alpha_fill,
                label=self.labels[p_idx],
            )
            if i > 0:
                ax.axvline(mu_s, color='k', ls=':', lw=1, alpha=0.4)

        ax.set_xlabel(self.xlabel, fontsize=13)
        ax.set_ylabel(self.ylabel, fontsize=13)
        ax.set_xlim(xlim if xlim is not None else self.mu_solute)

        if ylim:
            ax.set_ylim(ylim)
        else:
            e_stable_max = float(
                np.max(self._energies[order, t_idx]) * self._unit_scale
            )
            ax.set_ylim(y_floor, e_stable_max + 50)

        if legend:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_energies(self):
        """
        Vectorized computation of formation energies for all phases and
        temperatures, caching the result in ``self._energies``.

        All supercells are fully periodic and contain two identical defect
        interfaces by construction. The formation energy per interface area
        is evaluated as:

        .. math::

            E_f = \\frac{2 E_{\\rm alloy} - E_{\\rm pure}
                         - (N - 2n)\\mu_{\\rm host}
                         - 2n(\\mu_{\\rm bulk} + \\Delta\\mu_{\\rm solute})}
                        {2A}

        where :math:`N` is the total atom count, :math:`n` the number of
        solute atoms, and :math:`A` the cross-sectional area of one
        interface. The factor of 2 in the denominator normalises per
        interface rather than per cell.

        Raises
        ------
        ValueError
            If no phases have been registered via :meth:`add_phase`.

        Notes
        -----
        After this method runs the following attributes are populated:

        - ``self._energies`` : np.ndarray, shape ``(n_phases, n_temps, 2)``
          Formation energies [eV/Å²]. Axis 2 holds
          ``[E_f(mu_solute[0]), E_f(mu_solute[-1])]``.
        - ``self.labels`` : list of str
        - ``self.palette`` : list of colours
        """
        if not self.phases:
            raise ValueError("No phases added. Call add_phase() first.")

        df = pd.DataFrame(self.phases)

        # Shape: (n_phases, n_temps)
        e_alloy = np.array(df['e_alloy'].tolist())
        e_pure  = np.array(df['e_pure'].tolist())

        # Shape: (n_phases, 1) for broadcasting over temperatures
        area     = df['area'].values[:, np.newaxis]
        n_solute = df['n_solute'].values[:, np.newaxis]
        n_total  = df['n_total'].values[:, np.newaxis]

        # Solute chemical potential endpoints: shape (1, 1, 2) for broadcasting
        d_mu = self.mu_solute[np.newaxis, np.newaxis, :]

        # Host and solute chemical potential contributions
        # mu_host and mu_solute_bulk have shape (n_temps,); broadcast to (n_phases, n_temps)
        host_term   = (n_total - 2 * n_solute) * self.mu_host[np.newaxis, :]    # (n_phases, n_temps)
        solute_term = (
            2 * n_solute[:, :, np.newaxis] * (self.mu_solute_bulk[np.newaxis, :, np.newaxis] + d_mu)
        )                                                                         # (n_phases, n_temps, 2)

        raw = (
            2 * e_alloy[:, :, np.newaxis]
            - e_pure[:, :, np.newaxis]
            - host_term[:, :, np.newaxis]
            - solute_term
        )                                                          # (n_phases, n_temps, 2)

        # Normalise per interface: supercell contains 2 identical interfaces
        self._energies = raw / (2 * area[:, :, np.newaxis])

        self.labels = df['label'].tolist()
        self.palette = (
            sns.color_palette(self.colormap, n_colors=len(self.labels))
            if isinstance(self.colormap, str)
            else self.colormap
        )

    def _calculate_intersection(self, e1, e2):
        """
        Find the chemical potential at which two linear E_f(Δμ_solute)
        lines cross.

        Both formation energy curves are assumed to be linear between
        ``mu_solute[0]`` and ``mu_solute[-1]``, defined by their values at
        those two endpoints.

        Parameters
        ----------
        e1 : array-like, shape (2,)
            ``[E_f(mu_solute[0]), E_f(mu_solute[-1])]`` for the first phase
            [eV/Å²].
        e2 : array-like, shape (2,)
            ``[E_f(mu_solute[0]), E_f(mu_solute[-1])]`` for the second phase
            [eV/Å²].

        Returns
        -------
        float
            Chemical potential [eV] at the intersection point, or ``np.inf``
            if the two lines are parallel (slopes differ by less than the
            ``np.isclose`` tolerance).
        """
        mu_range = self.mu_solute[1] - self.mu_solute[0]
        slope1 = (e1[1] - e1[0]) / mu_range
        slope2 = (e2[1] - e2[0]) / mu_range
        if np.isclose(slope1, slope2):
            return np.inf
        return self.mu_solute[0] + (e2[0] - e1[0]) / (slope1 - slope2)