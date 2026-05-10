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

    def add_phase(self, label, n_solute, n_total, e_alloy, area, e_pure=None):
        """
        Register a defect phase with the diagram.

        Phases are stored in an internal registry and formation energies are
        recomputed lazily the next time a plot method is called. Calling this
        method invalidates any previously cached energies.

        The supercell always contains two defect interfaces by construction.
        The ``e_pure`` parameter controls which decoration case applies:

        **One decorated interface** (``e_pure`` provided):
            One interface carries the solute defect; the other is pure.
            Formation energy per interface:

            .. code-block:: none

                E_f = [2·E_alloy - E_pure - (N - 2n)·μ_host
                       - 2n·(μ_bulk + Δμ)] / (2·A)

        **Both interfaces decorated** (``e_pure=None``):
            Both interfaces are identically decorated. ``n_solute`` and
            ``n_total`` should be the per-interface counts (i.e. half the
            supercell values). Formation energy per interface:

            .. code-block:: none

                E_f = [E_alloy - (N - n)·μ_host - n·(μ_bulk + Δμ)] / A

        Parameters
        ----------
        label : str
            Human-readable name for the phase (used in legends and labels).
        n_solute : int
            Number of solute atoms per defect interface. For the one-decorated
            case this equals the total solute atoms in the supercell; for the
            both-decorated case this is half the supercell total.
        n_total : int
            Total number of atoms per defect interface. Same per-interface
            convention as ``n_solute``.
        e_alloy : float or array-like
            DFT total energy of the decorated supercell [eV]. Supply a scalar
            for a temperature-independent value or an array of length
            ``len(temperatures)`` for temperature-dependent energies (e.g.
            from free-energy corrections).
        area : float
            Cross-sectional area of one defect interface [Å²] used for energy
            normalisation.
        e_pure : float, array-like, or None, optional
            DFT total energy of the pure reference supercell [eV]. Same
            scalar-or-array convention as ``e_alloy``. Set to ``None``
            (default) for the both-decorated case, where no pure reference
            is needed.

        Warns
        -----
        UserWarning
            If ``e_alloy`` or ``e_pure`` (when provided) contains any NaN
            values the phase is skipped entirely and a warning is issued.

        Examples
        --------
        One decorated interface, scalar energies:

        >>> dpd.add_phase('GB-I', n_solute=1, n_total=96,
        ...               e_alloy=-400.1, e_pure=-398.5, area=120.0)

        One decorated interface, temperature-dependent energies:

        >>> dpd.add_phase('GB-II', n_solute=2, n_total=96,
        ...               e_alloy=[-401.1, -401.3, -401.6],
        ...               e_pure=[-398.5, -398.7, -399.0],
        ...               area=120.0)

        Both interfaces decorated (no pure reference needed):

        >>> dpd.add_phase('GB-III', n_solute=1, n_total=48,
        ...               e_alloy=-400.1, area=120.0)
        """
        e_alloy = np.atleast_1d(e_alloy)

        if np.any(np.isnan(e_alloy)):
            warnings.warn(
                f"Phase '{label}' contains NaN energies and will be excluded.",
                UserWarning,
                stacklevel=2,
            )
            return

        if e_pure is not None:
            e_pure = np.atleast_1d(e_pure)
            if np.any(np.isnan(e_pure)):
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
            'e_pure': e_pure,  # None signals both-decorated case
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

    def get_hull(self, temperature):
        """
        Return the convex hull vertices of the defect phase diagram at a
        given temperature.

        The hull is defined by the boundary points of each stability window —
        one energy value per entry in ``ordered_mu`` — connecting the lowest
        formation energy envelope across the full ``mu_solute`` range.

        Parameters
        ----------
        temperature : float
            Target temperature [K]. The nearest value in
            ``self.temperatures`` is selected automatically.

        Returns
        -------
        mu_values : np.ndarray, shape (n_boundaries,)
            Chemical potential [eV] at each hull vertex. Includes
            ``mu_solute[0]``, all phase-boundary intersections, and
            ``mu_solute[-1]``.
        energies : np.ndarray, shape (n_boundaries,)
            Formation energy [eV/Å²] of the stable phase at each
            corresponding ``mu_values`` entry.

        Notes
        -----
        Energies are returned in raw [eV/Å²] — multiply by ``1000`` for
        [meV/Å²] if needed, consistent with ``output_energy_units``.

        Examples
        --------
        >>> mu_hull, e_hull = dpd.get_hull(300)
        >>> plt.plot(mu_hull, 1000 * e_hull, marker='o', ms=7, lw=2)
        """
        if self._energies is None:
            self._compute_energies()

        t_idx = np.argmin(np.abs(self.temperatures - temperature))
        energies_at_t = self._energies[:, t_idx]
        order, ordered_mu, _ = self.get_stable_phases(energies_at_t)

        mu_values, energies = [], []
        for i, p_idx in enumerate(order):
            e = energies_at_t[p_idx]
            slope = (e[1] - e[0]) / (self.mu_solute[1] - self.mu_solute[0])
            for mu in [ordered_mu[i], ordered_mu[i + 1]]:
                mu_values.append(mu)
                energies.append(e[0] + slope * (mu - self.mu_solute[0]))

        # Deduplicate consecutive identical mu values at phase boundaries
        mu_arr = np.array(mu_values)
        e_arr  = np.array(energies)
        mask   = np.concatenate(([True], ~np.isclose(np.diff(mu_arr), 0)))
        return mu_arr[mask], e_arr[mask]

    def plot_at_temperature(
        self,
        temperature,
        xlim=None,
        ylim=None,
        alpha_fill=1.0,
        linewidth=2,
        figsize=(8, 5),
        fontsize=16,
        plot_intersections=True,
        legend=True,
    ):
        """
        Plot the defect phase diagram at a single temperature.

        Stable phases are rendered as filled regions bounded above by their
        formation energy line and below by the minimum formation energy.
        Unstable phases are shown as faint dashed lines for reference.
        Vertical dotted lines mark the chemical potentials at which the
        stable phase changes, with the μ value annotated.

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
            (transparent) and 1 (opaque). Default is ``1.0``.
        linewidth : float, optional
            Line width for the formation energy lines of both stable and
            unstable phases, and for the intersection marker lines.
            Default is ``2``.
        figsize : tuple of float, optional
            Figure dimensions as ``(width, height)`` in inches.
            Default is ``(8, 5)``.
        fontsize : int or float, optional
            Font size for axis labels and tick labels. Tick labels are
            rendered at the same size for consistency. Default is ``16``.
        plot_intersections : bool, optional
            Whether to draw vertical lines and annotate the μ value at each
            phase boundary. Default is ``True``.
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
        >>> fig, ax = dpd.plot_at_temperature(
        ...     600, ylim=(-500, 200), figsize=(10, 6), fontsize=14,
        ...     plot_intersections=False,
        ... )
        >>> ax.set_title('DPD at 600 K')
        >>> plt.show()
        """
        if self._energies is None:
            self._compute_energies()

        t_idx = np.argmin(np.abs(self.temperatures - temperature))
        order, ordered_mu, rejected = self.get_stable_phases(
            self._energies[:, t_idx]
        )

        fig, ax = plt.subplots(figsize=figsize)
        y_floor = float(np.min(self._energies[:, t_idx]) * self._unit_scale)

        # Resolve axis limits early so intersection labels can be positioned
        x_min = xlim[0] if xlim is not None else float(self.mu_solute[0])
        x_max = xlim[1] if xlim is not None else float(self.mu_solute[1])
        x_range = x_max - x_min

        if ylim:
            y_min_lim, y_max_lim = ylim
        else:
            e_stable_max = float(
                np.max(self._energies[order, t_idx]) * self._unit_scale
            )
            y_min_lim = y_floor
            y_max_lim = e_stable_max + 50
        y_range = y_max_lim - y_min_lim

        # All phases — dashed lines across full mu_solute range drawn first
        # so they are visible in regions where no phase is stable
        for p_idx in range(len(self.labels)):
            e = self._energies[p_idx, t_idx] * self._unit_scale
            if p_idx in rejected:
                ax.plot(
                    self.mu_solute, e,
                    color=self.palette[p_idx],
                    ls='--', alpha=0.3, lw=linewidth,
                    label=self.labels[p_idx],
                )
            else:
                ax.plot(
                    self.mu_solute, e,
                    color=self.palette[p_idx],
                    ls='--', alpha=0.7, lw=linewidth,
                )

        # Stable phases — filled stability windows drawn on top
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

            if i > 0 and plot_intersections:
                ax.axvline(mu_s, color='k', ls='--', lw=1.25, alpha=0.7)
                ax.text(
                    mu_s - 0.01 * x_range,
                    y_min_lim - 0.28 * y_range,
                    r'$\mu=${:.3f}'.format(mu_s),
                    fontsize=fontsize - 7,
                    rotation=90,
                )

        ax.set_xlabel(self.xlabel, fontsize=fontsize)
        ax.set_ylabel(self.ylabel, fontsize=fontsize, labelpad=-5)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min_lim, y_max_lim)

        if legend:
            ax.legend(fontsize=fontsize - 2, loc='upper left',
                      bbox_to_anchor=(1, 1))

        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_energies(self):
        """
        Vectorized computation of formation energies for all phases and
        temperatures, caching the result in ``self._energies``.

        Two decoration cases are supported, determined by whether ``e_pure``
        was supplied to :meth:`add_phase`:

        **One decorated interface** (``e_pure`` provided):

        .. math::

            E_f = \\frac{2 E_{\\rm alloy} - E_{\\rm pure}
                         - (N - 2n)\\mu_{\\rm host}
                         - 2n(\\mu_{\\rm bulk} + \\Delta\\mu_{\\rm solute})}
                        {2A}

        **Both interfaces decorated** (``e_pure=None``):

        .. math::

            E_f = \\frac{E_{\\rm alloy}
                         - (N - n)\\mu_{\\rm host}
                         - n(\\mu_{\\rm bulk} + \\Delta\\mu_{\\rm solute})}
                        {A}

        In both cases :math:`N`, :math:`n`, and :math:`A` are per-interface
        quantities.

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

        # Shape: (n_phases, 1) for broadcasting over temperatures
        area     = df['area'].values[:, np.newaxis]
        n_solute = df['n_solute'].values[:, np.newaxis]
        n_total  = df['n_total'].values[:, np.newaxis]

        # Boolean mask: True where e_pure was provided (one-decorated case)
        one_decorated = np.array(
            [p['e_pure'] is not None for p in self.phases], dtype=bool
        )

        # Solute chemical potential endpoints: shape (1, 1, 2) for broadcasting
        d_mu = self.mu_solute[np.newaxis, np.newaxis, :]

        # Initialise output array: (n_phases, n_temps, 2)
        n_temps = len(self.temperatures)
        raw = np.zeros((len(self.phases), n_temps, 2))

        # --- One-decorated case: E_f = [2·E_alloy - E_pure - ...] / (2·A) ---
        if one_decorated.any():
            e_pure_arr = np.array(
                [p['e_pure'] for p in self.phases if p['e_pure'] is not None]
            )                                                            # (n_one, n_temps)

            host_term_one = (
                (n_total - 2 * n_solute) * self.mu_host[np.newaxis, :]
            )[one_decorated]                                             # (n_one, n_temps)

            solute_term_one = (
                2 * n_solute[:, :, np.newaxis]
                * (self.mu_solute_bulk[np.newaxis, :, np.newaxis] + d_mu)
            )[one_decorated]                                             # (n_one, n_temps, 2)

            raw[one_decorated] = (
                2 * e_alloy[one_decorated, :, np.newaxis]
                - e_pure_arr[:, :, np.newaxis]
                - host_term_one[:, :, np.newaxis]
                - solute_term_one
            )

        # --- Both-decorated case: E_f = [E_alloy - (N-n)·μ_host - n·(...)] / A ---
        both_decorated = ~one_decorated
        if both_decorated.any():
            host_term_both = (
                (n_total - n_solute) * self.mu_host[np.newaxis, :]
            )[both_decorated]                                            # (n_both, n_temps)

            solute_term_both = (
                n_solute[:, :, np.newaxis]
                * (self.mu_solute_bulk[np.newaxis, :, np.newaxis] + d_mu)
            )[both_decorated]                                            # (n_both, n_temps, 2)

            raw[both_decorated] = (
                e_alloy[both_decorated, :, np.newaxis]
                - host_term_both[:, :, np.newaxis]
                - solute_term_both
            )

        # Normalise: one-decorated → 2·A; both-decorated → A
        divisor = np.where(
            one_decorated[:, np.newaxis, np.newaxis],
            2.0 * area[:, :, np.newaxis],
                  area[:, :, np.newaxis],
        )
        self._energies = raw / divisor

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