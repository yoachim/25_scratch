from functools import cache
import numpy as np
import warnings
import copy
import healpy as hp

import rubin_scheduler.utils as utils
import matplotlib.pylab as plt
from matplotlib import ticker


UNIT_LOOKUP_DICT = {"night": "Days", "fiveSigmaDepth": "mag", "airmass": "airmass"}


class BaseMetric(object):
    """Example of a simple metric."""

    def __init__(self, col="night", unit=None, name="name"):
        self.shape = None
        self.dtype = float
        self.col = col
        self.name = name
        if unit is None:
            self.unit = UNIT_LOOKUP_DICT[self.col]
        else:
            self.unit = unit

    def add_info(self, info):
        info["metric: name"] = self.name
        info["metric: col"] = self.col
        info["metric: unit"] = self.unit
        return info

    def __call__(self, visits, slice_point=None):
        pass

    def call_cached(self, hashable, visits=None, slice_point=None):
        """hashable should be something like a frozenset of
        visitIDs. If you use call_cached when slicepoint has
        important data (extinction, stellar density, etc),
        then this can give a different (wrong) result
        """
        self.visits = visits
        self.slice_point = slice_point
        return self.call_cached_post(hashable)

    # XXX--danger, this simple caching can cause a memory leak.
    # Probably need to do the caching in the slicer as before.
    @cache
    def call_cached_post(self, hashable):
        return self.__call__(self.visits, slice_point=self.slice_point)


class MeanMetric(BaseMetric):
    def __init__(self, col="night", unit=None, name="name"):
        super().__init__(col=col, unit=unit, name="mean")

    def __call__(self, visits, slice_point=None):
        return np.mean(visits[self.col])


class CountMetric(MeanMetric):
    def __init__(self, col="night", unit="#", name="Count"):
        super().__init__(col=col, unit=unit, name=name)

    def __call__(self, visits, slice_point=None):
        return np.size(visits[self.col])


class CoaddM5Metric(MeanMetric):
    def __init__(self, filtername, col="fiveSigmaDepth", unit=None):
        self.shape = None
        self.dtype = float
        self.col = col
        self.filtername = filtername
        if unit is None:
            self.unit = "Coadded Depth, %s (mags)" % self.filtername
        else:
            self.unit = unit

    def add_info(self, info):
        info["metric: name"] = "CoaddM5Metric" # self.__class__.name ?
        info["metric: col"] = self.col
        info["metric: unit"] = self.unit
        return info

    @staticmethod
    def coadd(single_visit_m5s):
        return 1.25 * np.log10(np.sum(10.0 ** (0.8 * single_visit_m5s)))

    def __call__(self, visits, slice_point=None):
        if np.size(np.unique(visits["filter"])) > 1:
            warnings.warn("Coadding depths in different filters")

        result = self.coadd(visits[self.col])
        return result


class FancyMetric(MeanMetric):
    """Example of returning multiple values in a metric"""

    def __init__(self, col="night"):
        self.shape = None
        self.dtype = list(zip(["mean", "std"], [float, float]))
        self.col = col
        self.empty = np.empty(1, dtype=self.dtype)

    def __call__(self, visits, slice_point=None):
        result = self.empty.copy()
        result["mean"] = np.mean(visits[self.col])
        result["std"] = np.std(visits[self.col])
        return result


class VectorMetric(MeanMetric):
    """Example of returning a vector"""

    def __init__(
        self, times=np.arange(60), col="night", time_col="night", function=np.add
    ):
        self.shape = np.size(times)
        self.dtype = float
        self.col = col
        self.function = function
        self.time_col = time_col
        self.times = times

    def add_info(self, info):
        info["metric: name, MeanMetric"]
        info["metric: times"] = self.times
        return info

    def __call__(self, visits, slice_point):

        visit_times = visits[self.time_col]
        visit_times.sort()
        to_count = np.ones(visit_times.size, dtype=int)
        result = self.function.accumulate(to_count)
        indices = np.searchsorted(visit_times, self.times, side="right")
        indices[np.where(indices >= np.size(result))] = np.size(result) - 1
        result = result[indices]
        return result


class Slicer(object):
    """

    Parameters
    ----------
    lon_col : `str`, optional
        Name of the longitude (RA equivalent) column.
    lat_col : `str`, optional
        Name of the latitude (Dec equivalent) column.
    rot_sky_pos_col_name : `str`, optional
        Name of the rotSkyPos column in the input  data.
        Only used if use_camera is True.
        Describes the orientation of the camera orientation on the sky.
    lat_lon_deg : `bool`, optional
        Flag indicating whether lat and lon values from input data are
        in degrees (True) or radians (False).
    missing : `float`, optional
        Bad value flag, relevant for plotting.
    leafsize : `int`, optional
        Leafsize value for kdtree.
    radius : `float`, optional
        Radius for matching in the kdtree.
        Equivalent to the radius of the FOV, in degrees.
    use_camera : `bool`, optional
        Flag to indicate whether to use the LSST camera footprint or not.
    camera_footprint_file : `str`, optional
        Name of the camera footprint map to use.
        Can be None, which will use the default file.
    """

    def __init__(
        self,
        nside=128,
        lon_col="fieldRA",
        lat_col="fieldDec",
        lat_lon_deg=True,
        leafsize=100,
        radius=2.45,
        use_camera=True,
        camera_footprint_file=None,
        rot_sky_pos_col_name="rotSkyPos",
        missing=np.nan,
        maps=None,
        cache=False,
    ):

        self.nside = int(nside)
        self.pix_area = hp.nside2pixarea(self.nside)
        self.nslice = hp.nside2npix(self.nside)
        self.shape = self.nslice
        self.lat_lon_deg = lat_lon_deg

        self.lon_col = lon_col
        self.lat_col = lat_col
        self.rot_sky_pos_col_name = rot_sky_pos_col_name
        self.use_camera = use_camera
        self.camera_footprint_file = camera_footprint_file
        self.leafsize = leafsize

        self.missing = missing

        self.radius = radius
        self.maps = maps

        self.cache = cache

        # Set up slice_point
        self.slice_points = {}
        self.slice_points["nside"] = nside
        self.slice_points["sid"] = np.arange(self.nslice)
        self.slice_points["ra"], self.slice_points["dec"] = utils._hpid2_ra_dec(
            self.nside, self.slice_points["sid"]
        )

    def __len__(self):
        """Return nslice, the number of slice_points in the slicer."""
        return self.nslice

    def __iter__(self):
        """Iterate over the slices."""
        self.islice = 0
        return self

    def __next__(self):
        """Returns results of self._slice_sim_data when iterating over slicer.

        Results of self._slice_sim_data should be dictionary of
        {'idxs': the data indexes relevant for this slice of the slicer,
        'slice_point': the metadata for the slice_point, which always
        includes 'sid' key for ID of slice_point.}
        """
        if self.islice >= self.nslice:
            raise StopIteration
        islice = self.islice
        self.islice += 1
        return self._slice_sim_data(islice)

    def __getitem__(self, islice):
        return self._slice_sim_data(islice)

    def _run_maps(self, maps):
        """Add map info to slice_points."""
        if maps is not None:
            for m in maps:
                self.slice_points = m.run(self.slice_points)

    def setup_slicer(self, sim_data, maps=None):
        """set up KDTree.

        Parameters
        -----------
        sim_data : `numpy.ndarray`
            The simulated data, including the location of each pointing.
        maps : `list` of `rubin_sim.maf.maps` objects, optional
            List of maps (such as dust extinction) that will run to build up
            additional data at each slice_point. This additional data
            is available to metrics via the slice_point dictionary.
        """
        if maps is not None:
            if self.cache and len(maps) > 0:
                warnings.warn(
                    "Warning:  Loading maps but cache on."
                    "Should probably set use_cache=False in slicer."
                )
            self._run_maps(maps)
        self._set_rad(self.radius)

        if self.lat_lon_deg:
            self.data_ra = np.radians(sim_data[self.lon_col])
            self.data_dec = np.radians(sim_data[self.lat_col])
            self.data_rot = np.radians(sim_data[self.rot_sky_pos_col_name])
        else:
            self.data_ra = sim_data[self.lon_col]
            self.data_dec = sim_data[self.lat_col]
            self.data_rot = sim_data[self.rot_sky_pos_col_name]
        if self.use_camera:
            self._setupLSSTCamera()

        self._build_tree(self.data_ra, self.data_dec, self.leafsize)

        def _slice_sim_data(islice):
            """Return indexes for relevant opsim data at slice_point
            (slice_point=lon_col/lat_col value .. usually ra/dec).
            """

            # Build dict for slice_point info
            slice_point = {"sid": islice}
            sx, sy, sz = utils._xyz_from_ra_dec(
                self.slice_points["ra"][islice], self.slice_points["dec"][islice]
            )
            # Query against tree.
            indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)

            if (self.use_camera) & (len(indices) > 0):
                # Find the indices *of those indices*
                # which fall in the camera footprint
                camera_idx = self.camera(
                    self.slice_points["ra"][islice],
                    self.slice_points["dec"][islice],
                    self.data_ra[indices],
                    self.data_dec[indices],
                    self.data_rot[indices],
                )
                indices = np.array(indices)[camera_idx]

            # Loop through all the slice_point keys.
            # If the first dimension of slice_point[key] has the same shape
            # as the slicer, assume it is information per slice_point.
            # Otherwise, pass the whole slice_point[key] information.
            # Useful for stellar LF maps where we want to pass only the
            # relevant LF and the bins that go with it.
            for key in self.slice_points:
                if len(np.shape(self.slice_points[key])) == 0:
                    keyShape = 0
                else:
                    keyShape = np.shape(self.slice_points[key])[0]
                if keyShape == self.nslice:
                    slice_point[key] = self.slice_points[key][islice]
                else:
                    slice_point[key] = self.slice_points[key]
            return {"idxs": indices, "slice_point": slice_point}

        setattr(self, "_slice_sim_data", _slice_sim_data)

    def _setupLSSTCamera(self):
        """If we want to include the camera chip gaps, etc."""
        self.camera = utils.LsstCameraFootprint(
            units="radians", footprint_file=self.camera_footprint_file
        )

    def _build_tree(self, sim_data_ra, sim_data_dec, leafsize=100):
        """Build KD tree on sim_dataRA/Dec.

        sim_dataRA, sim_dataDec = RA and Dec values (in radians).
        leafsize = the number of Ra/Dec pointings in each leaf node.
        """
        self.opsimtree = utils._build_tree(sim_data_ra, sim_data_dec, leafsize)

    def _set_rad(self, radius=1.75):
        """Set radius (in degrees) for kdtree search."""
        self.rad = utils.xyz_angular_radius(radius)

    def add_info(self, info):
        info["slicer: nside"] = self.nside

        return info

    def __call__(self, input_visits, metric_s, info=None):
        """

        Parameters
        ----------
        input_vistis : `np.array`
            Array with the visit information. If a pandas
            DataFrame gets passed in, it gets converted
            via .to_records method for slicing efficiency.
        metric_s : callable
            A callable function/class or list of callables
            that take an array of visits and slicepoints as
            input
        info : `dict`
            Dict or list of dicts for holding information
            about the analysis process.
        """

        if hasattr(input_visits, "to_records"):
            visits_array = input_visits.to_records(index=False)
        else:
            visits_array = input_visits

        if not isinstance(visits_array, np.ndarray):
            raise ValueError("input_visits should be numpy array or pandas DataFrame.")

        orig_info = copy.copy(info)
        # Construct the KD Tree for this dataset
        self.setup_slicer(visits_array)

        # Check metric_s and info are same length
        if info is not None:
            if isinstance(metric_s, list):
                matching_len = len(metric_s) == len(info)
                if not matching_len:
                    raise ValueError("Length of metric_s must match info length")

        # Naked metric sent in, wrap as a 1-element list
        if not isinstance(metric_s, list):
            metric_s = [metric_s]
            info = [info]

        for metric in metric_s:
            if self.cache:
                if not hasattr(metric, "call_cached"):
                    warnings.warn("Metric does not support cache, turning cache off")
                    self.cache = False
        # XXX-Check if the metric needs any maps loaded or
        # new columns added to the df

        results = []
        final_info = []
        # See what dtype the metric will return,
        # make an array to hold it.
        for metric, single_info in zip(metric_s, info):
            if hasattr(metric, "shape"):
                if metric.shape is None:
                    result = np.empty(self.shape, dtype=metric.dtype)
                else:
                    result = np.empty((self.shape, metric.shape), dtype=metric.dtype)
            else:
                result = np.empty(self.shape, dtype=float)
            result.fill(self.missing)
            results.append(result)

        for i, slice_i in enumerate(self):
            if len(slice_i["idxs"]) != 0:
                slicedata = visits_array[slice_i["idxs"]]
                for j, metric in enumerate(metric_s):
                    if self.cache:
                        results[j][i] = metric.call_cached(
                            frozenset(slicedata["observationId"].tolist()),
                            slicedata,
                            slice_point=slice_i["slice_point"],
                        )
                    else:
                        results[j][i] = metric(
                            slicedata, slice_point=slice_i["slice_point"]
                        )

        if orig_info is not None:
            for single_info, metric in zip(info, metric_s):
                if single_info is not None:
                    single_info = self.add_info(single_info)
                    if hasattr(metric, "add_info"):
                        single_info = metric.add_info(single_info)
                final_info.append(single_info)

        # Unwrap if single metric sent in
        if orig_info is None:
            if len(results) == 1:
                return results[0]
            return results
        else:
            if len(results) == 1:
                return results[0], final_info[0]
            return results, final_info


class BasePlot(object):
    def __init__(self, info=None):

        self.info = info
        self.plot_dict = self._gen_default_labels(info)


class PlotMoll:
    """Plot a mollweild projection of a HEALpix array.
    """
    def __init__(self, info=None):

        self.info = info
        self.moll_kwarg_dict = self._gen_default_labels(info)

    def _gen_default_labels(self, info):
        """ """
        if info is not None:
            result = {}
            if "run_name" in info.keys():
                result["title"] = info["run_name"]
            else:
                result["title"] = ''
            if "observations_subset" in info.keys():
                result["title"] += "\n"+info["observations_subset"]

            if "metric: unit" in info.keys():
                result["unit"] = info["metric: unit"]
        return result

    def default_cb_params(self):
        cb_params = {
            "shrink": 0.75,
            "aspect": 25,
            "pad": 0.1,
            "orientation": "horizontal",
            "format": "%.1f",
            "extendrect": False,
            "extend": "neither",
            "labelsize": None,
            "n_ticks": 10,
            "cbar_edge": True,
            "fontsize": None,
            "label": None,
        }
        return cb_params

    def __call__(
        self,
        inarray,
        fig=None,
        add_grat=True,
        grat_params="default",
        cb_params="default",
        log=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        inarray : `np.array`
            numpy array with proper HEALpix size.
        fig : `matplotlib.Figure`
            A matplotlib figure object. Default of None
            creates a new figure.
        add_grat : `bool`
            Add gratacule to the plot. Default True.
        grat_params : `dict`
            Dictionary of kwargs to pass to healpy.graticule.
            Default of "default" generates a reasonable dict.
        cb_params : `dict`
            Dictionary of color bar parameters. Default of "default"
            uses PlotMoll.default_cb_params to construct defaults. Setting
            to None uses the default healpy colorbar.
        log : `bool`
            Set the colorbar to be log. Default False.
        **kwargs
            Kwargs sent to healpy.mollview. E.g.,
            title, unit, rot, min, max. 

        """
        if fig is None:
            fig = plt.figure()

        if grat_params == "default":
            grat_params = {"dpar": 30, "dmer": 30}
        # Copy any auto-generated plot kwargs
        moll_kwarg_dict = copy.copy(self.moll_kwarg_dict)
        # Override if those things have been set with kwargs
        for key in kwargs:
            moll_kwarg_dict[key] = kwargs.get(key)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if cb_params is None:
                hp.mollview(inarray, **moll_kwarg_dict, fig=fig.number)
            else:
                hp.mollview(inarray, **moll_kwarg_dict, fig=fig.number, cbar=False)

        if add_grat:
            hp.graticule(**grat_params)
        self.ax = plt.gca()
        im = self.ax.get_images()[0]

        if cb_params == "default":
            cb_params = self.default_cb_params()
        else:
            defaults = self.default_cb_params()
            for key in cb_params:
                defaults[key] = cb_params[key]
            cb_params = defaults

        if cb_params["label"] is None:
            cb_params["label"] = moll_kwarg_dict["unit"]

        cb = plt.colorbar(
            im,
            shrink=cb_params["shrink"],
            aspect=cb_params["aspect"],
            pad=cb_params["pad"],
            orientation=cb_params["orientation"],
            format=cb_params["format"],
            extendrect=cb_params["extendrect"],
            extend=cb_params["extend"],
        )
        cb.set_label(cb_params["label"], fontsize=cb_params["fontsize"])

        if cb_params["labelsize"] is not None:
            cb.ax.tick_params(labelsize=cb_params["labelsize"])
        if log:
            tick_locator = ticker.LogLocator(numticks=cb_params["n_ticks"])
            cb.locator = tick_locator
            cb.update_ticks()
        else:
            if cb_params["n_ticks"] is not None:
                tick_locator = ticker.MaxNLocator(nbins=cb_params["n_ticks"])
                cb.locator = tick_locator
                cb.update_ticks()
        # If outputing to PDF, this fixes the colorbar white stripes
        if cb_params["cbar_edge"]:
            cb.solids.set_edgecolor("face")

        return fig


class PlotHist(BasePlot):

    def _gen_ylabel(self):
        return "#"

    def _gen_default_labels(self, info):
        """ """ 
        result = {"ylabel": self._gen_ylabel()}
        result["title"] = ""
        result["xlabel"] = ""
        if info is not None:
            if "run_name" in info.keys():
                result["title"] = info["run_name"]
            else:
                result["title"] = ""
            if "observations_subset" in info.keys():
                result["title"] += "\n"+info["observations_subset"]
            if "metric: unit" in info.keys():
                result["xlabel"] = info["metric: unit"]

        return result

    def __call__(
        self,
        inarray,
        fig=None,
        ax=None,
        title=None,
        xlabel=None,
        ylabel=None,
        histtype="step",
        bins="optimal",
        **kwargs,
    ):
        """
        Parameters
        ----------
        inarray : `np.array`
            Vector to be histogrammed.
        fig : `matplotlib.Figure`
            Matplotlib Figure object to use. Default None
            will generate a new Figure.
        ax : `matplotlib.Axes`
            Matplotlib Axes object to use. Default None
            will generate a new axes
        title,xlabel,ylabel : `str`
            title to set on Axes. Default None will
            use auto-generated title from _gen_default_labels method.
        histtype : `str`
            Histogram type passed to matplotlib.hist. Default `step`.
        bins : `np.array`
            bins passed to matplotlib.hist. Default "optimal" will 
            compute an "optimal" number of bins.
        **kwargs
            Additional keyword arguments passed to `matplotlib.hist`.
            E.g., range, log, align, cumulative, etc. Note histogram
            also passes through matplotlib.Patch properties like
            edgecolor, facecolor, linewidth.
        """

        if isinstance(bins, str):
            if bins == "optimal":
                bins = optimal_bins(inarray)

        overrides = {"title": title, "xlabel": xlabel, "ylabel": ylabel}
        plot_dict = copy.copy(self.plot_dict)
        for key in overrides:
            if overrides[key] is not None:
                plot_dict[key] = overrides[key]

        if fig is None:
            fig, ax = plt.subplots()

        _n, _bins, _patches = ax.hist(inarray, histtype=histtype, bins=bins, **kwargs)
        ax.set_title(plot_dict["title"])
        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])

        return fig


class PlotHealHist(PlotHist):
    """Make a histogram of a HEALpix array
    """
    def __init__(self, info=None, scale=1000):
        self.scale = scale
        super().__init__(info=info)

    def _gen_ylabel(self):
        return "Area (%is of sq degrees)" % self.scale

    def __call__(self, inarray, fig=None, ax=None, histtype="step",
                 bins="optimal", **kwargs):

        pix_area = hp.nside2pixarea(hp.npix2nside(np.size(inarray)),
                                    degrees=True) 
        weights = np.zeros(np.size(inarray)) + pix_area / self.scale
        super().__call__(inarray, fig=fig, ax=ax, histtype=histtype, weights=weights,
                         bins=bins, **kwargs)


class PlotLine(BasePlot):
    def __init__(self, info=None):
        self.info = info
        self.plot_dict = self._gen_default_labels(info)

    def _gen_default_labels(self, info):
        result = {"ylabel": ""}
        result["title"] = ""
        result["xlabel"] = ""
        if info is not None:
            if "run_name" in info.keys():
                result["title"] = info["run_name"]
            else:
                result["title"] = ""
            if "observations_subset" in info.keys():
                result["title"] += "\n"+info["observations_subset"]

        return result

    def _gen_default_grid(self):
        return {"alpha": 0.5}

    def __call__(self, x, y, fig=None, ax=None, title=None,
                 xlabel=None, ylabel=None, grid_params=None, **kwargs):

        if fig is None:
            fig, ax = plt.subplots()

        if grid_params is None:
            grid_params = self._gen_default_grid()

        overrides = {"title": title, "xlabel": xlabel, "ylabel": ylabel}
        plot_dict = copy.copy(self.plot_dict)
        for key in overrides:
            if overrides[key] is not None:
                plot_dict[key] = overrides[key]

        ax.plot(x, y, **kwargs)

        ax.set_title(plot_dict["title"])
        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])
        if isinstance(grid_params, dict):
            ax.grid(**grid_params)

        return fig


def gen_summary_row(info, summary_name, value):
    summary = copy.copy(info)
    summary["summary_name"] = summary_name
    summary["value"] = value
    return summary


def optimal_bins(
    datain, binmin=None, binmax=None, nbin_max=200, nbin_min=1, verbose=False
):
    """
    Set an 'optimal' number of bins using the Freedman-Diaconis rule.

    Parameters
    ----------
    datain : `numpy.ndarray` or `numpy.ma.MaskedArray`
        The data for which we want to set the bin_size.
    binmin : `float`
        The minimum bin value to consider. Default None uses minimum data value.
    binmax : `float`
        The maximum bin value to consider. Default None uses maximum data value.
    nbin_max : `int`
        The maximum number of bins to create.
        Sometimes the 'optimal bin_size' implies an unreasonably large number
        of bins, if the data distribution is unusual. Default 200.
    nbin_min : `int`
        The minimum number of bins to create. Default is 1.
    verbose : `bool`
        Turn on warning messages. Default False.

    Returns
    -------
    nbins : `int`
        The number of bins.
    """
    # if it's a masked array, only use unmasked values
    if hasattr(datain, "compressed"):
        data = datain.compressed()
    else:
        data = datain
    # Check that any good data values remain.
    if data.size == 0:
        nbins = nbin_max
        if verbose:
            warnings.warn(
                f"No unmasked data available for calculating optimal bin size: returning {nbins} bins"
            )
    # Else proceed.
    else:
        if binmin is None:
            binmin = np.nanmin(data)
        if binmax is None:
            binmax = np.nanmax(data)
        cond = np.where((data >= binmin) & (data <= binmax))[0]
        # Check if any data points remain within binmin/binmax.
        if np.size(data[cond]) == 0:
            nbins = nbin_max
            warnings.warn(
                "No data available for calculating optimal bin size within range of %f, %f"
                % (binmin, binmax)
                + ": returning %i bins" % (nbins)
            )
        else:
            iqr = np.percentile(data[cond], 75) - np.percentile(data[cond], 25)
            binwidth = 2 * iqr * (np.size(data[cond]) ** (-1.0 / 3.0))
            nbins = (binmax - binmin) / binwidth
            if nbins > nbin_max:
                if verbose:
                    warnings.warn(
                        "Optimal bin calculation tried to make %.0f bins, returning %i"
                        % (nbins, nbin_max)
                    )
                nbins = nbin_max
            if nbins < nbin_min:
                if verbose:
                    warnings.warn(
                        "Optimal bin calculation tried to make %.0f bins, returning %i"
                        % (nbins, nbin_min)
                    )
                nbins = nbin_min
    if np.isnan(nbins):
        if verbose:
            warnings.warn(
                "Optimal bin calculation calculated NaN: returning %i" % (nbin_max)
            )
        nbins = nbin_max
    return int(nbins)
