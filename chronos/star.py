#!/usr/bin/env python
"""
Stellar characterization module
"""
# Import standard library
from os.path import join
from pathlib import Path

# Import modules
from pprint import pprint
import numpy as np
import matplotlib.pyplot as pl
import lightkurve as lk
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import skewnorm
from astropy.visualization import hist
import astropy.units as u

# Import from package
from chronos.target import Target
from chronos.cluster import ClusterCatalog
from chronos.utils import (
    get_mamajek_table,
    get_mag_err_from_flux,
    get_err_quadrature,
    map_float,
    get_mist_eep_table,
    DATA_PATH,
)


__all__ = ["Star"]

# latest catalogues: GAIA2, APOGEE16, SDSS16, RAVE6, GES3 and GALAH2, ALL-WISE, 2MASS
# Asteroid Terrestrial-impact Last Alert System (ATLAS) and the All-Sky Automated Survey for Supernovae (ASAS-SN)
CATALOGS_STAR_PARMS = {
    "Carillo2020": "https://arxiv.org/abs/1911.07825",  # Gaia+APOGEE14+GALAH+RAVE5+LAMOST+SkyMapper for TESS host stars
    "Queiroz2020": "https://arxiv.org/abs/1710.09970",  # starhorse: using APOGEE+Gaia
    "HardegreeUllman2020": "https://arxiv.org/abs/2001.11511",  # Gaia2+LAMOST for K2 host stars (TICv8)
    # https://iopscience.iop.org/0067-0049/247/1/28/suppdata/apjsab7230t1_mrt.txt
    "Anders2019": "https://arxiv.org/abs/1904.11302",  # panstarrs+2MASS+AllWISE+some APOGEE
    # kinematic thin disc, thick disc, and halo membership probabilities:
    # https://zenodo.org/record/3546184#.Xt-UFIFq1Ol
    # Sloan Digital Sky Survey Apache Point Observatory Galaxy Evolution Experiment (APOGEE)
    "Ahumada2020": "https://arxiv.org/abs/1912.02905",  # SDSS DR16: using APOGEE DR2 -southern+eBOSS spectra
    "Buder2018": "https://arxiv.org/abs/1804.06041",  # GALAH DR2
    "Lin2020": "https://arxiv.org/abs/1911.05221",  # GALAH2=isochrone ages and init bulk met
    # distance- and extinction-corrected CMD, extinction maps as a function of distance, and density maps
    "Guiglion2020": "https://arxiv.org/abs/2004.12666",  # rave w/ CNN
    "McMillan2018": "https://arxiv.org/abs/1707.04554",  # TGAS+RAVE for radial velocities
    "Casey2016": "https://arxiv.org/abs/1609.02914",  # RAVE-on, ages
    "Auge2020": "https://arxiv.org/abs/2003.05459",  # M-giants asteroseismic distances
    "Sanders2018": "https://arxiv.org/abs/1806.02324",  # Isochrone ages for ∼3 million stars
    # ===with planet
    "Wittenmyer2020": "https://arxiv.org/abs/2005.10959",  # K2C1-13+GALAH/HERMES
    "HardegreeUllman2019": "https://arxiv.org/abs/1905.05900",  # keplerMdwarfs
}

#
# https://docs.google.com/document/d/1s6OgiJlBVwonAYvQ3VONB4ioWOn0SHtJesrnaXJETBA/edit
CATALOG_EB = {"": ""}  # parametees of EB in APOGEE16 stars


class Star(Target):
    def __init__(
        self,
        name=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        search_radius=3,
        mission="tess",
        prot=None,
        mcmc_steps=1000,
        burnin=500,
        thin=1,
        alpha=(0.56, 1.05),  # Morris+2020
        slope=(-0.50, 0.17),  # Morris+2020
        sigma_blur=3,
        use_skew_slope=False,
        nsamples=1e4,
        verbose=True,
        clobber=True,
    ):
        """
        Attributes
        ----------
        See inherited class: Target

        See starfit:
        https://github.com/timothydmorton/isochrones/blob/master/isochrones/starfit.py
        """
        # https://docs.python.org/3/library/inspect.html#inspect.getdoc
        super().__init__(
            name=name,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            search_radius=search_radius,
            verbose=verbose,
            clobber=clobber,
            mission=mission,
        )
        self.mcmc_steps = mcmc_steps
        self.burnin = burnin
        self.thin = thin
        self.prot = prot
        self.alpha = alpha
        self.slope = slope
        self.sigma_blur = sigma_blur
        self.use_skew_slope = use_skew_slope
        self.nsamples = int(nsamples)
        self.isochrones_model = None
        self.stardate = None
        self.iso_params = None
        self.iso_param_names = [
            "EEP",
            "log10(Age [yr])",
            "[Fe/H]",
            "ln(Distance)",
            "Av",
        ]
        self.iso_params0 = (329.58, 9.5596, -0.0478, 5.560681631015528, 0.0045)
        self.iso_params_init = {
            k: self.iso_params0[i] for i, k in enumerate(self.iso_param_names)
        }
        self.perc = [16, 50, 84]
        vizier = self.query_vizier(verbose=False)
        self.starhorse = (
            vizier["I/349/starhorse"]
            if "I/349/starhorse" in vizier.keys()
            else None
        )
        self.mist = None
        self.mist_eep_table = get_mist_eep_table()

    def estimate_Av(self, map="sfd", constant=None):
        """
        compute the extinction Av from color index E(B-V)
        estimated from dustmaps via Av=constant*E(B-V)

        Parameters
        ----------
        map : str
            dust map
        See below for conversion from E(B-V) to Av:
        https://dustmaps.readthedocs.io/en/latest/examples.html
        """
        try:
            import dustmaps
        except Exception:
            raise ModuleNotFoundError("pip install dustmaps")

        if map == "sfd":
            from dustmaps import sfd

            # sfd.fetch()
            dust_map = sfd.SFDQuery()
            constant = 2.742 if constant is None else constant
        elif map == "planck":
            from dustmaps import planck

            # planck.fetch()
            dust_map = planck.PlanckQuery()
            constant = 3.1 if constant is None else constant
        elif map == "bayestar":
            from dustmaps import bayestar

            bayestar.BayestarQuery()
            dust_map = 2.742 if constant is None else constant
        else:
            raise ValueError(f"Available maps: (sfd,planck,bayestar)")

        ebv = dust_map(self.target_coord)
        Av = constant * ebv
        return Av

    def get_spectral_type(
        self,
        columns="Teff B-V J-H H-Ks".split(),
        nsamples=int(1e4),
        return_samples=False,
        plot=False,
        clobber=False,
    ):
        """
        Interpolate spectral type from Mamajek table from
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        based on observables Teff and color indices.
        c.f. self.query_vizier_param("SpT")

        Parameters
        ----------
        columns : list
            column names of input parameters
        nsamples : int
            number of Monte Carlo samples (default=1e4)
        clobber : bool (default=False)
            re-download Mamajek table

        Returns
        -------
        interpolated spectral type

        Notes:
        It may be good to check which color index yields most accurate result

        Check sptype from self.query_simbad()
        """
        df = get_mamajek_table(clobber=clobber, verbose=self.verbose)
        if self.gaia_params is None:
            self.gaia_params = self.query_gaia_dr2_catalog(
                return_nearest_xmatch=True
            )
        if self.tic_params is None:
            self.tic_params = self.query_tic_catalog(
                return_nearest_xmatch=True
            )

        # effective temperature
        col = "teff"
        teff = self.gaia_params[f"{col}_val"]
        siglo = (
            self.gaia_params[f"{col}_val"]
            - self.gaia_params[f"{col}_percentile_lower"]
        )
        sighi = (
            self.gaia_params[f"{col}_percentile_upper"]
            - self.gaia_params[f"{col}_val"]
        )
        uteff = np.sqrt(sighi ** 2 + siglo ** 2)
        s_teff = (
            teff + np.random.randn(nsamples) * uteff
        )  # Monte Carlo samples

        # B-V color index
        bv_color = self.tic_params["Bmag"] - self.tic_params["Vmag"]
        ubv_color = (
            self.tic_params["e_Bmag"] + self.tic_params["e_Vmag"]
        )  # uncertainties add
        s_bv_color = (
            bv_color + np.random.randn(nsamples) * ubv_color
        )  # Monte Carlo samples

        # J-H color index
        jh_color = self.tic_params["Jmag"] - self.tic_params["Hmag"]
        ujh_color = (
            self.tic_params["e_Jmag"] + self.tic_params["e_Hmag"]
        )  # uncertainties add
        s_jh_color = (
            jh_color + np.random.randn(nsamples) * ujh_color
        )  # Monte Carlo samples

        # H-K color index
        hk_color = self.tic_params["Hmag"] - self.tic_params["Kmag"]
        uhk_color = (
            self.tic_params["e_Hmag"] + self.tic_params["e_Kmag"]
        )  # uncertainties add
        s_hk_color = (
            hk_color + np.random.randn(nsamples) * uhk_color
        )  # Monte Carlo samples

        # Interpolate
        interp = NearestNDInterpolator(
            df[columns].values, df["#SpT"].values, rescale=False
        )
        samples = interp(s_teff, s_bv_color, s_jh_color, s_hk_color)
        # encode category
        spt_cats = pd.Series(samples, dtype="category")  # .cat.codes
        spt = spt_cats.mode().values[0]
        if plot:
            nbins = np.unique(samples)
            pl.hist(samples, bins=nbins)
        if return_samples:
            return spt, samples
        else:
            return spt

    def get_age(
        self,
        lc=None,
        prot=None,
        amp=None,
        method="isochrones",
        return_samples=False,
        burnin=None,
        plot=False,
    ):
        """
        Parameters
        ----------
        method : str
            (default) isochrones
        """
        burnin = burnin if burnin is not None else self.burnin
        method = method if method is not None else "isochrones"
        if method == "isochrones":
            age, errp, errm, samples = self.get_age_from_color(
                return_samples=True, burnin=burnin, plot=plot
            )
        elif method == "prot":
            age, errp, errm, samples = self.get_age_from_rotation_period(
                prot=prot, return_samples=True, plot=plot
            )
        elif method == "amp":
            age, errp, errm, samples = self.get_age_from_rotation_amplitude(
                lc=lc, prot=prot, amp=amp, return_samples=True, plot=plot
            )
        else:
            msg = "Use method=[isochrones,prot,amp]"
            raise ValueError(msg)
        if return_samples:
            return age, errp, errm, samples
        else:
            return age, errp, errm

    def get_rotation_amplitude(self, lc, prot):
        """
        Smoothed rotation amplitude: See Morris+2020
        and references therein

        Parameters
        ----------
        lc : lk.lightcurve
                lightcurve to be folded
        prot : tuple
            rotation period
        """
        assert (lc is not None) & (prot is not None)
        assert isinstance(lc, lk.LightCurve)
        assert isinstance(prot, tuple), "prot should be a tuple (value,error)"

        amps = []
        prot_s = prot[0] + np.random.randn(self.nsamples) * prot[1]
        for p in prot_s:
            fold = lc.fold(period=p)
            # smooth the phase-folded lightcurve
            lc_blur = gaussian_filter(fold.flux, sigma=self.sigma_blur)
            amps.append(max(lc_blur) - min(lc_blur))
        return (np.median(amps), np.std(amps))

    def plot_age_vs_rotation_amplitude(self, min_age_Gyr=0.01, max_age_Gyr=4):
        """
        Fig. 3 in Morris+2020: https://arxiv.org/abs/2002.09135

        alpha error is quad(-0.31,+1.00)
        See
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html
        """
        t = np.linspace(min_age_Gyr, max_age_Gyr, self.nsamples)  # Byr
        alpha_s = (
            self.alpha[0] + np.random.randn(self.nsamples) * self.alpha[1]
        )
        if self.use_skew_slope:
            slope_s = skewnorm.rvs(-8, size=self.nsamples)
        else:
            slope_s = (
                self.slope[0] + np.random.randn(self.nsamples) * self.slope[1]
            )
        # make sure slope is non-positive
        slope_s = [i if i <= 0 else 0 for i in slope_s]

        A_s = alpha_s * t ** slope_s
        # remove out of sample/ replace with median value
        A_s = [i if i < 20 else np.nan for i in A_s]
        A_s = [i if i > 0.01 else np.nan for i in A_s]

        A = np.median(alpha_s) * t ** np.median(slope_s)
        pl.loglog(t, A_s, "k.", alpha=0.1)
        pl.loglog(t, A, "r-", label="median")
        pl.xlabel("Age [Gyr]")
        pl.ylabel("Rotation amplitude [%]")
        pl.legend()
        pl.title("Model from Morris+2020")

    def plot_skew_norm(self, a=-8, min_age_Gyr=0.01, max_age_Gyr=4):
        t = np.linspace(min_age_Gyr, max_age_Gyr, self.nsamples)  # Byr
        mean, var, skew, kurt = skewnorm.stats(a, moments="mvsk")
        print(mean, var, skew, kurt)

        rv = skewnorm(a)
        pl.plot(t, rv.pdf(t), "k-", lw=2, label="frozen pdf")
        pl.hist(
            skewnorm.rvs(a, size=int(1e4)),
            bins=30,
            normed=True,
            label="samples",
        )
        pl.axvline(-0.56 - 0.31, ls="--", c="k")
        pl.axvline(-0.56, ls="-", c="k")
        pl.axvline(-0.56 + 1.0, ls="--", c="k")

    def get_age_from_rotation_amplitude(
        self,
        lc=None,
        prot=None,
        amp=None,
        min_age_Gyr=0.01,
        max_age_Gyr=4,
        return_samples=False,
        plot=False,
    ):
        """
        Parameters
        ----------
        lc : lk.LightCurve
            lightcurve object that contains time and flux
        prot : tuple
            rotation period
        amp : tuple
            rotation amplitude; will be estimated if None

        Notes:
        ------
        Useful for getting upper limit (1-sigma) on age

        Model is based on Morris+2020:
        A[%]=a*t[Byr]**m, where
        a = (0.56,+1,-0.3)
        m = (-0.5+/-0.17)
        """
        if self.verbose:
            print("Estimating age using rotation amplitude\n")

        if amp is not None:
            errmsg = "amp should be a tuple (value,error)"
            assert isinstance(amp, tuple), errmsg
            if (amp[0] > 0.2) | (amp[1] > 0.2):
                print(f"amplitude is {amp[0]:.2f}*100%!")
        else:
            # estimate rotation period amplitude
            amp = self.get_rotation_amplitude(lc=lc, prot=prot)
        # convert to percent
        amp = (amp[0] * 100, amp[1] * 100)

        alpha_s = (
            self.alpha[0] + np.random.randn(self.nsamples) * self.alpha[1]
        )
        if self.use_skew_slope:
            slope_s = skewnorm.rvs(-8, size=self.nsamples)
        else:
            slope_s = (
                self.slope[0] + np.random.randn(self.nsamples) * self.slope[1]
            )
        # make sure slope is non-positive
        slope_s = np.array([i if i <= 0 else 0 for i in slope_s])
        A_s = amp[0] + np.random.randn(self.nsamples) * amp[1]

        age_s = (A_s / alpha_s) ** (1 / slope_s)  # in Gyr
        age_s = np.array([i if i > min_age_Gyr else np.nan for i in age_s])
        age_s = np.array([i if i < max_age_Gyr else np.nan for i in age_s])
        age_samples = age_s * 1e9  # convert Gyr to yr
        # remove out of sample
        idx = np.isnan(age_samples)
        fraction = sum(idx) / len(age_samples)
        if fraction > 0.3:
            print(
                f"More than {fraction*100:.2f}% of derived ages is outside (0.01,4) Gyr"
            )
        age_samples = age_samples[~idx]
        siglo, mid, sighi = np.percentile(age_samples, self.perc)
        errm = mid - siglo
        errp = sighi - mid
        if self.verbose:
            print(
                f"gyro age = {mid/1e6:.2f} + {errp/1e6:.2f} - {errm/1e6:.2f} Myr using rotation amplitude {amp[0]:.2f}+/-{amp[1]:.2f}%"
            )
        if plot:
            hist(age_samples / 1e6, bins="knuth")
            pl.axvline((mid + errp) / 1e6, 0, 1, ls="--", c="k")
            pl.axvline(mid / 1e6, 0, 1, ls="-", c="k")
            pl.axvline((mid - errm) / 1e6, 0, 1, ls="--", c="k")
            pl.title("Age from rotation amplitude")
            pl.xlabel("Age [Myr]")
            x1, x2 = pl.gca().get_xlim()
            x1 = 10 if x1 < 10 else x1
            x2 = 4e3 if x2 > 4e3 else x2
            pl.xlim(x1, x2)
        if return_samples:
            return (mid, errp, errm, age_samples)
        else:
            return (mid, errp, errm)

    def get_age_from_rotation_period(
        self, prot=None, return_samples=False, plot=False
    ):
        """
        See https://ui.adsabs.harvard.edu/abs/2019AJ....158..173A/abstract

        Parameters
        ----------
        prot : tuple
            stellar rotation period
        """
        try:
            import stardate as sd
        except Exception:
            raise ModuleNotFoundError("pip install stardate")

        if self.verbose:
            print("Estimating age from rotation period\n")

        prot = prot if prot is not None else self.prot
        if self.gaia_params is None:
            _ = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
        if self.tic_params is None:
            _ = self.query_tic_catalog(return_nearest_xmatch=True)
        if not self.validate_gaia_tic_xmatch():
            msg = f"TIC {self.ticid} does not match Gaia DR2 {self.gaiaid} properties"
            raise Exception(msg)

        prot = prot if prot is not None else self.prot
        if prot is not None:
            errmsg = "prot should be a tuple (value,error)"
            assert isinstance(prot, tuple), errmsg

        if self.verbose:
            print(f"Estimating age using gyrochronology\n")

        prot_samples = prot[0] + np.random.randn(self.nsamples) * prot[1]
        log10_period_samples = np.log10(prot_samples)

        bprp = self.gaia_params["bp_rp"]  # Gaia BP - RP color.
        bprp_err = 0.1
        bprp_samples = bprp + np.random.randn(self.nsamples) * bprp_err

        log10_age_yrs = np.array(
            [
                sd.lhf.age_model(x, y)
                for x, y in zip(log10_period_samples, bprp_samples)
            ]
        )
        age_samples = 10 ** log10_age_yrs
        siglo, mid, sighi = np.percentile(age_samples, self.perc)
        errm = mid - siglo
        errp = sighi - mid
        if self.verbose:
            print(
                f"gyro age = {mid/1e6:.2f} + {errp/1e6:.2f} - {errm/1e6:.2f} Myr using rotation period {prot[0]:.2f}+/-{prot[1]:.2f}d"
            )
        if plot:
            hist(age_samples / 1e6, bins="knuth")
            pl.axvline((mid + errp) / 1e6, 0, 1, ls="--", c="k")
            pl.axvline(mid / 1e6, 0, 1, ls="-", c="k")
            pl.axvline((mid - errm) / 1e6, 0, 1, ls="--", c="k")
            pl.title("Age from rotation period")
            pl.xlabel("Age [Myr]")
            x1, x2 = pl.gca().get_xlim()
            x1 = 10 if x1 < 10 else x1
            x2 = 5e3 if x2 > 5e3 else x2
            pl.xlim(x1, x2)
        if return_samples:
            return (mid, errp, errm, age_samples)
        else:
            return (mid, errp, errm)

    def get_iso_params(
        self,
        teff=None,
        logg=None,
        feh=None,
        add_parallax=True,
        add_dict=None,
        bands="G BP RP J H K W1 W2 W3 TESS".split(),
        min_mag_err=0.01,
        inflate_plx_err=True,
    ):
        """get parameters for isochrones

        Parameters
        ----------
        teff, logg, feh: tuple
            'gaia' populates Teff from gaia DR2
        bands : list
            list of photometric bands
        add_parallax : bool
            default=True
        add_dict : dict
            additional params
        min_mag_err : float
            minimum magnitude error
        inflate_plx_err : bool
            adds 0.01 parallax error in quadrature (default=True),
            prescription by Luri+2018

        Returns
        -------
        iso_params : dict
        """
        errmsg = "`bands must be a list`"
        assert isinstance(bands, list), errmsg
        if self.gaia_params is None:
            gp = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
        else:
            gp = self.gaia_params
        if self.tic_params is None:
            tp = self.query_tic_catalog(return_nearest_xmatch=True)
        else:
            tp = self.tic_params
        if not self.validate_gaia_tic_xmatch():
            msg = f"TIC {self.ticid} does not match Gaia DR2 {self.gaiaid} properties"
            raise Exception(msg)

        params = {}
        # spectroscopic constraints
        if teff == "gaia":
            # Use Teff from Gaia by default
            teff = gp["teff_val"]
            teff_err = get_err_quadrature(
                gp["teff_percentile_lower"], gp["teff_percentile_lower"]
            )
            if not np.any(np.isnan(map_float((tp["Teff"], tp["e_Teff"])))):
                if teff_err > tp["e_Teff"]:
                    # use Teff from TIC if Teff error is smaller
                    teff = tp["Teff"]
                    teff_err = tp["e_Teff"]
            params.update({"teff": (teff, teff_err)})
        elif teff is not None:
            assert isinstance(
                teff, tuple
            ), "teff must be a tuple (value,error)"
            teff, teff_err = teff[0], teff[1]
            params.update({"Teff": (teff, teff_err)})
        if feh is not None:
            # params.update({"feh": (tp["MH"], tp["e_MH"])})
            assert isinstance(feh, tuple), "feh must be a tuple (value,error)"
            params.update({"feh": (feh[0], feh[1])})
        if logg is not None:
            # params.update({"logg": (tp["logg"], tp["e_logg"])})
            assert isinstance(
                logg, tuple
            ), "logg must be a tuple (value,error)"
            params.update({"logg": (logg[0], logg[1])})
        if add_parallax:
            plx = gp["parallax"]
            if inflate_plx_err:
                # inflate error based on Luri+2018
                plx_err = get_err_quadrature(gp["parallax_error"], 0.1)
            else:
                plx_err = gp["parallax_error"]
            params.update({"parallax": (plx, plx_err)})

        # get magnitudes from vizier
        mags = self.query_vizier_mags()
        if (self.ticid is not None) and (self.toi_params is not None):
            Tmag_err = (
                self.toi_Tmag_err
                if self.toi_Tmag_err > min_mag_err
                else min_mag_err
            )
            if "TESS" in bands:
                params.update({"TESS": (self.toi_Tmag, Tmag_err)})
        if "G" in bands:
            gmag = gp["phot_g_mean_mag"]
            gmag_err = get_mag_err_from_flux(
                gp["phot_g_mean_flux"], gp["phot_g_mean_flux_error"]
            )
            gmag_err = gmag_err if gmag_err > min_mag_err else min_mag_err
            params.update({"G": (gmag, gmag_err)})
        if "BP" in bands:
            bpmag = gp["phot_bp_mean_mag"]
            bpmag_err = get_mag_err_from_flux(
                gp["phot_bp_mean_flux"], gp["phot_bp_mean_flux_error"]
            )
            bpmag_err = bpmag_err if bpmag_err > min_mag_err else min_mag_err
            params.update({"BP": (bpmag, bpmag_err)})
        if "RP" in bands:
            rpmag = gp["phot_rp_mean_mag"]
            rpmag_err = get_mag_err_from_flux(
                gp["phot_rp_mean_flux"], gp["phot_rp_mean_flux_error"]
            )
            rpmag_err = rpmag_err if rpmag_err > min_mag_err else min_mag_err
            params.update({"RP": (rpmag, rpmag_err)})
        if "B" in bands:
            # from tic catalog
            params.update({"B": (tp["Bmag"], tp["e_Bmag"])})
        if "V" in bands:
            # from tic catalog
            params.update({"V": (tp["Vmag"], tp["e_Vmag"])})
        if "J" in bands:
            # from tic catalog
            params.update({"J": (tp["Jmag"], tp["e_Jmag"])})
        if "H" in bands:
            # from tic catalog
            params.update({"H": (tp["Hmag"], tp["e_Hmag"])})
        if "K" in bands:
            # from tic catalog
            params.update({"K": (tp["Kmag"], tp["e_Kmag"])})
        # WISE
        for b in bands:
            if b[0] == "W":
                if f"{b}mag" in mags.index.tolist():
                    wmag = mags[f"{b}mag"]
                    wmag_err = mags[f"e_{b}mag"]
                    wmag_err = (
                        wmag_err if wmag_err > min_mag_err else min_mag_err
                    )
                    params.update({b: (round(wmag, 2), round(wmag_err, 2))})
                else:
                    print(f"{b} not in {mags.index.tolist()}")
            elif b[:2] == "Kp":
                if f"{b}mag" in mags.index.tolist():
                    wmag = mags[f"{b}mag"]
                    wmag_err = mags[f"e_{b}mag"]
                    wmag_err = (
                        wmag_err if wmag_err > min_mag_err else min_mag_err
                    )
                    params.update({b: (round(wmag, 2), round(wmag_err, 2))})
                else:
                    print(f"{b} not in {mags.index.tolist()}")

        # adds and/or overwrites above
        if add_dict is not None:
            assert isinstance(add_dict, dict)
            params.update(add_dict)
        # remove nan if there is any
        iso_params = {}
        for k in params:
            vals = map_float(params[k])
            if np.any(np.isnan(vals)):
                print(f"{k} is ignored due to nan ({vals})")
            else:
                iso_params[k] = vals
        self.iso_params = iso_params
        return iso_params

    def save_star_ini(self, outdir="."):
        """star.ini file for isochrones starfit script
        See:
        https://github.com/timothydmorton/isochrones/blob/master/README.rst
        """
        target_name = self.target_name.replace(" ", "")
        if self.iso_params is None:
            iso_params = self.get_iso_params()
        else:
            iso_params = self.iso_params
        starfit_arr = []
        for k in iso_params:
            vals = map_float(iso_params[k])
            if np.any(np.isnan(vals)):
                print(f"{k} is ignored due to nan ({vals})")
            else:
                if k in ["Teff", "parallax", "logg", "feh"]:
                    par = k
                else:
                    par = k.upper()
                starfit_arr.append(f"{par} = {vals[0]}, {vals[1]}")
        outdir = target_name if outdir == "." else outdir
        outpath = Path(outdir, "star.ini")
        if not Path(outdir).exists():
            Path(outdir).mkdir()
        np.savetxt(outpath, starfit_arr, fmt="%2s", header=target_name)
        print(f"Saved: {outpath}\n{starfit_arr}")

    def save_fpp_ini(self, outdir=".", fpp_params=None):
        """fpp.ini file for vespa calcfpp script
        See:
        https://github.com/timothydmorton/VESPA/blob/master/README.rst
        """
        target_name = self.target_name.replace(" ", "")

        fpp_arr = []
        fpp_arr.append(f"name = {target_name}")
        fpp_arr.append(f"ra = {self.target_coord.ra.deg:.4f}")
        fpp_arr.append(f"dec = {self.target_coord.dec.deg:.4f}")

        period = (
            fpp_params["period"] if fpp_params is not None else self.toi_period
        )
        if period is None:
            print("Manually append 'period' to file")
        else:
            fpp_arr.append(f"period = {period:.4f}")
        depth = (
            fpp_params["depth"] if fpp_params is not None else self.toi_depth
        )
        if depth is None:
            print("Manually append 'rprs' to file")
        else:
            fpp_arr.append(f"rprs = {depth:.4f}")
        if self.mission.lower() == "tess":
            fpp_arr.append(f"cadence = {30*u.minute.to(u.day):.2f}")
            fpp_arr.append(f"band = TESS")
        else:
            fpp_arr.append(f"cadence = {30*u.minute.to(u.day):.2f}")
            fpp_arr.append(f"band = Kepler")
        print("Double check entries for cadence and band.")
        fpp_arr.append(f"photfile = {target_name}-lc-folded.txt")
        fpp_arr.append("[constraints]")
        if self.mission.lower() == "tess":
            fpp_arr.append(f"maxrad = 60.0")  # arcsec
        else:
            fpp_arr.append(f"maxrad = 12.0")  # arcsec
        secthresh = fpp_params["secthresh"] if fpp_params is not None else None
        if depth is None:
            print("Manually append 'secthresh' to file")
        else:
            fpp_arr.append(f"secthresh = {secthresh}")
        outdir = target_name if outdir == "." else outdir
        outpath = Path(outdir, "fpp.ini")
        if not Path(outdir).exists():
            Path(outdir).mkdir()
        np.savetxt(outpath, fpp_arr, fmt="%2s", header=target_name)
        print(f"Saved: {outpath}\n{fpp_arr}")

    def init_isochrones(
        self,
        iso_params=None,
        model="mist",
        maxAV=None,
        max_distance=None,
        bands=None,
        binary_star=False,
    ):
        """initialize parameters for isochrones

        Parameters
        ----------
        iso_params : dict
            isochrone input
        model : str
            stellar evolution model grid (default=mist)
        maxAV : float
            maximum extinction [mag]
        max_distance : float
            maximum distance [pc]
        binary_star : bool
            use binary star model if True else False (default=False)
        Returns
        -------
        isochrones_model
        """
        try:
            from isochrones import (
                get_ichrone,
                SingleStarModel,
                BinaryStarModel,
            )
        except Exception:
            cmd = "pip install isochrones\n"
            cmd = "You may want to also install pymultinest for Nested Sampling.\n"
            cmd = "See https://github.com/JohannesBuchner/PyMultiNest"
            raise ModuleNotFoundError(cmd)

        mist = get_ichrone(model, bands=bands)
        self.mist = mist
        iso_params = (
            self.get_iso_params() if iso_params is None else iso_params
        )
        if self.verbose:
            print(iso_params)
        if binary_star:
            model = BinaryStarModel
        else:
            model = SingleStarModel
        self.isochrones_model = model(
            self.mist,
            maxAV=maxAV,
            max_distance=max_distance,
            ra=self.target_coord.ra.deg,
            dec=self.target_coord.dec.deg,
            name=self.target_name,
            **iso_params,
        )
        # set mass upper limit up to 10 Msol
        self.isochrones_model.set_bounds(mass=(0.1, 10))
        # set eep upper limit up to asymptotic giant branch
        self.isochrones_model.set_bounds(eep=(0, 808))
        return self.isochrones_model

    def run_isochrones(
        self, iso_params=None, binary_star=False, overwrite=False, **kwargs
    ):
        """
        Parameters
        ----------
        iso_params : dict
            isochrone input
        binary_star : bool
            use binary star model if True else False (default=False)
        overwrite : bool
            re-run isochrones from scratch
        Returns
        -------
        isochrones_model
        Note:
        * Use `init_isochones` for detailed isochrones model initialization.
        https://isochrones.readthedocs.io/en/latest/quickstart.html#Fit-physical-parameters-of-a-star-to-observed-data

        * See mod._priors for priors; for multi-star systems, see
        https://isochrones.readthedocs.io/en/latest/multiple.html
        FIXME: nsteps param in mod.fit() cannot be changed
        """
        # Create a dictionary of observables
        if self.mist is None:
            iso_params = (
                self.get_iso_params() if iso_params is None else iso_params
            )
            self.init_isochrones(iso_params=iso_params)
        else:
            print("Using previously initialized model.")

        model = self.isochrones_model
        if model._samples is not None:
            if not overwrite:
                if self.verbose:
                    print(
                        "Loading previous samples. Otherwise, try overwrite=True."
                    )
            else:
                if self.verbose:
                    print("Overwriting previous run.")
        if model.use_emcee:
            print("Method: Affine-invariant MCMC")
            # kwargs = {"niter": int(nsteps)}
        else:
            print("Method: Nested Sampling")
            # kwargs = {"n_live_points": int(nsteps)}
        # fit
        try:
            logprior0 = model.lnprior(self.iso_params0)
            loglike0 = model.lnlike(self.iso_params0)
            logpost0 = model.lnpost(self.iso_params0)
            msg = "Initial values:\n"
            msg += f"logpost=loglike+logprior = "
            msg += f"{loglike0:.2f} + {logprior0:.2f} = {logpost0:.2f}"
            if self.verbose:
                print(msg)
        except Exception as e:
            errmsg = f"Error: {e}\n"
            errmsg += "Error in calculating logprior. Check `iso_params` input values."
            raise ValueError(errmsg)

        # nsteps = nsteps if nsteps is not None else self.mcmc_steps
        model.fit(overwrite=overwrite, **kwargs)

        # Note: median!=MAP
        # iso_params0_ = model.samples.median().values
        iso_params0_ = model.map_pars
        logprior = model.lnprior(iso_params0_)
        loglike = model.lnlike(iso_params0_)
        logpost = model.lnpost(iso_params0_)
        msg = "Final values:\n"
        msg += f"logpost=loglike+logprior = "
        msg += f"{loglike:.2f} + {logprior:.2f} = {logpost:.2f}"
        if self.verbose:
            print(msg)
        if not model.use_emcee:
            print(f"Model evidence: {model.evidence}")
        return model

    def get_isochrones_prior_samples(self, nsamples=int(1e4)):
        """sample default priors

        Returns dataframe
        """
        model = self.isochrones_model
        errmsg = "self.run_isochrones"
        assert model is not None, errmsg
        samples = {}
        for param in model._priors:
            if param == "eep":
                age, feh = self.iso_params0[1], self.iso_params0[2]
                samples[param] = model._priors[param].sample(
                    nsamples, age=age, feh=feh
                )
            else:
                samples[param] = model._priors[param].sample(nsamples)
        return pd.DataFrame(samples)

    def plot_isochrones_priors(self, kind="kde"):
        """plot default priors

        TODO: add units and prior name
        ChabrierPrior: LogNormalPrior+PowerLawPrior
        FehPrior: feh PDF based on local SDSS distribution
        AgePrior: FlatLogPrior, log10(age)
        DistancePrior: PowerLawPrior
        AVPrior: FlatPrior
        EEP_prior: BoundedPrior (See self.mist_eep_table)
        """

        fig, axs = pl.subplots(2, 3, figsize=(8, 8), constrained_layout=True)
        ax = axs.flatten()

        df = self.get_isochrones_prior_samples()
        for i, col in enumerate(df.columns):
            _ = df[col].plot(kind=kind, ax=ax[i])
            ax[i].set_title(col)
            xlims = self.isochrones_model._priors[col].bounds
            if i not in [0, 3]:
                ax[i].set_ylabel("")
            if np.isfinite(xlims).any():
                ax[i].set_xlim(xlims)
        fig.suptitle("Priors")
        # fig.subplots_adjust(wspace=0.1)
        return fig

    def plot_posterior_eep(self):
        """
        """
        errmsg = "try self.run_isochrones()"
        assert self.isochrones_model._samples is not None, errmsg
        emin = self.isochrones_model.derived_samples.eep.min() - 100
        emax = self.isochrones_model.derived_samples.eep.max() + 100

        idx = self.mist_eep_table["EEP Number"].between(emin, emax)
        tab = self.mist_eep_table.loc[idx, ["EEP Number", "Phase"]]

        # plot kde
        ax = self.isochrones_model.derived_samples.eep.plot(kind="kde")
        n = 1
        for _, row in tab.iterrows():
            ax.axvline(
                row["EEP Number"], 0, 1, label=row["Phase"], ls="--", c=f"C{n}"
            )
            n += 1
        ax.set_xlabel("Equal Evolutionary Point")
        ax.set_title(self.target_name)
        ax.legend()
        return ax

    # @classmethod
    def get_isochrones_results(self):
        if self.isochrones_model is not None:
            return self.isochrones_model.derived_samples
        else:
            raise ValueError("Try self.run_isochrones()")

    # @classmethod
    def get_isochrones_results_summary(self):
        if self.isochrones_model is not None:
            return self.isochrones_model.derived_samples.describe()
        else:
            raise ValueError("Try self.run_isochrones()")

    def run_stardate(
        self,
        prot=None,
        iso_params=None,
        iso_params0=None,
        min_Av_err=0.01,
        optimize=False,
        nsteps=None,
        nburn=None,
        nthin=None,
    ):
        """
        Parameters
        ----------
        iso_params : dict
            isochrones parameters
        iso_params0 : list
            isochrones initial guesses
        optimize : bool
            optimize first before MCMC
        Returns None
        """
        try:
            import stardate as sd
        except Exception:
            cmd = "pip install git+https://github.com/RuthAngus/stardate.git#egg=stardate"
            raise ModuleNotFoundError(cmd)

        prot = prot if prot is not None else self.prot
        nsteps = nsteps if nsteps is not None else self.mcmc_steps
        nburn = nburn if nburn is not None else self.burnin
        nthin = nthin if nthin is not None else self.thin

        if prot is not None:
            errmsg = "prot should be a tuple (value,error)"
            assert isinstance(prot, tuple), errmsg
            prot, prot_err = prot[0], prot[1]
        else:
            prot, prot_err = None, None

        # Create a dictionary of observables
        iso_params = (
            self.get_iso_params() if iso_params is None else iso_params
        )
        # Init guesses
        iso_params0 = self.iso_params0 if iso_params0 is None else iso_params0

        # estimate extinction
        Av, Av_err = (self.estimate_Av(), min_Av_err)
        # Set up the star object.
        self.stardate = sd.Star(
            iso_params, prot=prot, prot_err=prot_err, Av=Av, Av_err=Av_err
        )
        if self.verbose:
            # add Av in dict for printing only
            iso_params.update({"Av": [Av, Av_err]})
            print("Input parameters:")
            pprint(iso_params)
            print("Init isochrones parameters:")
            pprint(self.iso_params_init)

        # Run the MCMC
        self.stardate.fit(
            inits=iso_params0,
            max_n=nsteps,
            thin_by=nthin,
            burnin=nburn,
            optimize=optimize,
        )
        self.iso_params0 = iso_params0

    def get_age_from_color(
        self, burnin=None, return_samples=False, plot=False
    ):
        """
        See https://ui.adsabs.harvard.edu/abs/2019AJ....158..173A/abstract

        """
        if self.verbose:
            print("Estimating age from isochrones\n")

        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate
        burnin = burnin if burnin is not None else self.burnin

        # Print the median age with the 16th and 84th percentile uncertainties.
        if self.verbose:
            _, _, _, age_samples = star.age_results(
                burnin=burnin
            )  # in log10(age/yr)
            age_samples = 10 ** age_samples
            siglo, mid, sighi = np.percentile(age_samples, self.perc)
            errp = sighi - mid
            errm = mid - siglo
            print(
                f"iso+gyro age = {mid/1e6:.2f} + {errp/1e6:.2f} - {errm/1e6:.2f} Myr"
            )
        if plot:
            hist(age_samples / 1e6, bins="knuth")
            pl.axvline((mid + errp) / 1e6, 0, 1, ls="--", c="k")
            pl.axvline(mid / 1e6, 0, 1, ls="-", c="k")
            pl.axvline((mid - errm) / 1e6, 0, 1, ls="--", c="k")
            pl.title("Age from isochrones")
            pl.xlabel("Age [Myr]")
            x1, x2 = pl.gca().get_xlim()
            x1 = 10 if x1 < 10 else x1
            x2 = 5e3 if x2 > 5e3 else x2
            pl.xlim(x1, x2)
        if return_samples:
            return (mid, errp, errm, age_samples)
        else:
            return (mid, errp, errm)

    def get_mass(self, burnin=None, return_samples=False):
        """get max aposteriori mass from stardate samples"""
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate
        burnin = burnin if burnin is not None else self.burnin
        mass, mass_errp, mass_errm, mass_samples = star.mass_results(
            burnin=burnin
        )
        print(
            "Mass = {0:.2f} + {1:.2f} - {2:.2f} M_sun".format(
                mass, mass_errp, mass_errm
            )
        )
        if return_samples:
            return (mass, mass_errp, mass_errm, mass_samples)
        else:
            return (mass, mass_errp, mass_errm)

    def get_feh(self, burnin=None, return_samples=False):
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate
        burnin = burnin if burnin is not None else self.burnin
        feh, feh_errp, feh_errm, feh_samples = star.feh_results(burnin=burnin)
        print(
            "feh = {0:.2f} + {1:.2f} - {2:.2f}".format(feh, feh_errp, feh_errm)
        )
        if return_samples:
            return (feh, feh_errp, feh_errm, feh_samples)
        else:
            return (feh, feh_errp, feh_errm)

    def get_distance(self, burnin=None, return_samples=False):
        """get max aposteriori age from stardate samples"""
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate
        burnin = burnin if burnin is not None else self.burnin
        (lnd, lnd_errp, lnd_errm, lnd_samples) = star.distance_results(
            burnin=burnin
        )
        print(
            "ln(distance) = {0:.2f} + {1:.2f} - {2:.2f} ".format(
                lnd, lnd_errp, lnd_errm
            )
        )
        if return_samples:
            return (lnd, lnd_errp, lnd_errm, lnd_samples)
        else:
            return (lnd, lnd_errp, lnd_errm)

    def get_Av(self, burnin=None, return_samples=False):
        """get max aposteriori Av from stardate samples"""
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate
        burnin = burnin if burnin is not None else self.burnin
        Av, Av_errp, Av_errm, Av_samples = star.Av_results(burnin=burnin)
        print("Av = {0:.2f} + {1:.2f} - {2:.2f}".format(Av, Av_errp, Av_errm))
        if return_samples:
            return (Av, Av_errp, Av_errm, Av_samples)
        else:
            return (Av, Av_errp, Av_errm)

    def get_star_inclination(self, Prot=None, Rstar=None, vsini=None):
        """estimate stellar inclination [deg]; queries vizier if
        arguments are None
        Prot : float
            rotational period [days]
        Rstar : float
            stellar radius [Rsun]
        vsini : float
            projected rotational velocity [km/s]
        """
        if (Prot is None) or (Rstar is None) or (vsini is None):
            if Prot is None:
                res = self.query_vizier_param("Prot")
                if len(res) == 0:
                    errmsg = "No Prot found in literature. Provide Prot."
                    print(errmsg)
                else:
                    print(f"Choose input for Prot:\n{res}")
            if vsini is None:
                res = self.query_vizier_param("vsini")
                if len(res) == 0:
                    errmsg = "No vsini found in literature. Provide vsini."
                    print(errmsg)
                else:
                    print(f"Choose input for vsini:\n{res}")
            else:
                if vsini > 200:
                    print("Note: vsini in km/s!")
            if Rstar is None:
                d = {}
                if self.toi_Rstar is not None:
                    d.update({"ticv8": self.toi_Rstar})
                if self.gaia_params is None:
                    g = self.query_gaia_dr2_catalog(return_nearest_xmatch=True)
                else:
                    g = self.gaia_params
                d.update({"gaia DR2": g.radius_val})
                print(f"Choose input for Rstar:\n{d}")
        else:
            v_rad = (
                2
                * np.pi
                * Rstar
                * u.Rsun.to(u.km)
                / (Prot * u.day.to(u.second))
            )
            i = np.arcsin(vsini / v_rad)
            return np.rad2deg(i)

    def plot_flatchain(self, burnin=None):
        """
        useful to estimate burn-in
        """
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate

        chain = star.sampler.chain
        nwalkers, nsteps, ndim = chain.shape
        fig, axs = pl.subplots(ndim, 1, figsize=(15, ndim), sharex=True)
        [
            axs.flat[i].plot(
                c, drawstyle="steps", color="k", alpha=4.0 / nwalkers
            )
            for i, c in enumerate(chain.T)
        ]
        [axs.flat[i].set_ylabel(l) for i, l in enumerate(self.iso_param_names)]
        return fig

    def plot_corner(
        self,
        posterior="physical",
        use_isochrones=True,
        columns=None,
        burnin=None,
        thin=None,
    ):
        """
        use_isochrones : bool
            use isochrones or stardate results
        posterior : str
            'observed', 'physical', 'derived'
        columns : list
            columns to plot if use_isochrones=True and posterior='derived'
        See https://isochrones.readthedocs.io/en/latest/starmodel.html
        """
        try:
            from corner import corner
        except Exception:
            raise ValueError("pip install corner")

        errmsg = "Try run_isochrones() or run_stardate()"
        assert (self.isochrones_model is not None) | (
            self.stardate is not None
        ), errmsg

        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin

        if use_isochrones:
            if self.isochrones_model is None:
                raise ValueError("Try self.run_isochrones()")
            else:
                star = self.isochrones_model

            if posterior == "observed":
                fig = star.corner_observed()
                # columns = star.observed_quantities
                # data = star._samples
                # fig = corner(data, labels=columns,
                #        quantiles=[0.16, 0.5, 0.84],
                #        truth_color='C1',
                #        show_titles=True, title_kwargs={"fontsize": 12})
            elif posterior == "physical":
                # fig = star.corner_physical()
                columns = star.physical_quantities
                data = star._derived_samples[columns]
                fig = corner(
                    data,
                    labels=columns,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_kwargs={"fontsize": 12},
                )
            elif posterior == "derived":
                if columns is None:
                    print("Supply any columns:")
                    print(star.derived_samples.columns)
                    return None
                else:
                    # fig = star.corner_derived(columns)
                    data = star._derived_samples[columns]
                    fig = corner(
                        data,
                        labels=columns,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                    )
            else:
                raise ValueError("Use posterior=(observed,physical,derived)")

        else:
            if self.stardate is None:
                raise ValueError("Try self.run_stardate()")
            else:
                star = self.stardate

            chain = star.sampler.chain
            nwalkers, nsteps, ndim = chain.shape
            samples = chain[:, burnin::thin, :].reshape((-1, ndim))

            from isochrones.mist import MIST_Isochrone

            mist = MIST_Isochrone()
            # samples needed to interpolate parameters in mist isochrones
            eep_samples = samples[:, 0]
            log_age_samples = samples[:, 1]
            feh_samples = samples[:, 2]

            if posterior == "observed":
                # fig = corner(samples, labels=self.iso_param_names)
                columns = self.iso_param_names
                fig = corner(
                    samples,
                    labels=columns,
                    quantiles=[0.16, 0.5, 0.84],
                    truth_color="C1",
                    show_titles=True,
                    title_kwargs={"fontsize": 12},
                )

            elif posterior == "physical":
                columns = [
                    "mass",
                    "radius",
                    "age",
                    "Teff",
                    "logg",
                    "feh",
                ]  # , 'distance', 'AV'
                derived_samples = mist.interp_value(
                    [eep_samples, log_age_samples, feh_samples], columns
                )
                fig = corner(
                    derived_samples,
                    labels=columns,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_kwargs={"fontsize": 12},
                )

            elif posterior == "derived":
                avail_columns = [
                    "eep",
                    "age",
                    "feh",
                    "mass",
                    "initial_mass",
                    "radius",
                    "density",
                    "logTeff",
                    "Teff",
                    "logg",
                    "logL",
                    "Mbol",
                    "dm_deep"
                    # 'delta_nu', 'nu_max', 'phase',
                    # 'J_mag', 'H_mag', 'K_mag', 'G_mag', 'BP_mag', 'RP_mag',
                    # 'W1_mag', 'W2_mag', 'W3_mag', 'TESS_mag', 'Kepler_mag',
                    # 'parallax', 'distance', 'AV'
                ]
                if columns is None:
                    print("Supply any columns:")
                    print(avail_columns)
                    return None
                else:
                    derived_samples = mist.interp_value(
                        [eep_samples, log_age_samples, feh_samples], columns
                    )
                    fig = corner(
                        derived_samples,
                        labels=columns,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                    )
        return fig

    @property
    def toi_Rstar(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Stellar Radius (R_Sun)"]
        )

    @property
    def toi_Rstar_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Stellar Radius (R_Sun) err"]
        )

    @property
    def starhorse_Teff(self):
        return (
            None
            if self.starhorse is None
            else self.starhorse["teff50"].quantity[0].value
        )

    @property
    def starhorse_Mstar(self):
        return (
            None
            if self.starhorse is None
            else self.starhorse["mass50"].quantity[0].value
        )

    @property
    def starhorse_logg(self):
        return (
            None
            if self.starhorse is None
            else self.starhorse["logg50"].quantity[0].value
        )

    @property
    def starhorse_met(self):
        return (
            None
            if self.starhorse is None
            else self.starhorse["met50"].quantity[0].value
        )

    @property
    def starhorse_Mstar_err(self):
        raise ValueError("starhorse table has no error!")

    # @property
    # def TGv8(self):
    #     pass
