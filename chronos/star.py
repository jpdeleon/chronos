#!/usr/bin/env python
"""
Stellar characterization module
"""
# Import standard library
from os.path import join

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
)


__all__ = ["Star"]


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
        mcmc_steps=1e4,
        burnin=1000,
        thin=10,
        alpha=(0.56, 1.05),  # Morris+2020
        slope=(-0.50, 0.17),  # Morris+2020
        sigma_blur=3,
        use_skew_slope=False,
        nsamples=1e4,
        verbose=True,
        clobber=True,
    ):
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

    def estimate_Av(self, map="sfd", constant=3.1):
        """
        compute the extinction Av from color index E(B-V)
        estimated from dustmaps via Av=constant*E(B-V)

        Parameters
        ----------
        map : str
            dust map; see https://dustmaps.readthedocs.io/en/latest/maps.html
        """
        try:
            import dustmaps
        except Exception:
            raise ModuleNotFoundError("pip install dustmaps")

        if map == "sfd":
            from dustmaps import sfd

            # sfd.fetch()
            dust_map = sfd.SFDQuery()
        elif map == "planck":
            from dustmaps import planck

            # planck.fetch()
            dust_map = planck.PlanckQuery()
        else:
            raise ValueError(f"Available maps: (sfd,planck)")

        ebv = dust_map(self.target_coord)
        Av = constant * ebv
        return Av

    def get_spectral_type(
        self,
        columns="Teff B-V J-H H-Ks".split(),
        nsamples=int(1e5),
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
        min_mag_err=0.01,
        add_jhk=False,
        inflate_plx_err=True,
        # phot_bands='G bp rp J H K'.split(),
        # star_params='teff logg parallax'.split()
    ):
        """
        get parameters for isochrones
        """
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

        if teff is None:
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
        else:
            assert isinstance(
                teff, tuple
            ), "teff must be a tuple (value,error)"
            teff, teff_err = teff[0], teff[1]

        gmag = gp["phot_g_mean_mag"]
        gmag_err = get_mag_err_from_flux(
            gp["phot_g_mean_flux"], gp["phot_g_mean_flux_error"]
        )
        gmag_err = gmag_err if gmag_err > min_mag_err else min_mag_err

        bpmag = gp["phot_bp_mean_mag"]
        bpmag_err = get_mag_err_from_flux(
            gp["phot_bp_mean_flux"], gp["phot_bp_mean_flux_error"]
        )
        bpmag_err = bpmag_err if bpmag_err > min_mag_err else min_mag_err

        rpmag = gp["phot_rp_mean_mag"]
        rpmag_err = get_mag_err_from_flux(
            gp["phot_rp_mean_flux"], gp["phot_rp_mean_flux_error"]
        )
        rpmag_err = rpmag_err if rpmag_err > min_mag_err else min_mag_err

        plx = gp["parallax"]
        if inflate_plx_err:
            # inflate error based on Luri+2018
            plx_err = get_err_quadrature(gp["parallax_error"], 0.1)
        else:
            plx_err = gp["parallax_error"]

        params = {
            "teff": (teff, teff_err),
            "G": (gmag, gmag_err),
            # "T": (tp['Tmag'], tp['e_Tmag']),
            "bp": (bpmag, bpmag_err),
            "rp": (rpmag, rpmag_err),
            "parallax": (plx, plx_err),
            # 'AV': self.estimate_Av()
        }
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

        if add_jhk:
            params.update(
                {
                    "J": (tp["Jmag"], tp["e_Jmag"]),
                    "H": (tp["Hmag"], tp["e_Hmag"]),
                    "K": (tp["Kmag"], tp["e_Kmag"]),
                }
            )
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

    def run_stardate(
        self,
        prot=None,
        iso_params0=None,
        mcmc_steps=None,
        burnin=None,
        thin=None,
    ):
        """
        """
        try:
            import stardate as sd
        except Exception:
            raise ModuleNotFoundError("pip install stardate")

        iso_params0 = (
            iso_params0 if iso_params0 is not None else self.iso_params0
        )
        prot = prot if prot is not None else self.prot
        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin

        prot = prot if prot is not None else self.prot
        if prot is not None:
            errmsg = "prot should be a tuple (value,error)"
            assert isinstance(prot, tuple), errmsg
            prot, prot_err = prot[0], prot[1]
        else:
            prot, prot_err = None, None
        mcmc_steps = mcmc_steps if mcmc_steps is not None else self.mcmc_steps

        # Create a dictionary of observables
        if self.iso_params is None:
            iso_params = self.get_iso_params()
        else:
            iso_params = self.iso_params

        # estimate extinction
        Av, Av_err = (self.estimate_Av(), 0.01)
        # Set up the star object.
        star = sd.Star(
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
        star.fit(inits=iso_params0, max_n=mcmc_steps)
        self.iso_params0 = iso_params0
        self.stardate = star
        return star

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

    def get_mass(self, burnin=None):
        """
        """
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
        raise NotImplementedError

    def get_feh(self, burnin=None):
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate
        burnin = burnin if burnin is not None else self.burnin
        feh, feh_errp, feh_errm, feh_samples = star.feh_results(burnin=burnin)
        print(
            "feh = {0:.2f} + {1:.2f} - {2:.2f}".format(feh, feh_errp, feh_errm)
        )
        raise NotImplementedError

    def get_distance(self, burnin=None):
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate
        burnin = burnin if burnin is not None else self.burnin
        (
            lndistance,
            lndistance_errp,
            lndistance_errm,
            lndistance_samples,
        ) = star.distance_results(burnin=burnin)
        print(
            "ln(distance) = {0:.2f} + {1:.2f} - {2:.2f} ".format(
                lndistance, lndistance_errp, lndistance_errm
            )
        )
        raise NotImplementedError

    def get_Av(self, burnin=None):
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate
        burnin = burnin if burnin is not None else self.burnin
        Av, Av_errp, Av_errm, Av_samples = star.Av_results(burnin=burnin)
        print("Av = {0:.2f} + {1:.2f} - {2:.2f}".format(Av, Av_errp, Av_errm))
        raise NotImplementedError

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

    def plot_corner(self, burnin=None, thin=None):
        try:
            from corner import corner
        except Exception:
            raise ValueError("pip install corner")
        if self.stardate is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.stardate
        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin

        chain = star.sampler.chain
        nwalkers, nsteps, ndim = chain.shape
        samples = chain[:, burnin::thin, :].reshape((-1, ndim))
        fig = corner(samples, labels=self.iso_param_names)
        return fig
