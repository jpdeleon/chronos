#!/usr/bin/env python
"""
Stellar characterization using stardate
"""
from os.path import join
from pprint import pprint
import numpy as np
import matplotlib.pyplot as pl
import lightkurve as lk
from scipy.ndimage import gaussian_filter
import astropy.units as u

from chronos.target import Target
from chronos.utils import get_mag_err_from_flux, get_err_quadrature, map_float

try:
    import stardate as sd
except Exception:
    raise ValueError("pip install stardate")

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
        search_radius=2 * u.arcsec,
        prot=None,
        mcmc_steps=1e4,
        burnin=1000,
        thin=10,
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
        )
        self.mcmc_steps = mcmc_steps
        self.burnin = burnin
        self.thin = thin
        self.prot = prot
        self.star = None
        self.iso_params = None
        self.param_names = [
            "EEP",
            "log10(Age [yr])",
            "[Fe/H]",
            "ln(Distance)",
            "Av",
        ]
        self.inits = (329.58, 9.5596, -0.0478, 5.560681631015528, 0.0045)
        self.init_params = {
            k: self.inits[i] for i, k in enumerate(self.param_names)
        }

    def get_age(
        self,
        lc=None,
        prot=None,
        amp=None,
        sigma_blur=3,
        nsamples=1e4,
        method="isochrones",
        return_samples=True,
        burnin=None,
    ):
        """
        method : str
            (default) isochrones
        """
        nsamples = int(nsamples)
        burnin = burnin if burnin is not None else self.burnin
        method = method if method is not None else "isochrones"
        if method == "isochrones":
            age, errp, errm, samples = self.get_age_from_color(
                return_samples=True, burnin=burnin
            )
        elif method == "prot":
            age, errp, errm, samples = self.get_age_from_rotation_period(
                prot=prot, nsamples=nsamples, return_samples=True
            )
        elif method == "amp":
            age, errp, errm, samples = self.get_age_from_rotation_amplitude(
                lc=lc,
                prot=prot,
                amp=amp,
                sigma_blur=sigma_blur,
                nsamples=nsamples,
                return_samples=True,
            )
        else:
            msg = "Use method=[isochrones,prot,amp]"
            raise ValueError(msg)
        if return_samples:
            return age, errp, errm, samples
        else:
            return age, errp, errm

    def get_age_from_rotation_amplitude(
        self,
        lc,
        prot,
        amp=None,
        sigma_blur=3,
        nsamples=1e4,
        alpha=(0.56, 0.3),
        slope=(-0.5, 0.17),
        return_samples=True,
    ):
        """
        lc : lk.lightcurve
            lightcurve to be folded
        prot : tuple
            rotation period
        amp : tuple
            rotation amplitude; will be estimated if None
        Notes:
        ------
        Based on Morris+2020:
        A[%]=a*t[Byr]**m, where
        a = (0.56,+1,-0.3)
        m = (-0.5+/-0.17)
        """
        nsamples = int(nsamples)
        assert (lc is not None) & isinstance(lc, lk.LightCurve)
        errmsg = "prot should be a tuple (value,error)"
        assert (prot is not None) & isinstance(prot, tuple), errmsg

        if amp is not None:
            errmsg = "amp should be a tuple (value,error)"
            amp = (amp[0] * 100, amp[1] * 100)
            assert isinstance(amp, tuple), errmsg
        else:
            # estimate rotation period amplitude
            amps = []
            prot_s = prot[0] + np.random.randn(nsamples) * prot[1]
            for p in prot_s:
                fold = lc.fold(period=p)
                # smooth the phase-folded lightcurve
                lc_blur = gaussian_filter(fold.flux, sigma=sigma_blur)
                amps.append(max(lc_blur) - min(lc_blur))
            amp = (np.median(amps) * 100, np.std(amps) * 100)

        # estimate age from amp based on Morris+2020
        t = np.linspace(1e-2, 13.5, nsamples)  # Gyr
        alpha_s = alpha[0] + np.random.randn(nsamples) * alpha[1]
        slope_s = slope[0] + np.random.randn(nsamples) * slope[1]

        age_Byr = lambda a, m, A: (A / a) ** (1 / m)
        A_s = amp[0] + np.random.randn(nsamples) * amp[1]
        ages = []
        for A in A_s:
            x = np.random.choice(alpha_s)
            y = np.random.choice(slope_s)
            ages.append(age_Byr(x, y, A))
        ages = np.array(ages)

        age_samples = ages[(ages > 0) & ~np.isnan(ages) & (ages < t[-1])]
        age_samples = age_samples * 1e9  # in yr
        mid, siglo, sighi = np.percentile(age_samples, [50, 16, 84])
        errm = mid - siglo
        errp = sighi - mid
        if self.verbose:
            print(
                f"gyro age = {mid/1e6:.2f} + {errp/1e6:.2f} - {errm/1e6:.2f} Myr using rotation amplitude {amp[0]:.2f}+/-{amp[1]:.2f}%"
            )
        if return_samples:
            return (mid, errp, errm, age_samples)
        else:
            return (mid, errp, errm)

    def get_age_from_rotation_period(
        self, prot=None, nsamples=1e4, return_samples=False
    ):
        """
        See https://ui.adsabs.harvard.edu/abs/2019AJ....158..173A/abstract

        prot : tuple
            stellar rotation period
        """
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

        prot_samples = prot[0] + np.random.randn(int(nsamples)) * prot[1]
        log10_period_samples = np.log10(prot_samples)

        bprp = self.gaia_params["bp_rp"]  # Gaia BP - RP color.
        bprp_err = 0.1
        bprp_samples = bprp + np.random.randn(int(nsamples)) * bprp_err

        log10_age_yrs = np.array(
            [
                sd.lhf.age_model(x, y)
                for x, y in zip(log10_period_samples, bprp_samples)
            ]
        )
        age_samples = 10 ** log10_age_yrs
        mid, siglo, sighi = np.percentile(age_samples, [50, 16, 84])
        errm = mid - siglo
        errp = sighi - mid
        if self.verbose:
            print(
                f"gyro age = {mid/1e6:.2f} + {errp/1e6:.2f} - {errm/1e6:.2f} Myr using rotation period {prot[1]:.2f}+/-{prot[1]:.2f}d"
            )
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
        add_jhk=True,
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
        self, prot=None, inits=None, mcmc_steps=None, burnin=None, thin=None
    ):
        """
        """
        inits = inits if inits is not None else self.inits
        prot = prot if prot is not None else self.prot
        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin

        prot = prot if prot is not None else self.prot
        if prot is not None:
            errmsg = "prot should be a tuple (value,error)"
            assert isinstance(prot, tuple), errmsg
            prot, prot_err = prot[0], prot[1]
            msg = "Estimating age using isochrones+gyrochronology\n"
        else:
            msg = "Estimating age using isochrones\n"
            prot, prot_err = None, None
        mcmc_steps = mcmc_steps if mcmc_steps is not None else self.mcmc_steps

        if self.verbose:
            print(msg)
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
            print("Init parameters:")
            pprint(self.init_params)

        # Run the MCMC
        star.fit(inits=inits, max_n=mcmc_steps)
        self.inits = inits
        self.star = star
        return star

    def get_age_from_color(self, burnin=None, return_samples=False):
        """
        See https://ui.adsabs.harvard.edu/abs/2019AJ....158..173A/abstract

        """
        if self.star is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.star
        burnin = burnin if burnin is not None else self.burnin

        # Print the median age with the 16th and 84th percentile uncertainties.
        if self.verbose:
            _, _, _, samples = star.age_results(
                burnin=burnin
            )  # in log10(age/yr)
            samples = 10 ** samples
            mid, siglo, sighi = np.percentile(samples, [50, 16, 84])
            errp = sighi - mid
            errm = mid - siglo
            print(
                f"iso+gyro age = {mid/1e6:.2f} + {errp/1e6:.2f} - {errm/1e6:.2f} Myr"
            )
        if return_samples:
            return (mid, errp, errm, samples)
        else:
            return (mid, errp, errm)

    def get_mass(self, burnin=None):
        """
        """
        if self.star is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.star
        burnin = burnin if burnin is not None else self.burnin
        mass, mass_errp, mass_errm, mass_samples = star.mass_results(
            burnin=burnin
        )
        print(
            "Mass = {0:.2f} + {1:.2f} - {2:.2f} M_sun".format(
                mass, mass_errp, mass_errm
            )
        )
        NotImplementedError

    def get_feh(self, burnin=None):
        if self.star is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.star
        burnin = burnin if burnin is not None else self.burnin
        feh, feh_errp, feh_errm, feh_samples = star.feh_results(burnin=burnin)
        print(
            "feh = {0:.2f} + {1:.2f} - {2:.2f}".format(feh, feh_errp, feh_errm)
        )
        NotImplementedError

    def get_distance(self, burnin=None):
        if self.star is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.star
        burnin = burnin if burnin is not None else self.burnin
        lndistance, lndistance_errp, lndistance_errm, lndistance_samples = star.distance_results(
            burnin=burnin
        )
        print(
            "ln(distance) = {0:.2f} + {1:.2f} - {2:.2f} ".format(
                lndistance, lndistance_errp, lndistance_errm
            )
        )
        NotImplementedError

    def get_Av(self, burnin=None):
        if self.star is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.star
        burnin = burnin if burnin is not None else self.burnin
        Av, Av_errp, Av_errm, Av_samples = star.Av_results(burnin=burnin)
        print("Av = {0:.2f} + {1:.2f} - {2:.2f}".format(Av, Av_errp, Av_errm))
        NotImplementedError

    def plot_flatchain(self, burnin=None):
        """
        useful to estimate burn-in
        """
        if self.star is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.star

        chain = star.sampler.chain
        nwalkers, nsteps, ndim = chain.shape
        fig, axs = pl.subplots(ndim, 1, figsize=(15, ndim), sharex=True)
        [
            axs.flat[i].plot(
                c, drawstyle="steps", color="k", alpha=4.0 / nwalkers
            )
            for i, c in enumerate(chain.T)
        ]
        [axs.flat[i].set_ylabel(l) for i, l in enumerate(self.param_names)]
        return fig

    def plot_corner(self, burnin=None, thin=None):
        try:
            from corner import corner
        except Exception:
            raise ValueError("pip install corner")
        if self.star is None:
            raise ValueError("Try self.run_stardate()")
        else:
            star = self.star
        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin

        chain = star.sampler.chain
        nwalkers, nsteps, ndim = chain.shape
        samples = chain[:, burnin::thin, :].reshape((-1, ndim))
        fig = corner(samples, labels=self.param_names)
        return fig
