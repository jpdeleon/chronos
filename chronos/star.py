#!/usr/bin/env python
"""
Stellar characterization using stardate
"""
from os.path import join
import numpy as np
import matplotlib.pyplot as pl
import astropy.units as u

from chronos.target import Target

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
        self.param_names = [
            "EEP",
            "log10(Age [yr])",
            "[Fe/H]",
            "ln(Distance)",
            "Av",
        ]

    def get_age(
        self, prot=None, method="isochrones", return_samples=False, burnin=None
    ):
        """
        method : str
            (default) isochrones
        """
        burnin = burnin if burnin is not None else self.burnin
        method = method if method is not None else "isochrones"
        if method == "isochrones":
            age, errp, errm, samples = self.get_isochrone_age(
                return_samples=return_samples, burnin=burnin
            )
        elif method == "gyro":
            age, errp, errm, samples = self.get_gyro_age(
                prot=prot, return_samples=return_samples
            )
        if return_samples:
            return age, errp, errm, samples
        else:
            return age, errp, errm

    def get_spot_age():
        """
        """
        NotImplementedError

    def get_gyro_age(self, prot=None, nsamples=1e4, return_samples=False):
        """
        See https://ui.adsabs.harvard.edu/abs/2019AJ....158..173A/abstract

        FIXME: Implement Monte Carlo
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
            print(f"Estimating age using gyrochronology")

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
                f"stellar age = {mid/1e6:.2f} + {errp/1e6:.2f} - {errm/1e6:.2f} Myr"
            )
        if return_samples:
            return (mid, errp, errm, age_samples)
        else:
            return (mid, errp, errm)

    def run_stardate(self, prot=None, mcmc_steps=None, burnin=None, thin=None):
        """
        """
        prot = prot if prot is not None else self.prot
        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin

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
            prot, prot_err = prot[0], prot[1]
            msg = "Estimating age using isochrones+gyrochronology"
        else:
            msg = "Estimating age using isochrones"
            prot, prot_err = None, None
        mcmc_steps = mcmc_steps if mcmc_steps is not None else self.mcmc_steps

        if self.verbose:
            print(msg)
        # Create a dictionary of observables
        iso_params = {
            "teff": (self.gaia_params["teff_val"], 100),
            "logg": (self.tic_params["logg"], self.tic_params["e_logg"]),
            "G": (self.gaia_params["phot_g_mean_mag"], 0.01),
            "bp": (self.gaia_params["phot_bp_mean_mag"], 0.01),
            "rp": (self.gaia_params["phot_rp_mean_mag"], 0.01),
            "J": (self.tic_params["Jmag"], 0.01),
            "H": (self.tic_params["Hmag"], 0.01),
            "K": (self.tic_params["Kmag"], 0.01),
            "parallax": (
                self.gaia_params["parallax"],
                self.gaia_params["parallax_error"],
            ),  # mas
            # 'AV': self.estimate_Av()
        }

        # Set up the star object.
        Av, Av_err = (
            self.estimate_Av(),
            0.01,
        )  # based on stardate documentation
        star = sd.Star(
            iso_params, prot=prot, prot_err=prot_err, Av=Av, Av_err=Av_err
        )
        # Run the MCMC
        star.fit(max_n=mcmc_steps)
        self.star = star
        return star

    def get_isochrone_age(self, burnin=None, return_samples=False):
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
                f"stellar age = {mid/1e6:.2f} + {errp/1e6:.2f} - {errm/1e6:.2f} Myr"
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
