"""
planet characterization module
"""
import numpy as np
import matplotlib.pyplot as pl
from scipy import stats
import astropy.units as u
import pandas as pd
import astropy.constants as c
from astropy.visualization import hist

from chronos.star import Star
from chronos.utils import get_RV_K, get_RM_K

__all__ = ["Planet"]


class Planet(Star):
    def __init__(
        self,
        starname=None,
        toiid=None,
        ticid=None,
        epicid=None,
        gaiaDR2id=None,
        ra_deg=None,
        dec_deg=None,
        search_radius=3,
        mission="tess",
        # prot=None,
        verbose=True,
        clobber=True,
        prot=None,
        mcmc_steps=1e4,
        burnin=5e3,
        thin=1,
    ):
        super().__init__(
            name=starname,
            toiid=toiid,
            ticid=ticid,
            epicid=epicid,
            gaiaDR2id=gaiaDR2id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            search_radius=search_radius,
            mission=mission,
            prot=prot,
            mcmc_steps=mcmc_steps,
            burnin=burnin,
            thin=thin,
            verbose=verbose,
            clobber=clobber,
        )
        self.starname = None
        self.planet_params = None
        self.harps_bank_rv = None

    def get_Rp_from_depth(
        self,
        depth=None,
        Rstar=None,
        nsamples=10000,
        percs=[50, 16, 84],
        plot=False,
        apply_contratio=False,
        return_samples=False,
    ):
        """
        estimate Rp from depth via Monte Carlo to account for Rs err
        Rs is taken from TICv8
        """
        if depth is None:
            depth = (self.toi_depth, self.toi_depth_err)
        else:
            assert isinstance(depth, tuple)
        if Rstar is None:
            # TICv8 since starhorse has no Rstar
            if self.tic_params is None:
                _ = self.query_tic_catalog(return_nearest_xmatch=True)
            # FIXME: self.tic_params.rad is different from self.toi_Rstar
            # e.g. see toi 179
            Rstar = (self.toi_Rstar, self.toi_Rstar_err)
        else:
            assert isinstance(Rstar, tuple)
        # FIXME: is TOI depth raw or undiluted?
        depth = stats.truncnorm(
            a=(0 - self.toi_depth) / self.toi_depth_err,
            b=(1 - self.toi_depth) / self.toi_depth_err,
            loc=self.toi_depth,
            scale=self.toi_depth_err,
        ).rvs(size=nsamples)

        if apply_contratio:
            # undiluted depth
            depth = depth * (1 + self.tic_params["contratio"])
        Rstar = stats.norm(loc=Rstar[0], scale=Rstar[1]).rvs(nsamples)

        Rp_samples = np.sqrt(depth) * Rstar * u.Rsun.to(u.Rearth)
        Rp, Rp_lo, Rp_hi = np.percentile(Rp_samples, percs)
        Rp, Rp_siglo, Rp_sighi = Rp, Rp - Rp_lo, Rp_hi - Rp
        if plot:
            _ = hist(Rp_samples, bins="scott")
        if return_samples:
            return (Rp, Rp_siglo, Rp_sighi, Rp_samples)
        else:
            return (Rp, Rp_siglo, Rp_sighi)

    def validate_Rp(self):
        """compare Rp from TOI depth and Rp in TOI;
        if their difference is large:
        * Rp_from_toi_depth>>Rp_from_toi, then
        toi depth is undiluted/overcorrected or Rstar is too small
        * Rp_from_toi_depth<<Rp_from_toi, then
        toi depth is diluted/undercorrected or Rstar is too large
        NOTE: need to confirm if Rp in TOI is un/diluted
        """
        Rp_from_toi_depth = (
            np.sqrt(self.toi_depth) * self.toi_Rstar * u.Rsun.to(u.Rearth)
        )
        Rp_from_toi = self.toi_Rp
        relative_error = (Rp_from_toi_depth / Rp_from_toi - 1) * 100
        errmsg = "relative error between TOI Rp & sqrt(depth)*Rs>50%"
        assert abs(relative_error) < 50, errmsg
        if self.verbose:
            print(f"Relative error: {relative_error:.1f}%")

    def get_Mp_from_MR_relation(
        self, Rp=None, dataset="kepler", use_toi_params=True, **kwargs
    ):
        """
        https://shbhuk.github.io/mrexo/

        FIXME: how to get posterior samples from MR relation?
        """
        try:
            from mrexo import predict_from_measurement, generate_lookup_table
        except Exception:
            raise ModuleNotFoundError("pip install mrexo")

        if Rp is None:
            if use_toi_params:
                Rp, Rp_err = self.toi_Rp, self.toi_Rp_err
                Mp, qtls, iron_planet = predict_from_measurement(
                    measurement=Rp,
                    measurement_sigma=Rp_err,
                    predict="mass",
                    dataset=dataset,
                    **kwargs,
                )
            else:
                # estimate from depth to account for Rs err
                Rp, Rp_siglo, Rp_sighi, Rp_samples = self.get_Rp_from_depth(
                    return_samples=True
                )
                Mp, qtls, iron_planet = predict_from_measurement(
                    measurement=np.median(Rp_samples),
                    measurement_sigma=np.std(Rp_samples),
                    predict="mass",
                    dataset=dataset,
                    **kwargs,
                )

        else:
            assert isinstance(Rp, tuple)
            Mp, qtls, iron_planet = predict_from_measurement(
                measurement=Rp[0],
                measurement_sigma=Rp[1],
                predict="mass",
                dataset=dataset,
                **kwargs,
            )

        # return 1-d array
        Mp_siglo = Mp - qtls[0]
        Mp_sighi = qtls[1] - Mp
        return (Mp, Mp_siglo, Mp_sighi)

    def get_RV_amplitude(
        self,
        P_days=None,
        mp_Mearth=None,
        Ms_Msun=None,
        use_Rp_from_depth=True,
        ecc=0.0,
        inc_deg=90,
        return_samples=False,
        plot=False,
    ):
        """FIXME: implement Monte Carlo
        """
        all_unavailable = (
            (P_days is None) & (Ms_Msun is None) & (mp_Mearth is None)
        )
        if all_unavailable:
            # estimate Mp from Rp via radius-relation
            Rp, Rp_err = self.toi_Rp, self.toi_Rp_err
            mp_Mearth, Mp_siglo, Mp_sighi = self.get_Mp_from_MR_relation(
                (Rp, Rp_err), use_toi_params=use_Rp_from_depth
            )
            # FIXME: is there a way to get Mp posterior?
            mp_Mearth_err = np.sqrt(
                (Mp_sighi - mp_Mearth) ** 2 + (mp_Mearth - Mp_siglo) ** 2
            )
            # get toi period
            P_days, P_days_err = self.toi_period, self.toi_period_err
            # get star mass from starhorse catalog
            Ms_Msun = self.starhorse_Mstar
            Ms_Msun_err = 0.1
            if self.verbose:
                print(f"P from TOI: {P_days:.4f}+/-{P_days_err:.4f} d")
                print(f"Rp from TOI: {Rp:.2f}+/-{Rp_err:.2f} Rearth")
                print(
                    f"Mp estimate using Rp via MR relation: {mp_Mearth:.2f}-{Mp_siglo:.2f}+{Mp_sighi:.2f} Mearth"
                )
                print(
                    f"Mstar from starhorse catalog: {Ms_Msun:.2f}+/-{Ms_Msun_err} Msun"
                )
            assert not (
                (P_days is None) & (Ms_Msun is None) & (mp_Mearth is None)
            )
        results = get_RV_K(
            P_days=(P_days, P_days_err),
            mp_Mearth=(mp_Mearth, mp_Mearth_err),
            Ms_Msun=(Ms_Msun, Ms_Msun_err),
            ecc=ecc,
            inc_deg=inc_deg,
            return_samples=return_samples,
            plot=plot,
        )
        if self.verbose:
            print(
                f"RV K: {results[0]:.2f}-{results[1]:.2f}+{results[2]:.2f} m/s"
            )
        return results

    # def get_RM_amplitude(self, ecc=0.0, inc_deg=90):
    #     return get_RM_K(vsini_kms, rp_Rearth, Rs_Rsun)

    def query_harps_rv(self):
        """ """
        data_url = "https://www2.mpia-hd.mpg.de/homes/trifonov"
        table = self.query_harps_bank_table()
        targets = table["Target"].values
        fp = f"{data_url}/{targets[0]}_RVs/{targets[0]}_harps_all-data_v1.csv"
        try:
            df = pd.read_csv(fp, delimiter=";")
            self.harps_bank_rv = df
            return df
        except Exception as e:
            print(e)
            print(f"Check url: {fp}")

    def plot_harps_rv(self):
        """ """
        if self.harps_bank_rv is None:
            rv = self.query_harps_rv()
            assert rv is not None
        else:
            rv = self.harps_bank_rv.copy()

        fig, ax = pl.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        ax = ax.flatten()

        n = 0
        bjd0 = rv.BJD.astype(int).min()
        for col in [
            "RV_mlc_nzp",
            "RV_drs_nzp",
            "RV_mlc",
            "RV_drs",
            "RV_mlc_j",
            "CRX",
        ]:
            ax[n].errorbar(
                rv.BJD - bjd0,
                rv[col],
                yerr=rv["e_" + col],
                marker=".",
                label=col,
                ls="",
            )
            ax[n].set_xlabel(f"BJD-{bjd0}")
            ax[n].set_ylabel(col)
            n += 1
        fig.suptitle(
            f" HARPS RV bank: {self.harps_bank_table.Target.values[0]}"
        )
        return fig

    @property
    def toi_Rp(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Planet Radius (R_Earth)"]
        )

    @property
    def toi_Rp_err(self):
        return (
            None
            if self.toi_params is None
            else self.toi_params["Planet Radius (R_Earth) err"]
        )


# def logg_model(mp_Mearth, rp_Rearth):
#     """Compute the surface gravity from the planet mass and radius."""
#     mp, rp = Mearth2kg(mp_Mearth), Rearth2m(rp_Rearth)
#     return np.log10(G * mp / (rp * rp) * 1e2)
#
#
# def transmission_spectroscopy_depth(
#     Rs_Rsun, mp_Mearth, rp_Rearth, Teq, mu, Nscaleheights=5
# ):
#     """Compute the expected signal in transit spectroscopy in ppm assuming
#     the signal is seen at 5 scale heights."""
#     g = 10 ** logg_model(mp_Mearth, rp_Rearth) * 1e-2
#     rp = Rearth2m(rp_Rearth)
#     D = (rp / Rsun2m(Rs_Rsun)) ** 2
#     H = kb * Teq / (mu * mproton * g)
#     return Nscaleheights * 2e6 * D * H / rp
#
#
# def is_Lagrangestable(Ps, Ms, mps, eccs):
#     """Compute if a system is Lagrange stable (conclusion of barnes+
#     greenberg 06).
#     mp_i = Mearth"""
#     Ps, mps, eccs = np.array(Ps), np.array(mps), np.array(eccs)
#     smas = AU2m(semimajoraxis(Ps, Ms, mps))
#     stable = np.zeros(mps.size - 1)
#     for i in range(1, mps.size):
#         mu1 = Mearth2kg(mps[i - 1]) / Msun2kg(Ms)
#         mu2 = Mearth2kg(mps[i]) / Msun2kg(Ms)
#         alpha = mu1 + mu2
#         gamma1 = np.sqrt(1 - float(eccs[i - 1]) ** 2)
#         gamma2 = np.sqrt(1 - float(eccs[i]) ** 2)
#         delta = np.sqrt(smas[i] / smas[i - 1])
#         deltas = np.linspace(1.000001, delta, 1e3)
#         LHS = (
#             alpha ** (-3.0)
#             * (mu1 + mu2 / (deltas ** 2))
#             * (mu1 * gamma1 + mu2 * gamma2 * deltas) ** 2
#         )
#         RHS = 1.0 + 3 ** (4.0 / 3) * mu1 * mu2 / (alpha ** (4.0 / 3))
#         fint = interp1d(LHS, deltas, bounds_error=False, fill_value=1e8)
#         deltacrit = fint(RHS)
#         stable[i - 1] = True if delta >= 1.1 * deltacrit else False
#     return stable
#
#
# def sigma_depth(P, rp, Rs, Ms, b, N, Ttot, sig_phot):
#     """Compute the expected uncertainty on the transit depth from a
#      lightcurve with N measurements taken over Ttot days and with
#      measurement uncertainty sig_phot."""
#     delta = (Rearth(rp) / Rsun2m(Rs)) ** 2
#     sma = AU2m(semimajoraxis(P, Ms, 0))
#     tau0 = P / (2 * np.pi) * Rsun2m(Rs) / sma  # days
#     T = 2 * tau0 * np.sqrt(1 - b * b)
#     Gamma = N / Ttot  # days^-1
#     Q = np.sqrt(Gamma * T) * delta / sig_phot
#     sig_delta = delta / Q
#     return sig_delta
