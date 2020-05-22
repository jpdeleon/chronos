"""
planet characterization module
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as pl
from scipy import stats
import astropy.units as u
import pandas as pd
import astropy.constants as c
from astropy.visualization import hist
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive as nea
from tqdm import tqdm

from chronos.star import Star
from chronos.gls import Gls
from chronos.utils import get_RV_K, get_RM_K
from chronos.config import DATA_PATH

__all__ = ["Planet"]


class LC:
    def __init__(self, t, y, dy=None):
        self.time = t
        self.flux = y
        self.error = dy


class Planet(Star):
    def __init__(
        self,
        starname=None,
        letter="b",
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
        self.starname = starname
        self.letter = letter
        self.planet_params = None
        self.harps_bank_rv = None
        self.harps_bank_target_name = None
        self.nea_params = None

    def get_nea_params(self, query_string=None):
        """
        query_string : str
            e.g. V1298 Tab b (case-sensitive)
        """
        if query_string is None:
            query_string = f"{self.target_name.title()} {self.letter}"
        params = nea.query_planet(query_string, all_columns=True).to_pandas()
        # dataframe to series
        if len(params) > 0:
            params = params.T[0]
            params.name = query_string
            self.nea_params = params
            return params

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

    def query_harps_rv(self, save_csv=True):
        """
        For column meanings:
        https://www2.mpia-hd.mpg.de/homes/trifonov/HARPS_RVBank_header.txt

        DRS : Data Reduction Software (pipeline)
        SERVAL : SpEctrum Radial Velocity AnaLyser (new pipeline)

        NZP : nightly zero point

        Activity indicators
        -------------------
        Halpha : H-alpha index
        NaD1 : Na DI index
        NaD2 : Na DII index
        dLW : differential line width to
        measure variations in the spectral line widths;
        CRX: chromatic RV index of the spectra
        to measure wavelength dependence of the RV from
        individual spectral orders as induced by e.g. spots;
        MLCRX : ML Chromatic index (Slope over log wavelength)

        RV contributions
        ----------------
        SNR_DRS : Signal-to-noise ratio in order 55
        BERV : Barycentric Earth radial velocity
        DRIFT : drift measure
        dNZP_mlc : Contribution from intra-night systematics
        SA : Contribution from secular acceleration
        CONTRAST_DRS
        BIS : Bisector span
        f_RV : observation flag

        Note the 2015 instrumental RV jump (fiber upgrade);
        intra-night drifts in DRS RVs <2015;
        https://arxiv.org/pdf/2001.05942.pdf
        """
        data_url = "https://www2.mpia-hd.mpg.de/homes/trifonov"
        table = self.query_harps_bank_table()
        targets = table["Target"].values
        filename = f"{targets[0]}_harps_all-data_v1.csv"
        local_fp = Path(DATA_PATH, filename)
        if local_fp.exists():
            fp = local_fp
            delimiter = ","
        else:
            fp = f"{data_url}/{targets[0]}_RVs/{filename}"
            delimiter = ";"
        try:
            df = pd.read_csv(fp, delimiter=delimiter)
            if not local_fp.exists():
                if save_csv:
                    df.to_csv(local_fp, index=False)
                    print("Saved: ", local_fp)
            else:
                print("Loaded: ", local_fp)
        except Exception as e:
            print(e)
            print(f"Check url: {fp}")
        self.harps_bank_rv = df
        self.validate_harps_rv()
        self.harps_bank_target_name = self.harps_bank_table.Target.unique()[0]
        return df

    def validate_harps_rv(self):
        if self.harps_bank_rv is None:
            raise ValueError("Try self.query_harps_rv()")
        else:
            rv = self.harps_bank_rv.copy()
        assert ((rv.BJD - rv.BJD_DRS) < 0.1).any()
        daytime = rv["f_RV"] == 32
        if sum(daytime) > 0:
            print(
                f"{sum(daytime)} out of {len(daytime)} are not within nautical twilight."
            )
        low_snr = rv["f_RV"] == 64
        if sum(low_snr) > 0:
            print(f"{sum(low_snr)} out of {len(low_snr)} have low SNR.")
        too_hi_snr = rv["f_RV"] == 124
        if sum(too_hi_snr) > 0:
            print(
                f"{sum(too_hi_snr)} out of {len(too_hi_snr)} have too high SNR."
            )
        if self.verbose:
            print("harps bank data validated.")

    def plot_harps_rv_scatter(self, data_type="rv"):
        """ """
        if self.harps_bank_rv is None:
            rv = self.query_harps_rv()
            assert rv is not None
        else:
            rv = self.harps_bank_rv.copy()

        if data_type == "rv":
            ncols, nrows, figsize = 3, 2, (9, 6)
            columns = [
                "RV_mlc_nzp",
                "RV_drs_nzp",
                "RV_mlc",
                "RV_drs",
                "RV_mlc_j",
                "SNR_DRS",
            ]
            title = f" HARPS RV bank: {self.harps_bank_target_name}"
        elif data_type == "activity":
            ncols, nrows, figsize = 3, 3, (9, 9)
            columns = [
                "CRX",
                "dLW",
                "Halpha",
                "NaD1",
                "NaD2",
                "FWHM_DRS",
                "CONTRAST_DRS",
                "BIS",
                "MLCRX",
            ]
            title = f"{self.harps_bank_target_name} Activity indicators"
        else:
            raise ValueError("Use rv or activity")
        fig, ax = pl.subplots(
            nrows,
            ncols,
            figsize=figsize,
            constrained_layout=True,
            # sharex=True
        )
        ax = ax.flatten()

        n = 0
        bjd0 = rv.BJD.astype(int).min()
        for col in columns:
            e_col = "e_" + col
            if e_col in rv.columns:
                ax[n].errorbar(
                    rv.BJD - bjd0,
                    rv[col],
                    yerr=rv[e_col],
                    marker=".",
                    label=col,
                    ls="",
                )
            else:
                ax[n].plot(
                    rv.BJD - bjd0, rv[col], marker=".", label=col, ls=""
                )
            ax[n].set_xlabel(f"BJD-{bjd0}")
            ax[n].set_ylabel(col)
            n += 1
        fig.suptitle(title)
        return fig

    def plot_harps_pairplot(self, columns=None):
        try:
            import seaborn as sb
        except Exception:
            raise ModuleNotFoundError("pip install seaborn")
        if self.harps_bank_rv is None:
            rv = self.query_harps_rv()
            assert rv is not None
        else:
            rv = self.harps_bank_rv.copy()

        if columns is None:
            columns = [
                "RV_mlc_nzp",
                "RV_drs_nzp",
                "CRX",
                "dLW",
                "Halpha",
                "NaD1",
                "NaD2",
                "FWHM_DRS",
                "BIS",
            ]
        # else:
        #     cols = rv.columns
        #     idx = cols.isin(columns)
        #     errmsg = f"{cols[idx]} column not in\n{cols.tolist()}"
        #     assert np.all(idx), errmsg
        g = sb.PairGrid(rv[columns], diag_sharey=False)
        g.map_upper(sb.scatterplot)
        g.map_lower(sb.kdeplot, colors="C0")
        g.map_diag(sb.kdeplot, lw=2)
        return g

    def plot_harps_rv_gls(
        self,
        columns=None,
        Porb=None,
        Prot=None,
        plims=(0.5, 27),
        use_period=True,
    ):
        """
        plims : tuple
            period limits (min,max)

        See Fig. 16 in https://arxiv.org/pdf/2001.05942.pdf
        """
        if self.harps_bank_rv is None:
            rv = self.query_harps_rv()
            assert rv is not None
        else:
            rv = self.harps_bank_rv.copy()

        if columns is None:
            columns = [
                "RV_mlc_nzp",
                "RV_drs_nzp",
                "CRX",
                "dLW",
                "Halpha",
                "NaD1",
                "NaD2",
                "FWHM_DRS",
                "CONTRAST_DRS",
                "BIS",
            ]
        fig, axs = pl.subplots(len(columns), 1, figsize=(10, 10), sharex=True)
        ax = axs.flatten()
        if self.verbose:
            print(
                f"Computing generalized Lomb-Scargle periodograms:\n{columns}"
            )
        for n, col in enumerate(tqdm(columns)):
            err = rv["e_" + col] if "e_" + col in rv.columns else None
            lc = LC(rv.BJD, rv[col], err)
            gls = Gls(lc, Pbeg=plims[0], Pend=plims[1], ofac=2)

            fbest = gls.best["f"]
            # T0 = gls.best["T0"]

            ax[n].plot(
                1 / gls.f if use_period else gls.f,
                gls.power,
                "b-",
                linewidth=0.5,
                c=f"C{n+1}",
                label=col,
            )
            # mark the highest peak
            ax[n].plot(
                1 / fbest if use_period else fbest,
                gls.power[gls.p.argmax()],
                "r.",
                label=f"$P = {1/fbest:.2f}$",
            )

            Porb = self.toi_period if Porb is None else Porb
            if Porb is not None:
                ax[n].axvline(
                    Porb if use_period else 1 / Porb, 0, 1, c="k", ls="-", lw=2
                )
            if Prot is not None:
                ax[n].axvline(
                    Prot if use_period else 1 / 1 / Prot,
                    0,
                    1,
                    c="k",
                    ls="--",
                    lw=2,
                )
            if plims is not None:
                assert isinstance(plims, tuple)
            ax[n].set_xlim(*plims)
            ax[n].legend(loc=0)
        if Porb is not None:
            ax[0].annotate("Porb", xy=(Porb, ax[0].get_ylim()[1]))
        if Prot is not None:
            ax[0].annotate("Prot", xy=(Prot, ax[0].get_ylim()[1]))
        ax[len(columns) // 2].set_ylabel("Lomb-Scargle Power")
        ax[n].set_xlabel("Period (days)")
        fig.subplots_adjust(hspace=0)
        return fig

    def plot_harps_rv_corr_matrix(self, columns=None):
        try:
            import seaborn as sb
        except Exception:
            raise ModuleNotFoundError("pip install seaborn")

        if self.harps_bank_rv is None:
            rv = self.query_harps_rv()
            assert rv is not None
        else:
            rv = self.harps_bank_rv.copy()

        if columns is None:
            columns = [
                "RV_mlc_nzp",
                "RV_drs_nzp",
                "CRX",
                "dLW",
                "Halpha",
                "NaD1",
                "NaD2",
                "FWHM_DRS",
                "CONTRAST_DRS",
                "BIS",
            ]
        # compute correlation
        corr = rv[columns].corr()

        # generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = pl.subplots(1, 1, figsize=(10, 10))

        # draw the heatmap with the mask and correct aspect ratio
        ax = sb.heatmap(
            corr,
            mask=mask,
            vmax=0.3,  # cmap=cmap,
            square=True,
            xticklabels=corr.index,
            yticklabels=corr.columns,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5, "label": "correlation"},
            ax=ax,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
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
