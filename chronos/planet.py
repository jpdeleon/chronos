"""
planet characterization module
"""
import subprocess
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy import stats
import astropy.units as u
import pandas as pd
import astropy.constants as c
from astropy.visualization import hist
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive as nea

from chronos.star import Star
from chronos.lightcurve import ShortCadence
from chronos.constants import TESS_TIME_OFFSET
from chronos.utils import get_RV_K, get_RM_K, get_tois
from chronos.config import DATA_PATH

__all__ = ["Planet"]


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
        """
        Attributes
        ----------
        starname : str
            host star name
        letter : str
            planet letter (default=b)
        """
        self.starname = starname
        self.letter = letter
        self.planet_params = None
        self.nea_params = None
        self.lcfit_cmd = None

    def describe_system(self):
        Rp, Rp_err = self.toi_Rp, self.toi_Rp_err
        rearth = r"R$_{\oplus}$"
        Porb = self.toi_period
        Rstar, Rstar_err = self.toi_Rstar, self.toi_Rstar_err
        rsun = r"R$_{\odot}$"
        Mstar = self.starhorse_Mstar
        msun = r"M$_{\odot}$"
        spec_type = self.get_spectral_type()
        desc = f"A {Rp:.1f}+/-{Rp_err:.1f} {rearth} planet orbiting an {spec_type} "
        desc += (
            f"({Rstar:.1f}+/-{Rstar_err:.1f} {rsun}, {Mstar:.1f} {msun}) star "
        )
        desc += f"every {Porb:.2f} d."
        print(r"{}".format(desc))

    def save_ini_vespa(self, outdir=".", fpp_params=None):
        """fpp.ini file for vespa calcfpp script
        See:
        https://github.com/timothydmorton/VESPA/blob/master/README.rst
        """
        # errmsg = "This method is available for TESS mission"
        # assert self.mission=='tess', errmsg
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
            fpp_arr.append("band = TESS")
        else:
            fpp_arr.append(f"cadence = {30*u.minute.to(u.day):.2f}")
            fpp_arr.append("band = Kepler")
        print("Double check entries for cadence and band.")
        fpp_arr.append(f"photfile = {target_name}-lc-folded.txt")
        fpp_arr.append("[constraints]")
        if self.mission.lower() == "tess":
            fpp_arr.append("maxrad = 60.0")  # arcsec
        else:
            fpp_arr.append("maxrad = 12.0")  # arcsec
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

    def save_ini_johannes(
        self,
        outdir=".",
        save_lc=False,
        cadence="short",
        lctype="pdcsap",
        teff=None,
        logg=None,
        feh=None,
        rstar=None,
        period=None,
        epoch=None,
        duration=None,
    ):
        """target.ini file for johannes lcfit script
        Parameters
        ----------
        outdir : str
        save_lc : bool (default=False)
            save raw lcs in each sector needed to run johannes' lcfit script
            and folded lcs in each sector and candidate to run vespa's calc_fpp
        cadence : str
            (default='short')
        lctype : str
            (default='pdcsap')
        teff : tuple
        logg : tuple
        feh : tuple
        rstar : tuple
        period : float
            orbital period [d]
        epoch : float
            [BJD]
        duration : float
            transit duration [d]
        Example file
        ------------
        [planets]
            [[251319382.03]]
                per = 3.708550700
                t0 = 2458102.387129
                t14 = 0.06
                rprs = 0.0140929
            [[251319382.02]]
        [star]
            teff = 5875, 100
            logg = 4.92, 0.08
            feh = 0.0, 0.1
            rad = 0.95, 0.05
        """
        # errmsg = "This method is available for TESS mission"
        # assert self.mission=='tess', errmsg

        if teff is not None:
            assert isinstance(teff, tuple)
        if logg is not None:
            assert isinstance(logg, tuple)
        if feh is not None:
            assert isinstance(feh, tuple)
        if rstar is not None:
            assert isinstance(rstar, tuple)

        Teff = self.toi_Teff if teff is None else teff[0]
        Teff_err = self.toi_Teff_err if teff is None else teff[1]
        log10g = self.toi_logg if logg is None else logg[0]
        log10g_err = self.toi_logg_err if logg is None else logg[1]
        metfeh = self.toi_feh if feh is None else feh[0]
        metfeh_err = self.toi_feh_err if feh is None else feh[1]
        Rstar = self.toi_Rstar if rstar is None else rstar[0]
        Rstar_err = self.toi_Rstar_err if rstar is None else rstar[1]

        target_name = self.target_name.replace(" ", "")

        # get TOI candidate ephemerides
        tois = get_tois(clobber=False)
        df = tois[tois["TIC ID"] == self.ticid]
        nplanets = len(df)
        errmsg = f"{target_name} has {nplanets}"
        if period is not None:
            assert len(period) == nplanets, errmsg
        if epoch is not None:
            assert len(epoch) == nplanets, errmsg
        if duration is not None:
            assert len(duration) == nplanets, errmsg

        outdir = target_name if outdir == "." else outdir
        lcs = []
        if save_lc:
            if cadence == "short":
                print(f"Querying {cadence} cadence PDCSAP light curve")
                sc = ShortCadence(ticid=self.ticid)
                for sector in tqdm(self.all_sectors):
                    try:
                        lc = sc.get_lc(lctype=lctype, sector=sector)
                        # BTJD
                        lc.time = lc.time + TESS_TIME_OFFSET
                        lcname = f"{target_name}-{lctype}-s{sector}-raw.txt"
                        lcpath = Path(outdir, lcname)
                        # for johannes
                        lc.to_csv(
                            lcpath,
                            columns=["time", "flux"],
                            header=False,
                            sep=" ",
                            index=False,
                        )
                        lcs.append(lc)
                        print("Saved: ", lcpath)
                    except Exception as e:
                        print(
                            f"Error: Cannot save {cadence} cadence light curve\n{e}"
                        )
            else:
                errmsg = "Only cadence='short' is available for now."
                raise ValueError(errmsg)
        else:
            lcname = f"{target_name}-{lctype}-s{self.all_sectors[0]}-raw.txt"

        out = f"#command: lcfit -b {self.mission.upper()[0]} -i {lcname} -c {target_name}.ini "
        out += "-o johannes --mcmc-burn 500 --mcmc-thin 10 --mcmc-steps 1000 -k 501"
        self.lcfit_cmd = out
        out += "\n[planets]\n"
        # flat_lcs = []
        for i, (k, row) in enumerate(df.iterrows()):
            d = row
            per = d["Period (days)"] if period is None else period[i]
            t0 = d["Epoch (BJD)"] if epoch is None else epoch[i]
            dur = (
                d["Duration (hours)"] / 24 if duration is None else duration[i]
            )
            if save_lc:
                for lc in lcs:
                    # save folded lc for vespa
                    # TODO: merge all sectors in one folded lc
                    flat = sc.get_flat_lc(
                        lc,
                        period=per,
                        epoch=t0,
                        duration=dur * 24,
                        window_length=5 * dur,
                        edge_cutoff=0.1,
                    )
                    flat = flat.remove_nans()
                    # flat_lcs.append(flat)
                    lcname2 = (
                        f"{target_name}-0{i+1}-{lctype}-s{lc.sector}-flat.txt"
                    )
                    lcpath2 = Path(outdir, lcname2)
                    flat.to_csv(
                        lcpath2,
                        columns=["time", "flux"],
                        header=False,
                        sep=" ",
                        index=False,
                    )
                    print("Saved: ", lcpath2)

                    fold = flat.fold(period=per, t0=t0).remove_nans()
                    lcname3 = (
                        f"{target_name}-0{i+1}-{lctype}-s{lc.sector}-fold.txt"
                    )
                    lcpath3 = Path(outdir, lcname3)
                    fold.to_csv(
                        lcpath3,
                        columns=["time", "flux"],
                        header=False,
                        sep=" ",
                        index=False,
                    )
                    print("Saved: ", lcpath3)
            print(f"==={d.TOI}===")
            out += f"\t[[{d.TOI}]]\n"
            out += f"\t\tper = {per:.6f}\n"
            out += f"\t\tt0 = {t0:.6f}\n"
            out += f"\t\tt14 = {dur:.2f}\n"
        out += "[star]\n"
        out += f"\tteff = {Teff:.0f}, {Teff_err:.0f}\n"
        out += f"\tlogg = {log10g:.2f}, {log10g_err:.2f}\n"
        out += f"\tfeh = {metfeh:.2f}, {metfeh_err:.2f}\n"
        out += f"\trad = {Rstar:.2f}, {Rstar_err:.2f}\n"

        outpath = Path(outdir, f"{target_name}.ini")
        if not Path(outdir).exists():
            Path(outdir).mkdir()
        with open(outpath, "w") as file:
            file.write(out)
        print(f"Saved: {outpath}\n{out}")

    def save_ini(
        self,
        outdir=".",
        kwargs_iso={"bands": "G BP RP J H K W1 W2 W3 TESS".split()},
        kwargs_johannes={"save_lc": True},
        kwargs_vespa={"fpp_params": None},
    ):
        """
        save .ini files for isochrones, johannes, & vespa scripts
        """
        try:
            self.save_ini_isochrones(outdir=outdir, **kwargs_iso)
        except Exception as e:
            print(f"Error in `save_ini_isochrones`\n{e}")
        try:
            self.save_ini_johannes(outdir=outdir, **kwargs_johannes)
        except Exception as e:
            print(f"Error in `save_ini_johannes`\n{e}")
        try:
            self.save_ini_vespa(outdir=outdir, **kwargs_vespa)
        except Exception as e:
            print(f"Error in `save_ini_vespa`\n{e}")

    def run_lcfit_script(self):
        """
        fit transits by launching a process that calls the lcfit script
        """
        if self.lcfit_cmd is None:
            errmsg = "First, run: `save_ini_johannes(save_lc=True)`"
            raise ValueError(errmsg)
        else:
            print(self.lcfit_cmd)
            cmd = self.lcfit_cmd.split("command: ")[1]
            # outdir = cmd.split('-o ')[1].split(' ')[0]
            outdir = self.target_name.replace(" ", "")
            subprocess.call(["cd", outdir], shell=True)
            subprocess.call(cmd.split(" "), cwd=outdir)

    def run_vespa_script(self):
        """
        calculate fpp by launching a process that calls the calcfpp script
        """
        raise NotImplementedError()

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

    def validate_t14(self):
        """
        """

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
        """
        Compute the RV semiamplitude in m/s via Monte Carlo
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
                print(f"Porb from TOI: {P_days:.4f}+/-{P_days_err:.4f} d")
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
        else:
            P_days, P_days_err = P_days[0], P_days[1]
            mp_Mearth, mp_Mearth_err = mp_Mearth[0], mp_Mearth[1]
            Ms_Msun, Ms_Msun_err = Ms_Msun[0], Ms_Msun[1]

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

    @property
    def toi_RpRs(self):
        RpRs = (
            self.toi_params["Planet Radius (R_Earth)"]
            * u.Rearth.to(u.Rsun)
            / self.toi_params["Stellar Radius (R_Sun)"]
        )
        return None if self.toi_params is None else RpRs

    @property
    def toi_RpRs_err(self):
        RpRs_err = (
            self.toi_params["Planet Radius (R_Earth) err"]
            * u.Rearth.to(u.Rsun)
            / self.toi_params["Stellar Radius (R_Sun) err"]
        )
        return None if self.toi_params is None else RpRs_err


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
