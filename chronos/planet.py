"""
planet characterization module
"""
from uncertainties import unumpy
import astropy.units as u
import astropy.constants as c

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

    def get_planet_mass(self, Mp, Mp_err, dataset="kepler"):
        """
        https://shbhuk.github.io/mrexo/
        """
        try:
            from mrexo import predict_from_measurement, generate_lookup_table
        except Exception:
            raise ModuleNotFoundError("pip install mrexo")

        predicted_mass, qtls, iron_planet = predict_from_measurement(
            measurement=Mp,
            measurement_sigma=Mp_err,
            predict="mass",
            dataset=dataset,
        )
        return predicted_mass, qtls

    # def get_RV_amplitude(self, ecc=0.0, inc_deg=90):
    #     P_days = self.toi_period
    #     Ms_Msun = self.toi
    #     mp_Mearth = self.
    #     return get_RV_K(P_days, Ms_Msun, mp_Mearth, ecc=ecc, inc_deg=inc_deg)
    #
    # def get_RM_amplitude(self, ecc=0.0, inc_deg=90):
    #     return get_RM_K(vsini_kms, rp_Rearth, Rs_Rsun)


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
