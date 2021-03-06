# -*- coding: utf-8 -*-

r"""
helper functions for transit modeling
"""

import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import newton
from astropy import units as u
from astropy import constants as c
import batman

LOG_TWO_PI = np.log(2 * np.pi)

__all__ = ["get_likelihoods_mass_grid", "get_HEB_depth_from_masses"]


def get_likelihoods_mass_grid(
    m1,
    m2s,
    m3s,
    obs,
    log10age,
    tracks,
    feh,
    bands=["TESS", "J", "H", "K"],
    b=0,
    use_tshape=False,
    obs_min=0,
    obs_max=1,
    occultation=False,
):
    """
    compute model likelihood over a mass grid of secondary and tertiary
    stars in a HEB system. See also `plot_likelihood_grid`.

    Parameters
    ----------
    m1 : float
        central star mass
    m2s : list
        list of secondary star masses
    m3s : list
        list of tertiary star masses
    tracks : str
        MIST isochrones track from isochrone
    obs : tuple
        (value, error) of the parameter of interest e.g. observed transit depth
    log10age : float
        age of the system
    feh : float
        metallicity of the system
    bands : list
        list of band
    """
    errmsg = "obs must be a tuple of (value, error)"
    assert isinstance(obs, tuple), errmsg

    mass_grids = {}
    for bp in bands:
        mass_grid = np.zeros((len(m3s), len(m2s)))
        for i, m2 in enumerate(m2s):
            for j, m3 in enumerate(m3s):
                if occultation:
                    calc = get_HEB_depth_from_masses(
                        m1,
                        m2,
                        m3,
                        tracks,
                        log10age,
                        feh,
                        band=bp,
                        occultation=True,
                    )
                else:
                    calc = get_HEB_depth_from_masses(
                        m1,
                        m2,
                        m3,
                        tracks,
                        log10age,
                        feh,
                        band=bp,
                        occultation=False,
                    )
                if use_tshape:
                    calc = tshape_approx(np.sqrt(calc), b=b)
                    # calc = max_k(calc)
                if (calc >= obs_min) & (calc <= obs_max):
                    ll = likelihood(calc, obs[0], obs[1])
                else:
                    ll = np.nan
                mass_grid[j, i] = ll
        mass_grids[bp] = mass_grid
    return mass_grids


def get_HEB_depth_from_masses(
    mass1,
    mass2,
    mass3,
    tracks,
    log10age,
    feh,
    F0=1,
    band="TESS",
    occultation=False,
):
    """
    compute the passband-dependent eclipse depth given masses of the hierarchical system,
    assuming MIST, b=0, and m3 eclipsing m2
    Parameters
    ----------
    mass1, mass2, mass3 : float
        mass components of an HEB
    tracks : obj
        MIST isochrones track from isochrone
    log10age : float
        age of the system
    feh : float
        metallicity of the system
    F0 : float
        flux contamination factor
    band : str
        band
    occultation : bool
        compute depth during occultation (default=False)
    """
    band = band + "_mag"
    star1 = tracks.generate(mass1, log10age, feh, return_dict=True)
    mag1 = star1[band]

    star2 = tracks.generate(mass2, log10age, feh, return_dict=True)
    mag2 = star2[band]

    star3 = tracks.generate(mass3, log10age, feh, return_dict=True)
    mag3 = star3[band]

    # rstar1 = star1["radius"]
    rstar2 = star2["radius"]
    rstar3 = star3["radius"]

    # mag = -2.5*log10(F/F0)
    f1 = F0 * 10 ** (-0.4 * mag1)
    f2 = F0 * 10 ** (-0.4 * mag2)
    f3 = F0 * 10 ** (-0.4 * mag3)

    # total flux during out of transit/eclipse
    f_out = f1 + f2 + f3

    if occultation:
        # flux during eclipse
        f_in = f1 + f2
    else:
        # flux during transit
        f_in = f1 + f2 - f2 * (rstar3 / rstar2) ** 2 + f3

    return 1 - f_in / f_out


def get_EB_depth_from_masses(
    mass1, mass2, tracks, log10age, feh, F0=1, band="TESS", occultation=False
):
    """
    compute the passband-dependent eclipse depth given masses of the binary system,
    assuming MIST, b=0, and m2 eclipsing m1
    Parameters
    ----------
    mass1, mass2 : float
        mass components of an EB
    tracks : obj
        MIST isochrones track from isochrone
    log10age : float
        age of the system
    feh : float
        metallicity of the system
    F0 : float
        flux contamination factor
    band : str
        band
    occultation : bool
        compute depth during occultation (default=False)
    """
    assert mass1 >= mass2
    band = band + "_mag"
    star1 = tracks.generate(mass1, log10age, feh, return_dict=True)
    mag1 = star1[band]

    star2 = tracks.generate(mass2, log10age, feh, return_dict=True)
    mag2 = star2[band]

    rstar1 = star1["radius"]
    rstar2 = star2["radius"]

    # mag = -2.5*log10(F/F0)
    f1 = F0 * 10 ** (-0.4 * mag1)
    f2 = F0 * 10 ** (-0.4 * mag2)

    # total flux during out of transit/eclipse
    f_out = f1 + f2

    if occultation:
        # flux during eclipse
        f_in = f1
    else:
        # flux during transit
        f_in = f1 - f1 * (rstar2 / rstar1) ** 2 + f2

    return 1 - f_in / f_out


def likelihood(model, data, err):
    return (1 / np.sqrt(2 * np.pi * err ** 2)) * np.exp(
        -((data - model) / err) ** 2
    )


def blackbody_temperature(bmag, vmag):
    """
    calculate blackbody temperature using the Ballesteros formula; Eq. 14 in
    https://arxiv.org/pdf/1201.1809.pdf
    """
    t_bb = 4600 * (
        (1 / (0.92 * (bmag - vmag) + 1.7))
        + (1 / (0.92 * (bmag - vmag) + 0.62))
    )
    return t_bb


def u_to_q(u1, u2):
    """convert limb-darkening coefficients from q to u
    See Kipping 2013, eq. 15 & 16:
    https://arxiv.org/pdf/1311.1170v3.pdf
    """
    q1 = (u1 + u2) ** 2
    q2 = u1 / (2 * (u1 + u2))
    return q1, q2


def q_to_u(q1, q2):
    """convert limb-darkening coefficients from q to u
    See Kipping 2013, eq. 17 & 18:
    https://arxiv.org/pdf/1311.1170v3.pdf
    """
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return u1, u2


def a_from_bkpt14(b, k, p, t14, i=np.pi / 2):
    """scaled semi-major axis [R_sun]
    See Winn 2014 ("Transits and Occultations"), eq. 14

    RpRs = 0.0092
    Check: a_from_bkpt14(b=0, k=RpRs, p=365.25, t14=13/24, i=np.pi/2) = 216.6
    """
    assert i < 3.15, "inc should be in radians"
    numer = np.sqrt((k + 1) ** 2 - b ** 2)
    denom = np.sin(i) * np.sin(t14 * np.pi / p)
    return numer / denom


def i_from_abew(a, b, e=0, w=0):
    """Orbital inclination from the impact parameter, scaled semi-major axis, eccentricity and argument of periastron
    See Winn 2014 ("Transits and Occultations"), eq. 7
    Parameters
    ----------
      b  : impact parameter       [-]
      a  : scaled semi-major axis [R_Star]
      e  : eccentricity           [-]
      w  : argument of periastron [rad]
    Returns
    -------
      i  : inclination            [rad]

    Check: i_from_abew(a=216.6, b=0, e=0, w=0) = np.pi/2 = 1.57
    """
    if (e != 0) | (w != 0):
        return np.arccos(b / a * (1 + e * np.sin(w)) / (1 - e ** 2))
    else:
        return np.arccos(b / a)


def b_from_aiew(a, i, e=0, w=0):
    """impact parameter
    See Seager & Mallen-Ornelas 2003, eq. 13
    """
    return a * np.cos(i)


def t14_ecc(a, b, k, p, e, w, tr_sign=1):
    r"""transit duration for eccentric orbit
    RpRs = 0.0092
    Check: t14_ecc(a=216.6, b=0, k=RpRs, p=365.25, e=0, w=np.pi, tr_sign=1)=0.54=13 hr
    """
    # i = i_from_abew(a, b, e, w)
    ae = np.sqrt(1.0 - e ** 2) / (1.0 + tr_sign * e * np.sin(w))
    return t14_circ(a, b, k, p) * ae


def t14_circ(a, b, k, p):
    """transit duration for circular orbit
    See Winn 2014 ("Transits and Occultations"), eq. 14
    """
    i = i_from_abew(a, b)
    alpha = np.sqrt((1 + k) ** 2 - b ** 2)
    return (p / np.pi) * np.arcsin(alpha / np.sin(i) / a)


def t23_circ(a, b, k, p):
    """in-transit duration
    See Winn 2014 ("Transits and Occultations"), eq. 15
    """
    i = i_from_abew(a, b)
    alpha = np.sqrt((1 - k) ** 2 - b ** 2)
    return (p / np.pi) * np.arcsin(alpha / np.sin(i) / a)


def t14_from_abkp(a, b, k, p, e=0.0, w=0.0, tr_sign=1):
    if (e != 0) | (w != 0):
        return t14_ecc(a, b, k, p, e, w, tr_sign)
    else:
        return t14_circ(a, b, k, p)


def t14max_from_pmrr(p, ms, rs, rp):
    """Compute the maximum transit duration in days:
    Eq. 10 in Hippke & Heller 2019
    Parameters
    ----------
    p : period [day]
    ms : star mass [Msun]
    rs : star radius [Rsun]
    rp : planet radius [Rearth]
    Returns
    -------
    t14 : transit duration [day]
    """
    constant = 4 / (np.pi * c.G)
    Porb = p * u.day
    Ms = ms * u.Msun.to(u.kg) * u.kg
    Rs = rs * u.Rsun.to(u.m) * u.m
    Rp = rp * u.Rearth.to(u.m) * u.m
    t14 = (Rp + Rs) * (constant * Porb / Ms) ** (1 / 3)
    return t14.to(u.day).value


def t14_from_pmrr(p, ms, rs, rp, b=0, mp=0.0, e=0.0, w=0.0):
    """Compute the transit width (duration) in days.
    Parameters
    ----------
    p : period [day]
    ms : star mass [Msun]
    rs : star radius [Rsun]
    rp : planet radius [Rearth]
    b : impact parameter
    e : eccentricity
    w : argument of periastron [deg]
    Returns
    -------
    t14 : transit duration [day]

    Check: t14_from_pmrr(p=365.25, ms=1, rs=1, rp=1, b=0, e=0, w=0.1)=0.54
    """
    sma = sma_from_pmm(p, ms, mp) * u.au.to(u.Rsun)
    rp = rp * u.Rearth.to(u.m)
    rs = rs * u.Rsun.to(u.m)
    ms = ms * u.Msun.to(u.kg)
    w = np.deg2rad(w)
    return (
        p
        / (np.pi * sma)
        * np.sqrt((1 + rp / rs) ** 2 - b * b)
        * (np.sqrt(1 - e ** 2) / (1 + e * np.sin(w)))
    )


def sma_from_pmm(p, ms, mp=0):
    """ Compute the semimajor axis in AU from Kepler's third law.
    Parameters
    ----------
    p : period [d]
    ms : star mass [Msun]
    mp : planet mass [Mearth]
    Returns
    -------
    a : semi-major axis [au]

    Check: sma_from_mp(365, 1, 1)=
    """
    G = c.G.value
    p = p * u.day.to(u.second)
    mp = mp * u.Mearth.to(u.kg)
    ms = ms * u.Msun.to(u.kg)
    a = a = (G * (ms + mp) * p ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3)
    return a * u.m.to(u.au)


def a_from_prho(p, rho, cgs=True):
    """Scaled semi-major axis from the stellar density and planet's orbital period.
    Parameters
    ----------
    period : orbital period  [d]
    rho    : stellar density [g/cm^3]
    Returns
    -------
    as : scaled semi-major axis [R_star]

    Check: as_from_prho(rho=1.44, period=365.)=215
    Note: 1*u.au.to(u.Rsun)=215
    """
    if cgs:
        rho = rho * u.g / u.cm ** 3
        G = c.G.cgs
    else:
        rho = rho * u.kg / u.m ** 3
        G = c.G
    p = (p * u.day.to(u.second)) * u.second
    aRs = ((rho * G * p ** 2) / (3 * np.pi)) ** (1 / 3)
    return aRs.value


def sma_from_prhor(p, rho, rs):
    """Semi-major axis from the stellar density, stellar radius, and planet's orbital period.
    Parameters
    ----------
      rho    : stellar density [g/cm^3]
      p : orbital period  [d]
      rs  : stellar radius  [R_Sun]
    Returns
    -------
      a : semi-major axis [AU]

    Check: a_from_prhors(rho=1.41, p=365., rs=1.)=1
    """
    return a_from_prho(p, rho) * rs * u.Rsun.to(u.au)


def p_from_am(sma, ms):
    """Orbital period from the semi-major axis and stellar mass.
    Parameters
    ----------
      sma  : semi-major axis [AU]
      ms   : stellar mass    [M_Sun]
    Returns
    -------
      p    : Orbital period  [d]

     Check: p_from_am(a=1., ms=1.)=365
    """
    a = sma * u.au.to(u.m)
    ms = ms * u.Msun.to(u.kg)
    G = c.G.value
    p = np.sqrt((4 * np.pi ** 2 * a ** 3) / (G * ms))
    return p * u.second.to(u.day)


def tshape_approx(k, b=0):
    """transit shape approximation
    See Seager & Mallen-Ornelas 2003, eq. 15
    """
    alpha = (1 - k) ** 2 - b ** 2
    beta = (1 + k) ** 2 - b ** 2
    return np.sqrt(alpha / beta)


def max_k(tshape):
    """maximum depth due to contaminant
    Seager & Mallen-Ornelas 2003, eq. 21


    Check: max_k(ts)*u.Rsun.to(u.Rearth)=1
    """
    return (1 - tshape) / (1 + tshape)


def af_transit(e, w):
    """Calculates the -- factor during the transit"""
    return (1.0 - e ** 2) / (1.0 + e * np.sin(w))


def rho_from_ap(a, p):
    """stellar density assuming circular orbit
    See Kipping+2013, eq. 4:
    https://arxiv.org/pdf/1311.1170v3.pdf
    """
    p = p * u.d
    gpcc = u.g / u.cm ** 3
    rho_mks = 3 * np.pi / c.G / p ** 2 * a ** 3
    return rho_mks.to(gpcc).value


def rho_from_prrt(p, rs, rp, t14, b=0, cgs=False):
    """Compute the stellar density in units of the solar density (1.41 g/cm3)
    from the transit parameters.
    Parameters
    ----------
    p : orbital period [day]
    rp : planet radius [Rearth]
    rs : star radius [Rsun]
    tdur : transit duration [day]
    b : impact parameter
    Returns
    -------
    rho K stellar density [gcc]

    rp, Rs, T, P = Rearth2m(rp_Rearth), Rsun2m(Rs_Rsun), days2sec(T_days), \
                   days2sec(P_days)
    D = (rp / Rs)**2
    rho = 4*np.pi**2 / (P*P*G) * (((1+np.sqrt(D))**2 - \
                                   b*b*(1-np.sin(np.pi*T/P)**2)) / \
                                  (np.sin(np.pi*T/P)**2))**(1.5)  # kg/m3
    """
    kgmc = u.kg / u.m ** 3
    gcc = u.g / u.cm ** 3

    G = c.G.value
    rs = rs * u.Rsun.to(u.m)
    rp = rp * u.Rearth.to(u.m)
    t14 = t14 * u.day.to(u.second)
    p = p * u.day.to(u.second)
    rho = (
        4
        * np.pi ** 2
        / (G * p ** 2)
        * (
            ((1 + rp / rs) ** 2 - b * b * (1 - np.sin(np.pi * t14 / p) ** 2))
            / (np.sin(np.pi * t14 / p) ** 2)
        )
        ** (1.5)
    )  # kg/m3
    if cgs:
        return rho * kgmc.to(gcc)
    else:
        return rho


def logg_from_rhor(rho, r):
    r = (r * u.R_sun).cgs
    gpcc = u.g / u.cm ** 3
    rho *= gpcc
    g = 4 * np.pi / 3 * c.G.cgs * rho * r
    return np.log10(g.value)


def logg_from_mr(mp, rp):
    """Compute the surface gravity from the planet mass and radius.
    Parameters
    ----------
    m : planet mass [Mearth]
    r : planet mass [Rearth]
    """
    G = c.G.value
    mp = mp * u.Mearth.to(u.kg)
    mp = rp * u.Rearth.to(u.m)
    return np.log10(G * mp / (rp * rp) * 1e2)


def rho_from_gr(logg, r, cgs=True):
    kgmc = u.kg / u.m ** 3
    r = (r * u.R_sun).cgs
    g = 10 ** logg * u.cm / u.s ** 2
    rho = 3 * g / (r * c.G.cgs * 4 * np.pi)
    if cgs:
        return rho.value
    else:
        return rho.to(kgmc)


# def logg_southworth(P_days, K_ms, aRp, ecc=0.0, inc_deg=90.0):
#     """Compute the surface gravity in m/s^2 from the equation in Southworth
#     et al 2007."""
#     P, inc = days2sec(P_days), unumpy.radians(inc_deg)
#     return (
#         2
#         * np.pi
#         * K_ms
#         * aRp
#         * aRp
#         * unumpy.sqrt(1 - ecc * ecc)
#         / (P * unumpy.sin(inc))
#     )
#
#
# def tcirc(P_days, Ms_Msun, mp_Mearth, rp_Rearth):
#     """Compute the circularization timescale for a rocky planet
#     in years. From Goldreich & Soter 1966."""
#     Q = 1e2  # for a rocky exoplanet
#     P, Ms, mp, rp, sma = (
#         days2yrs(P_days),
#         Msun2kg(Ms_Msun),
#         Mearth2kg(mp_Mearth),
#         Rearth2m(rp_Rearth),
#         semimajoraxis(P_days, Ms_Msun, mp_Mearth),
#     )
#     return 2.0 * P * Q / (63 * np.pi) * mp / Ms * (AU2m(sma) / rp) ** 5
#
#
# def sample_rhostar(a_samples, p):
#     """
#     Given samples of the scaled semi-major axis and the period,
#     compute samples of rhostar
#     """
#     rho = []
#     n = int(1e4) if len(a_samples) > 1e4 else len(a_samples)
#     for a in a_samples[np.random.randint(len(a_samples), size=n)]:
#         rho.append(rho_from_mr(p, a).value)
#     return np.array(rho)
#
#
# def sample_logg(rho_samples, rstar, urstar):
#     """
#     Given samples of the stellar density and the stellar radius
#     (and its uncertainty), compute samples of logg
#     """
#     rs = rstar + urstar * np.random.randn(len(rho_samples))
#     idx = rs > 0
#     return logg(rho_samples[idx], rs[idx])
#
#
# def sample_ephem(orb, tc_samples, n=10000):
#     tc_samples = np.array(tc_samples).T
#     ephem = []
#     for tc_s in tc_samples[np.random.randint(tc_samples.shape[0], size=n)]:
#         ephem.append(stats.simple_ols(orb, tc_s))
#     return np.array(ephem)
#
#
def rho_from_mr(m, r, unit="sun", cgs=True):
    gcc = u.g / u.cm ** 3
    kgmc = u.kg / u.m ** 3
    if unit == "sun":
        r = r * u.Rsun.to(u.m)
        m = m * u.Msun.to(u.kg)
    elif unit == "earth":
        r = r * u.Rearth.to(u.m)
        m = m * u.Mearth.to(u.kg)
    elif unit == "jup":
        r = r * u.Rjup.to(u.m)
        m = m * u.Mjup.to(u.kg)
    else:
        raise ValueError("unit=[sun,earth,jup]")
    volume = (4.0 / 3.0) * np.pi * r ** 3
    rho = m / volume
    if cgs:
        return rho * kgmc.to(gcc)
    else:
        return rho


#
#
# def ll_normal_es(o, m, e):
#     """Normal log likelihood for scalar err: average standard deviation."""
#     return (
#         -o.size * np.log(e)
#         - 0.5 * o.size * LOG_TWO_PI
#         - 0.5 * np.square(o - m).sum() / e ** 2
#     )
#
#
# def ll_normal_ev(o, m, e):
#     """Normal log likelihood for vector err"""
#     return (
#         -np.sum(np.log(e))
#         - 0.5 * o.size * LOG_TWO_PI
#         - 0.5 * np.sum((o - m) ** 2 / e ** 2)
#     )
#
#
# class TransitModel:
#     """Parameterization: k,q1,q2,tc,p,rho,b"""
#
#     def __init__(self, time, e=0, w=0, ld_power="quadratic"):
#         self.time = time
#         self.transit_params = batman.TransitParams()
#         self.transit_params.limb_dark = ld_power
#         self.pv = None
#         self.e = e
#         self.w = w
#
#     def compute_flux(self, param):
#         """Transit model based on batman"""
#         t0, p, k, rho, b, q1, q2 = [
#             param.get(i) for i in "t0 p k rho b q1 q2".split()
#         ]
#         a = a_from_prho(p, rho)
#         inc = np.rad2deg(i_from_abew(a, b, e=self.e, w=self.w))
#
#         self.transit_params.t0 = t0
#         self.transit_params.per = p
#         self.transit_params.rp = k
#         self.transit_params.a = a
#         self.transit_params.inc = inc
#         self.transit_params.u = q_to_u(q1, q2)
#         self.transit_params.ecc = self.e
#         self.transit_params.w = self.w
#         m = batman.TransitModel(self.transit_params, self.time)
#         return m.light_curve(self.transit_params)
#
#
# if __name__ == "__main__":
#     time = np.linspace(-0.5, 0.5, 100)
#     params = {
#         "t0": 0.0,
#         "p": 8.0,
#         "k": 0.1,
#         "rho": 1.41,
#         "b": 0.1,
#         "q1": 0.1,
#         "q2": 0.1,
#     }
#
#     tm = TransitModel(time, e=0, w=0)
#
#     model = tm.compute_flux(params)
#     pl.plot(time, model, "-")
#     pl.xlabel("Time [days]")
#     pl.ylabel("Relative Flux")
#     pl.show()

# https://gist.github.com/danhey/804a224d96823d0b3406a1c4118048c4
def from_geometry(dphi):
    psi = newton(compute_psi, 0.5, args=(dphi,))
    ecc = np.abs(ecc_func(psi))
    w = argper(ecc, psi)
    return ecc, w


def compute_psi(psi, dphi):
    return psi - np.sin(psi) - 2 * np.pi * dphi


def ecc_func(psi):
    return np.sin(0.5 * (psi - np.pi)) * (
        1.0 - 0.5 * (np.cos(0.5 * (psi - np.pi))) ** 2
    ) ** (-0.5)


def argper(ecc, psi):
    if ecc <= 0.0:
        return 0.0
    return np.arccos(
        1.0 / ecc * (1.0 - ecc ** 2) ** 0.5 * np.tan(0.5 * (psi - np.pi))
    )


# def lnlike(theta, t, f):
#     """
#     """
#     k, t0, p, a, b, q1, q2, sig, c0, c1, c2, c3 = theta
#     m = K2_transit_model(theta, t) + baseline(theta, t)
#     resid = f - m
#     inv_sigma2 = 1.0 / (sig ** 2)
#
#     return -0.5 * (np.sum((resid) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
#
#
# def lnprob(theta, t, f):
#     """
#     """
#     k, t0, p, a, b, q1, q2, sig, c1, c2, c3, c4 = theta
#     inc = np.arccos(b / a)
#     if np.any(np.array(theta[:-4]) < 0):
#         return -np.inf
#     if inc > np.pi / 2.0:
#         return -np.inf
#
#     ll = lnlike(theta, t, f)
#     return ll if np.isfinite(ll) else -np.inf
#
#
# def solve_w(obs, y):
#     """
#     solve for constant coefficients;
#     sys_model is evaluate simply by np.dot(X,w)
#     """
#     X = np.c_[np.atleast_2d(obs).T]
#     try:
#         w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
#     except:
#         w = np.linalg.lstsq(X, y)
#     return w, X
#
#
# def systematics_model(w, aux_vec, time):
#     """
#     systematics model consists of linear combination
#     of constant coefficients (computed here)
#     and auxiliary vectors:
#
#     top n observables, vert_offset, time
#
#     The functional form of the model is
#     s = w0+w1X1+w2X2+...+wnXn
#     """
#
#     vert_offset = np.ones_like(time)
#     # construct X with time
#     X = np.c_[np.concatenate((vert_offset[None, :], time[None, :], aux_vec)).T]
#
#     # compute systematics model
#     sys_model = np.dot(X, w)
#
#     # make sure no nan
#     # assert np.any(~np.isnan(sys_model))
#
#     return sys_model


# def RM_K(vsini_kms, rp_Rearth, Rs_Rsun):
#     '''Compute the approximate semi-amplitude for the Rossiter-McLaughlin
#     effect in m/s.'''
#     D = (Rearth2m(rp_Rearth) / Rsun2m(Rs_Rsun))**2
#     return (vsini_kms*D / (1-D)) * 1e3
#
# def logg_model(mp_Mearth, rp_Rearth):
#     '''Compute the surface gravity from the planet mass and radius.'''
#     mp, rp = Mearth2kg(mp_Mearth), Rearth2m(rp_Rearth)
#     return np.log10(G*mp/(rp*rp) * 1e2)
#
#
# def logg_southworth(P_days, K_ms, aRp, ecc=0., inc_deg=90.):
#     '''Compute the surface gravity in m/s^2 from the equation in Southworth
#     et al 2007.'''
#     P, inc = days2sec(P_days), unumpy.radians(inc_deg)
#     return 2*np.pi*K_ms*aRp*aRp * unumpy.sqrt(1-ecc*ecc) / (P*unumpy.sin(inc))
#
#
# def tcirc(P_days, Ms_Msun, mp_Mearth, rp_Rearth):
#     '''Compute the circularization timescale for a rocky planet
#     in years. From Goldreich & Soter 1966.'''
#     Q = 1e2   # for a rocky exoplanet
#     P, Ms, mp, rp, sma = days2yrs(P_days), Msun2kg(Ms_Msun), \
#                          Mearth2kg(mp_Mearth), Rearth2m(rp_Rearth), \
#                          semimajoraxis(P_days, Ms_Msun, mp_Mearth)
#     return 2.*P*Q/(63*np.pi) * mp/Ms * (AU2m(sma) / rp)**5
#
#
# def transmission_spectroscopy_depth(Rs_Rsun, mp_Mearth, rp_Rearth, Teq, mu,
#                                     Nscaleheights=5):
#     '''Compute the expected signal in transit spectroscopy in ppm assuming
#     the signal is seen at 5 scale heights.'''
#     g = 10**logg_model(mp_Mearth, rp_Rearth) * 1e-2
#     rp = Rearth2m(rp_Rearth)
#     D = (rp / Rsun2m(Rs_Rsun))**2
#     H = kb*Teq / (mu*mproton*g)
#     return Nscaleheights * 2e6 * D * H / rp
#
#
# def stellar_density(P_days, T_days, Rs_Rsun, rp_Rearth, b):
#     '''Compute the stellar density in units of the solar density (1.41 g/cm3)
#     from the transit parameters.'''
#     rp, Rs, T, P = Rearth2m(rp_Rearth), Rsun2m(Rs_Rsun), days2sec(T_days), \
#                    days2sec(P_days)
#     D = (rp / Rs)**2
#     rho = 4*np.pi**2 / (P*P*G) * (((1+np.sqrt(D))**2 - \
#                                    b*b*(1-np.sin(np.pi*T/P)**2)) / \
#                                   (np.sin(np.pi*T/P)**2))**(1.5)  # kg/m3
#     rhoSun = 3*Msun2kg(1) / (4*np.pi*Rsun2m(1)**3)
#     return rho  / rhoSun
#
#
# def astrometric_K(P_days, Ms_Msun, mp_Mearth, dist_pc):
#     '''Compute the astrometric semi-amplitude in micro-arcsec.'''
#     P, Ms, mp, dist = days2sec(P_days), Msun2kg(Ms_Msun), \
#                       Mearth2kg(mp_Mearth), pc2m(dist_pc)
#     Krad = (G*P*P / (4*np.pi*np.pi*Ms*Ms))**(1./3) * mp /dist
#     return np.rad2deg(Krad) * 3.6e3 * 1e6
#
#
# def is_Lagrangestable(Ps, Ms, mps, eccs):
#     '''Compute if a system is Lagrange stable (conclusion of barnes+
#     greenberg 06).
#     mp_i = Mearth'''
#     Ps, mps, eccs = np.array(Ps), np.array(mps), np.array(eccs)
#     smas = AU2m(semimajoraxis(Ps, Ms, mps))
#     stable = np.zeros(mps.size-1)
#     for i in range(1, mps.size):
#         mu1 = Mearth2kg(mps[i-1]) / Msun2kg(Ms)
#         mu2 = Mearth2kg(mps[i]) / Msun2kg(Ms)
#         alpha = mu1+mu2
#         gamma1 = np.sqrt(1-float(eccs[i-1])**2)
#         gamma2 = np.sqrt(1-float(eccs[i])**2)
#         delta = np.sqrt(smas[i]/smas[i-1])
#         deltas = np.linspace(1.000001, delta, 1e3)
#         LHS = alpha**(-3.) * (mu1 + mu2/(deltas**2)) * \
#               (mu1*gamma1 + mu2*gamma2*deltas)**2
#         RHS = 1. + 3**(4./3) * mu1*mu2/(alpha**(4./3))
#         fint = interp1d(LHS, deltas, bounds_error=False, fill_value=1e8)
#         deltacrit = fint(RHS)
#         stable[i-1] = True if delta >= 1.1*deltacrit else False
#     return stable

# dphi = ph_secondary - ph_primary
# geom_ecc, geom_per0 = from_geometry(dphi)
