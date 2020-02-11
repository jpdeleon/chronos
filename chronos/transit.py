# -*- coding: utf-8 -*-

r"""
helper functions for transit modeling
"""

import matplotlib.pyplot as pl
import numpy as np
from astropy import units as u
from astropy import constants as c
import batman

LOG_TWO_PI = np.log(2 * np.pi)


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


def t14_from_pmrr(p, ms, rs, rp, b=0, mp=0.0, e=0.0, w=0.0):
    """Compute the transit width (duration) in days.
    Parameters
    ----------
    p : period [day]
    ms : star mass [Msun]
    rs : star radius [Rsun]
    b : impact parameter
    mp : planet mass [Mearth]
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


def tshape_approx(a, b, k):
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


def rho_from_gr(logg, r):
    r = (r * u.R_sun).cgs
    g = 10 ** logg * u.cm / u.s ** 2
    rho = 3 * g / (r * c.G.cgs * 4 * np.pi)
    return rho


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
# def rho_from_mr(m, r, unit="sun", cgs=True):
#     gcc = u.g / u.cm ** 3
#     kgmc = u.kg / u.m ** 3
#     if unit == "sun":
#         r = r * u.Rsun.to(u.m)
#         m = m * u.Msun.to(u.kg)
#     elif unit == "earth":
#         r = r * u.Rearth.to(u.m)
#         m = m * u.Mearth.to(u.kg)
#     elif unit == "jup":
#         r = r * u.Rjup.to(u.m)
#         m = m * u.Mjup.to(u.kg)
#     else:
#         raise ValueError("unit=[sun,earth,jup]")
#     volume = (4.0 / 3.0) * np.pi * r ** 3
#     rho = m / volume
#     if cgs:
#         return rho * kgmc.to(gcc)
#     else:
#         return rho
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
