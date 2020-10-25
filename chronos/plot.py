# -*- coding: utf-8 -*-
r"""
Module for plotting cluster properties.

For inspiration, see http://www.astroexplorer.org/
"""
import sys
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import pandas as pd
import lightkurve as lk

# from transitleastsquares import final_T0_fit
from astropy.coordinates import Angle, SkyCoord, Distance
from astropy.visualization import ZScaleInterval
from astropy.time import Time
from astroquery.mast import Catalogs
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from scipy.ndimage import zoom
from astroquery.skyview import SkyView
from astroplan.plots import plot_finder_image
from astropy.timeseries import LombScargle
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import deepdish as dd

# Import from package
from chronos.target import Target
from chronos.lightcurve import plot_fold_lc
from chronos.cluster import ClusterCatalog, Cluster
from chronos.constants import Kepler_pix_scale, TESS_pix_scale
from chronos.utils import (
    get_toi,
    get_tois,
    PadWithZeros,
    get_mamajek_table,
    parse_aperture_mask,
    is_point_inside_mask,
    is_gaiaid_in_cluster,
    get_fluxes_within_mask,
    get_rotation_period,
)

__all__ = [
    "plot_tls",
    "plot_odd_even",
    "plot_hrd_spectral_types",
    "plot_rotation_period",
    "plot_possible_NEBs",
    "plot_interactive",
    "plot_aperture_outline",
    "plot_aperture_outline2",
    "plot_gaia_sources_on_survey",
    "plot_gaia_sources_on_tpf",
    "plot_cluster_kinematics",
    "get_dss_data",
    "plot_archival_images",
    "plot_dss_image",
    "plot_likelihood_grid",
    "plot_out_of_transit",
    "df_to_gui",
]

# http://gsss.stsci.edu/SkySurveys/Surveys.htm
dss_description = {
    "dss1": "POSS1 Red in the north; POSS2/UKSTU Blue in the south",
    "poss2ukstu_red": "POSS2/UKSTU Red",
    "poss2ukstu_ir": "POSS2/UKSTU Infrared",
    "poss2ukstu_blue": "POSS2/UKSTU Blue",
    "poss1_blue": "POSS1 Blue",
    "poss1_red": "POSS1 Red",
    "all": "best among all plates",
    "quickv": "Quick-V Survey",
    "phase2_gsc2": "HST Phase 2 Target Positioning (GSC 2)",
    "phase2_gsc1": "HST Phase 2 Target Positioning (GSC 1)",
}


class MidPointLogNorm(mcolors.LogNorm):
    """
    Log normalization with midpoint offset

    from
    https://stackoverflow.com/questions/48625475/python-shifted-logarithmic-colorbar-white-color-offset-to-center
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        mcolors.LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = (
            [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)],
            [0, 0.5, 1],
        )
        return np.ma.masked_array(np.interp(np.log(value), x, y))


def plot_likelihood_grid(mass_grid, m2s, m3s, cmap="default", aspect_ratio=1):
    """
    Parameters
    ----------
    mass_grid : 3-d array
        mass grid of likelihood values
    """
    fig, ax = pl.subplots(1, 1, figsize=(8, 8))
    xmin, xmax = m2s[0], m2s[-1]
    ymin, ymax = m3s[0], m3s[-1]

    # norm = MidPointLogNorm(
    #    vmin=mass_grid.min(), vmax=mass_grid.max(), midpoint=0
    # )
    # plot matrix
    cbar = ax.imshow(
        mass_grid,
        origin="lower",
        interpolation="none",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        # norm=norm
    )
    pl.colorbar(
        cbar, ax=ax, label="Likelihood", orientation="vertical"  # shrink=0.9,
    )

    # add labels
    ax.set_aspect(aspect_ratio)
    pl.setp(
        ax,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        xlabel="secondary star mass (Msun)",
        ylabel="tertiary star mass (Msun)",
    )
    return fig


def plot_mass_radius_diagram():
    """
    https://github.com/oscaribv/fancy-massradius-plot/blob/master/mass_radius_plot.ipynb
    """
    errmsg = "To be added later"
    raise NotImplementedError(errmsg)


def plot_cluster_map(
    target_coord=None,
    catalog_name="Bouma2019",
    cluster_name=None,
    offset=10,
    ax=None,
):
    """
    """
    tra = target_coord.ra.deg
    tdec = target_coord.dec.deg
    if ax is None:
        fig, ax = pl.subplot(1, 1, figsize=(5, 5))
    if cluster_name is None:
        cc = ClusterCatalog(catalog_name)
        cat = cc.query_catalog()
        coords = SkyCoord(
            ra=cat["ra"],
            dec=cat["dec"],
            distance=cat["distance"],
            unit=("deg", "deg", "pc"),
        )
        ax.scatter(coords.ra.deg, coords.dec.deg, "ro")
    else:
        c = Cluster(catalog_name=catalog_name, cluster_name=cluster_name)
        mem = c.query_cluster_members()
        rsig = mem["ra"].std()
        dsig = mem["dec"].std()
        r = np.sqrt(rsig ** 2 + dsig ** 2)
        circle = pl.Circle((tra, tdec), r, color="r")
        ax.plot(mem["ra"], mem["dec"], "r.", alpha=0.1)
        ax.add_artist(circle)
    ax.plot(tra, tdec, "bx")
    ax.ylim(tdec - offset, tdec + offset)
    ax.xlim(tra - offset, tra + offset)
    return fig


def plot_orientation_on_tpf(tpf, ax=None):
    """
    Plot the orientation arrows on tpf

    Returns
    -------
    tpf read from lightkurve

    """
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(5, 5))
    mean_tpf = np.nanmean(tpf.flux, axis=0)
    zmin, zmax = ZScaleInterval(contrast=0.5)
    ax.matshow(mean_tpf, vmin=zmin, vmax=zmax, origin="lower")
    _ = plot_orientation(tpf, ax=ax)
    return ax


def plot_orientation(tpf, ax):
    """overlay orientation arrows on tpf plot
    """
    nx, ny = tpf.flux.shape[1:]
    x0, y0 = tpf.column + int(0.9 * nx), tpf.row + int(0.2 * ny)
    # East
    tmp = tpf.get_coordinates()
    ra00, dec00 = tmp[0][0][0][0], tmp[1][0][0][0]
    ra10, dec10 = tmp[0][0][0][-1], tmp[1][0][0][-1]
    theta = np.arctan((dec10 - dec00) / (ra10 - ra00))
    if (ra10 - ra00) < 0.0:
        theta += np.pi
    # theta = -22.*np.pi/180.
    x1, y1 = 1.0 * np.cos(theta), 1.0 * np.sin(theta)
    ax.arrow(x0, y0, x1, y1, head_width=0.2, color="white")
    ax.text(x0 + 1.5 * x1, y0 + 1.5 * y1, "E", color="white")
    # North
    theta = theta + 90.0 * np.pi / 180.0
    x1, y1 = 1.0 * np.cos(theta), 1.0 * np.sin(theta)
    ax.arrow(x0, y0, x1, y1, head_width=0.2, color="white")
    ax.text(x0 + 1.5 * x1, y0 + 1.5 * y1, "N", color="white")
    return ax


def plot_gaia_sources_on_tpf(
    tpf,
    target_gaiaid,
    gaia_sources=None,
    sap_mask="pipeline",
    depth=None,
    kmax=1,
    dmag_limit=8,
    fov_rad=None,
    cmap="viridis",
    figsize=None,
    ax=None,
    invert_xaxis=False,
    invert_yaxis=False,
    pix_scale=TESS_pix_scale,
    **mask_kwargs,
):
    """
    plot gaia sources brighter than dmag_limit; only annotated with starids
    are those that are bright enough to cause reproduce the transit depth;
    starids are in increasing separation

    dmag_limit : float
        maximum delta mag to consider; computed based on depth if None

    TODO: correct for proper motion difference between
    survey image and gaia DR2 positions
    """
    assert target_gaiaid is not None
    img = np.nanmedian(tpf.flux, axis=0)
    # make aperture mask
    mask = parse_aperture_mask(tpf, sap_mask=sap_mask, **mask_kwargs)
    ax = plot_aperture_outline(
        img, mask=mask, imgwcs=tpf.wcs, figsize=figsize, cmap=cmap, ax=ax
    )
    if fov_rad is None:
        nx, ny = tpf.shape[1:]
        diag = np.sqrt(nx ** 2 + ny ** 2)
        fov_rad = (0.4 * diag * pix_scale).to(u.arcmin)

    if gaia_sources is None:
        print(
            "Querying Gaia sometimes hangs. Provide `gaia_sources` if you can."
        )
        target_coord = SkyCoord(
            ra=tpf.header["RA_OBJ"], dec=tpf.header["DEC_OBJ"], unit="deg"
        )
        gaia_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="Gaia", version=2
        ).to_pandas()
    assert len(gaia_sources) > 1, "gaia_sources contains single entry"
    # find sources within mask
    # target is assumed to be the first row
    idx = gaia_sources["source_id"].astype(int).isin([target_gaiaid])
    target_gmag = gaia_sources.loc[idx, "phot_g_mean_mag"].values[0]
    if depth is not None:
        # compute delta mag limit given transit depth
        dmag_limit = (
            np.log10(kmax / depth - 1) if dmag_limit is None else dmag_limit
        )

        # get min_gmag inside mask
        ra, dec = gaia_sources[["ra", "dec"]].values.T
        pix_coords = tpf.wcs.all_world2pix(np.c_[ra, dec], 0)
        contour_points = measure.find_contours(mask, level=0.1)[0]
        isinside = [
            is_point_inside_mask(contour_points, pix) for pix in pix_coords
        ]
        min_gmag = gaia_sources.loc[isinside, "phot_g_mean_mag"].min()
        if (target_gmag - min_gmag) != 0:
            print(
                f"target Gmag={target_gmag:.2f} is not the brightest within aperture (Gmag={min_gmag:.2f})"
            )
    else:
        min_gmag = gaia_sources.phot_g_mean_mag.min()  # brightest
        dmag_limit = (
            gaia_sources.phot_g_mean_mag.max()
            if dmag_limit is None
            else dmag_limit
        )

    base_ms = 128.0  # base marker size
    starid = 1
    # if very crowded, plot only top N
    gmags = gaia_sources.phot_g_mean_mag
    dmags = gmags - target_gmag
    rank = np.argsort(dmags.values)
    for index, row in gaia_sources.iterrows():
        # FIXME: why some indexes are missing?
        ra, dec, gmag, id = row[["ra", "dec", "phot_g_mean_mag", "source_id"]]
        dmag = gmag - target_gmag
        pix = tpf.wcs.all_world2pix(np.c_[ra, dec], 0)[0]
        contour_points = measure.find_contours(mask, level=0.1)[0]

        color, alpha = "red", 1.0
        # change marker color and transparency depending on the location and dmag
        if is_point_inside_mask(contour_points, pix):
            if int(id) == int(target_gaiaid):
                # plot x on target
                ax.plot(
                    pix[1],
                    pix[0],
                    marker="x",
                    ms=base_ms / 16,
                    c="k",
                    zorder=3,
                )
            if depth is not None:
                # compute flux ratio with respect to brightest star
                gamma = 1 + 10 ** (0.4 * (min_gmag - gmag))
                if depth > kmax / gamma:
                    # orange if flux is insignificant
                    color = "C1"
        else:
            # outside aperture
            color, alpha = "C1", 0.5

        ax.scatter(
            pix[1],
            pix[0],
            s=base_ms / 2 ** dmag,  # fainter -> smaller
            c=color,
            alpha=alpha,
            zorder=2,
            edgecolor=None,
        )
        # choose which star to annotate
        if len(gmags) < 20:
            # sparse: annotate all
            ax.text(pix[1], pix[0], str(starid), color="white", zorder=100)
        elif len(gmags) > 50:
            # crowded: annotate only 15 smallest dmag ones
            if rank[starid - 1] < 15:
                ax.text(pix[1], pix[0], str(starid), color="white", zorder=100)
            elif (color == "red") & (dmag < dmag_limit):
                # plot if within aperture and significant source of dilution
                ax.text(pix[1], pix[0], str(starid), color="white", zorder=100)
        elif color == "red":
            # neither sparse nor crowded
            # annotate if inside aperture
            ax.text(pix[1], pix[0], str(starid), color="white", zorder=100)
        starid += 1
    # Make legend with 4 sizes representative of delta mags
    dmags = dmags[dmags < dmag_limit]
    _, dmags = pd.cut(dmags, 3, retbins=True)
    for dmag in dmags:
        size = base_ms / 2 ** dmag
        # -1, -1 is outside the fov
        # dmag = 0 if float(dmag)==0 else 0
        ax.scatter(
            -1,
            -1,
            s=size,
            c="red",
            alpha=0.6,
            edgecolor=None,
            zorder=10,
            clip_on=True,
            label=r"$\Delta m= $" + f"{dmag:.1f}",
        )
    ax.legend(fancybox=True, framealpha=0.5)
    # set img limits
    xdeg = (nx * pix_scale).to(u.arcmin)
    ydeg = (ny * pix_scale).to(u.arcmin)
    # orient such that north is up; east is left
    if invert_yaxis:
        # ax.invert_yaxis()  # increasing upward
        raise NotImplementedError()
    if invert_xaxis:
        # ax.invert_xaxis() #decresing rightward
        raise NotImplementedError()
    if hasattr(ax, "coords"):
        ax.coords[0].set_major_formatter("dd:mm")
        ax.coords[1].set_major_formatter("dd:mm")
    pl.setp(
        ax, xlim=(0, nx), ylim=(0, ny), xlabel=f"({xdeg:.2f} x {ydeg:.2f})"
    )
    return ax


def plot_gaia_sources_on_survey(
    tpf,
    target_gaiaid,
    gaia_sources=None,
    fov_rad=None,
    depth=0.0,
    kmax=1.0,
    sap_mask="pipeline",
    survey="DSS2 Red",
    verbose=True,
    ax=None,
    outline_color="C0",  # pink
    figsize=None,
    invert_xaxis=False,
    invert_yaxis=False,
    pix_scale=TESS_pix_scale,
    **mask_kwargs,
):
    """Plot (superpose) Gaia sources on archival image

    Parameters
    ----------
    target_coord : astropy.coordinates
        target coordinate
    gaia_sources : pd.DataFrame
        gaia sources table
    fov_rad : astropy.unit
        FOV radius
    survey : str
        image survey; see from astroquery.skyview import SkyView;
        SkyView.list_surveys()
    verbose : bool
        print texts
    ax : axis
        subplot axis
    outline_color : str
        aperture outline color (default=C6)
    kwargs : dict
        keyword arguments for aper_radius, percentile
    Returns
    -------
    ax : axis
        subplot axis

    TODO: correct for proper motion difference between
    survey image and gaia DR2 positions
    """
    assert target_gaiaid is not None
    ny, nx = tpf.flux.shape[1:]
    if fov_rad is None:
        diag = np.sqrt(nx ** 2 + ny ** 2)
        fov_rad = (0.4 * diag * pix_scale).to(u.arcmin)
    target_coord = SkyCoord(ra=tpf.ra * u.deg, dec=tpf.dec * u.deg)
    if gaia_sources is None:
        print(
            "Querying Gaia sometimes hangs. Provide `gaia_sources` if you can."
        )
        gaia_sources = Catalogs.query_region(
            target_coord, radius=fov_rad, catalog="Gaia", version=2
        ).to_pandas()
    assert len(gaia_sources) > 1, "gaia_sources contains single entry"
    # make aperture mask
    mask = parse_aperture_mask(tpf, sap_mask=sap_mask, **mask_kwargs)
    maskhdr = tpf.hdu[2].header
    # make aperture mask outline
    contour = np.zeros((ny, nx))
    contour[np.where(mask)] = 1
    contour = np.lib.pad(contour, 1, PadWithZeros)
    highres = zoom(contour, 100, order=0, mode="nearest")
    extent = np.array([-1, nx, -1, ny])

    if verbose:
        print(
            f"Querying {survey} ({fov_rad:.2f} x {fov_rad:.2f}) archival image"
        )
    # -----------create figure---------------#
    if ax is None:
        # get img hdu for subplot projection
        try:
            hdu = SkyView.get_images(
                position=target_coord.icrs.to_string(),
                coordinates="icrs",
                survey=survey,
                radius=fov_rad,
                grid=False,
            )[0][0]
        except Exception:
            errmsg = "survey image not available"
            raise FileNotFoundError(errmsg)
        fig = pl.figure(figsize=figsize)
        # define scaling in projection
        ax = fig.add_subplot(111, projection=WCS(hdu.header))
    # plot survey img
    if str(target_coord.distance) == "nan":
        target_coord = SkyCoord(ra=target_coord.ra, dec=target_coord.dec)
    nax, hdu = plot_finder_image(
        target_coord, ax=ax, fov_radius=fov_rad, survey=survey, reticle=False
    )
    imgwcs = WCS(hdu.header)
    mx, my = hdu.data.shape
    # plot mask
    _ = ax.contour(
        highres,
        levels=[0.5],
        extent=extent,
        origin="lower",
        linewidths=[3],
        colors=outline_color,
        transform=ax.get_transform(WCS(maskhdr)),
    )
    idx = gaia_sources["source_id"].astype(int).isin([target_gaiaid])
    target_gmag = gaia_sources.loc[idx, "phot_g_mean_mag"].values[0]

    for index, row in gaia_sources.iterrows():
        marker, s = "o", 100
        r, d, mag, id = row[["ra", "dec", "phot_g_mean_mag", "source_id"]]
        pix = imgwcs.all_world2pix(np.c_[r, d], 1)[0]
        if int(id) != int(target_gaiaid):
            gamma = 1 + 10 ** (0.4 * (mag - target_gmag))
            if depth > kmax / gamma:
                # too deep to have originated from secondary star
                edgecolor = "C1"
                alpha = 1  # 0.5
            else:
                # possible NEBs
                edgecolor = "C3"
                alpha = 1
        else:
            s = 200
            edgecolor = "C2"
            marker = "s"
            alpha = 1
        nax.scatter(
            pix[0],
            pix[1],
            marker=marker,
            s=s,
            edgecolor=edgecolor,
            alpha=alpha,
            facecolor="none",
        )
    # orient such that north is up; left is east
    if invert_yaxis:
        # ax.invert_yaxis()
        raise NotImplementedError()
    if invert_xaxis:
        # ax.invert_xaxis()
        raise NotImplementedError()
    if hasattr(ax, "coords"):
        ax.coords[0].set_major_formatter("dd:mm")
        ax.coords[1].set_major_formatter("dd:mm")
    # set img limits
    pl.setp(
        nax,
        xlim=(0, mx),
        ylim=(0, my),
        title="{0} ({1:.2f}' x {1:.2f}')".format(survey, fov_rad.value),
    )
    return ax


def get_dss_data(
    ra,
    dec,
    survey="poss2ukstu_red",
    plot=False,
    height=1,
    width=1,
    epoch="J2000",
):
    """
    Digitized Sky Survey (DSS)
    http://archive.stsci.edu/cgi-bin/dss_form
    Parameters
    ----------
    survey : str
        (default=poss2ukstu_red) see `dss_description`
    height, width : float
        image cutout height and width [arcmin]
    epoch : str
        default=J2000
    Returns
    -------
    hdu
    """
    survey_list = list(dss_description.keys())
    if survey not in survey_list:
        raise ValueError(f"{survey} not in:\n{survey_list}")
    base_url = "http://archive.stsci.edu/cgi-bin/dss_search?v="
    url = f"{base_url}{survey}&r={ra}&d={dec}&e={epoch}&h={height}&w={width}&f=fits&c=none&s=on&fov=NONE&v3"
    try:
        hdulist = fits.open(url)
        # hdulist.info()

        hdu = hdulist[0]
        # data = hdu.data
        # header = hdu.header
        if plot:
            _ = plot_dss_image(hdu)
        return hdu
    except Exception as e:
        if isinstance(e, OSError):
            print(f"Error: {e}\nsurvey={survey} image is likely unavailable.")
        else:
            raise Exception(f"Error: {e}")


def plot_dss_image(hdu, cmap="gray", contrast=0.5, ax=None):
    """
    Plot output of get_dss_data:
    hdu = get_dss_data(ra, dec)
    """
    data, header = hdu.data, hdu.header
    interval = ZScaleInterval(contrast=contrast)
    zmin, zmax = interval.get_limits(data)

    if ax is None:
        fig = pl.figure(constrained_layout=True)
        ax = fig.add_subplot(projection=WCS(header))
    ax.imshow(data, vmin=zmin, vmax=zmax, cmap=cmap)
    ax.set_xlabel("RA")
    ax.set_ylabel("DEC")
    title = f"{header['SURVEY']} ({header['FILTER']})\n"
    title += f"{header['DATE-OBS'][:10]}"
    ax.set_title(title)
    # set RA from hourangle to degree
    if hasattr(ax, "coords"):
        ax.coords[0].set_major_formatter("dd:mm:ss")
        ax.coords[1].set_major_formatter("dd:mm:ss")
    return ax


def plot_archival_images(
    ra,
    dec,
    survey1="dss1",
    survey2="ps1",  # "poss2ukstu_red",
    filter="i",
    fp1=None,
    fp2=None,
    height=1,
    width=1,
    cmap="gray",
    reticle=True,
    color="red",
    contrast=0.5,
    return_baseline=False,
):
    """
    Plot two archival images
    See e.g.
    https://s3.amazonaws.com/aasie/images/1538-3881/159/3/100/ajab5f15f2_hr.jpg
    Uses reproject to have identical fov:
    https://reproject.readthedocs.io/en/stable/

    Parameters
    ----------
    ra, dec : float
        target coordinates in degrees
    survey1, survey2 : str
        survey from which the images will come from
    fp1, fp2 : path
        filepaths if the images were downloaded locally
    height, width
        fov of view in arcmin (default=1')
    filter : str
        (g,r,i,z,y) filter if survey = PS1
    cmap : str
        colormap (default='gray')
    reticle : bool
        plot circle to mark the original position of target in survey1
    color : str
        default='red'
    contrast : float
        ZScale contrast
    Notes:
    ------
    Account for space motion:
    https://docs.astropy.org/en/stable/coordinates/apply_space_motion.html

    The position offset can be computed as:
    ```
    import numpy as np
    pm = np.hypot(pmra, pmdec) #mas/yr
    offset = pm*baseline_year/1e3
    ```
    """
    if (survey1 == "ps1") or (survey2 == "ps1"):
        try:
            import panstarrs3 as p3

            fov = np.hypot(width, height) * u.arcmin
            ps = p3.Panstarrs(
                ra=ra,
                dec=dec,
                fov=fov.to(u.arcsec),
                format="fits",
                color=False,
            )
            img, hdr = ps.get_fits(filter=filter, verbose=False)
        except Exception:
            raise ModuleNotFoundError(
                "pip install git+https://github.com/jpdeleon/panstarrs3.git"
            )

    # poss1
    if fp1 is not None and fp2 is not None:
        hdu1 = fits.open(fp1)[0]
        hdu2 = fits.open(fp2)[0]
    else:
        if survey1 == "ps1":
            hdu1 = fits.open(ps.get_url()[0])[0]
            hdu1.header["DATE-OBS"] = Time(
                hdu1.header["MJD-OBS"], format="mjd"
            ).strftime("%Y-%m-%d")
            hdu1.header["FILTER"] = hdu1.header["FPA.FILTER"].split(".")[0]
            hdu1.header["SURVEY"] = "Panstarrs1"
        else:
            hdu1 = get_dss_data(
                ra, dec, height=height, width=width, survey=survey1
            )
        if survey2 == "ps1":
            hdu2 = fits.open(ps.get_url()[0])[0]
            hdu2.header["DATE-OBS"] = Time(
                hdu2.header["MJD-OBS"], format="mjd"
            ).strftime("%Y-%m-%d")
            hdu2.header["FILTER"] = hdu2.header["FPA.FILTER"].split(".")[0]
            hdu2.header["SURVEY"] = "Panstarrs1"
        else:
            hdu2 = get_dss_data(
                ra, dec, height=height, width=width, survey=survey2
            )
    try:
        from reproject import reproject_interp
    except Exception:
        cmd = "pip install reproject"
        raise ModuleNotFoundError(cmd)

    array, footprint = reproject_interp(hdu2, hdu1.header)

    fig = pl.figure(figsize=(10, 5), constrained_layout=True)
    interval = ZScaleInterval(contrast=contrast)

    # data1 = hdu1.data
    header1 = hdu1.header
    ax1 = fig.add_subplot("121", projection=WCS(header1))
    _ = plot_dss_image(hdu1, cmap=cmap, contrast=contrast, ax=ax1)
    if reticle:
        c = Circle(
            (ra, dec),
            0.001,
            edgecolor=color,
            facecolor="none",
            lw=2,
            transform=ax1.get_transform("fk5"),
        )
        ax1.add_patch(c)
    filt1 = (
        hdu1.header["FILTER"]
        if hdu1.header["FILTER"] is not None
        else survey1.split("_")[1]
    )
    # zmin, zmax = interval.get_limits(data1)
    # ax1.imshow(array, origin="lower", vmin=zmin, vmax=zmax, cmap="gray")
    title = f"{header1['SURVEY']} ({filt1})\n"
    title += f"{header1['DATE-OBS'][:10]}"
    ax1.set_title(title)
    # set RA from hourangle to degree
    if hasattr(ax1, "coords"):
        ax1.coords[0].set_major_formatter("dd:mm:ss")
        ax1.coords[1].set_major_formatter("dd:mm:ss")

    # recent
    data2, header2 = hdu2.data, hdu2.header
    ax2 = fig.add_subplot("122", projection=WCS(header1))
    # _ = plot_dss_image(hdu2, ax=ax2)
    zmin, zmax = interval.get_limits(data2)
    ax2.imshow(array, origin="lower", vmin=zmin, vmax=zmax, cmap=cmap)
    if reticle:
        c = Circle(
            (ra, dec),
            0.001,
            edgecolor=color,
            facecolor="none",
            lw=2,
            transform=ax2.get_transform("fk5"),
        )
        ax2.add_patch(c)
        # ax2.scatter(ra, dec, 'r+')
    filt2 = (
        hdu2.header["FILTER"]
        if hdu2.header["FILTER"] is not None
        else survey2.split("_")[1]
    )
    ax2.coords["dec"].set_axislabel_position("r")
    ax2.coords["dec"].set_ticklabel_position("r")
    ax2.coords["dec"].set_axislabel("DEC")
    ax2.set_xlabel("RA")
    title = f"{header2['SURVEY']} ({filt2})\n"
    title += f"{header2['DATE-OBS'][:10]}"
    ax2.set_title(title)
    # set RA from hourangle to degree
    if hasattr(ax2, "coords"):
        ax2.coords[0].set_major_formatter("dd:mm:ss")
        ax2.coords[1].set_major_formatter("dd:mm:ss")
    if return_baseline:
        baseline = int(header2["DATE-OBS"][:4]) - int(header1["DATE-OBS"][:4])
        return fig, baseline
    else:
        return fig


def plot_aperture_outline(
    img,
    mask,
    ax=None,
    imgwcs=None,
    cmap="viridis",
    outline_color="C6",
    figsize=None,
):
    """
    see https://github.com/rodluger/everest/blob/56f61a36625c0d9a39cc52e96e38d257ee69dcd5/everest/standalone.py
    """
    interval = ZScaleInterval(contrast=0.5)
    ny, nx = mask.shape
    contour = np.zeros((ny, nx))
    contour[np.where(mask)] = 1
    contour = np.lib.pad(contour, 1, PadWithZeros)
    highres = zoom(contour, 100, order=0, mode="nearest")
    extent = np.array([-1, nx, -1, ny])

    if ax is None:
        fig, ax = pl.subplots(
            subplot_kw={"projection": imgwcs}, figsize=figsize
        )
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
    _ = ax.contour(
        highres,
        levels=[0.5],
        linewidths=[3],
        extent=extent,
        origin="lower",
        colors=outline_color,
    )
    zmin, zmax = interval.get_limits(img)
    ax.matshow(
        img, origin="lower", cmap=cmap, vmin=zmin, vmax=zmax, extent=extent
    )
    # verts = cs.allsegs[0][0]
    return ax


def plot_aperture_outline2(
    img,
    mask,
    ax=None,
    imgwcs=None,
    cmap="viridis",
    outline_color="C6",
    figsize=None,
):
    """
    see https://github.com/afeinstein20/eleanor/blob/master/eleanor/visualize.py#L78
    """
    interval = ZScaleInterval(contrast=0.5)
    f = lambda x, y: mask[int(y), int(x)]
    g = np.vectorize(f)

    if ax is None:
        fig, ax = pl.subplots(
            subplot_kw={"projection": imgwcs}, figsize=figsize
        )
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
    x = np.linspace(0, mask.shape[1], mask.shape[1] * 100)
    y = np.linspace(0, mask.shape[0], mask.shape[0] * 100)
    extent = [0 - 0.5, x[:-1].max() - 0.5, 0 - 0.5, y[:-1].max() - 0.5]
    X, Y = np.meshgrid(x[:-1], y[:-1])
    Z = g(X[:-1], Y[:-1])
    # plot contour
    _ = ax.contour(
        Z[::-1],
        levels=[0.5],
        colors=outline_color,
        linewidths=[3],
        extent=extent,
        origin="lower",
    )
    zmin, zmax = interval.get_limits(img)
    # plot image
    ax.matshow(
        img, origin="lower", cmap=cmap, vmin=zmin, vmax=zmax, extent=extent
    )
    return ax


def plot_possible_NEBs(gaia_sources, depth, gaiaid=None, kmax=1.0, ax=None):
    """
    """
    assert len(gaia_sources) > 1, "gaia_sources contains single entry"
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(5, 5))

    if gaiaid is None:
        # nearest match (first entry row=0) is assumed as the target
        gaiaid = gaia_sources.iloc[0]["source_id"]
    idx = gaia_sources.source_id.isin([gaiaid])
    target_gmag = gaia_sources.loc[idx, "phot_g_mean_mag"].values[0]

    good, bad, dmags = [], [], []
    for index, row in gaia_sources.iterrows():
        id, mag = row[["source_id", "phot_g_mean_mag"]]
        if int(id) != gaiaid:
            dmag = mag - target_gmag
            gamma = 1 + 10 ** (0.4 * dmag)
            ax.plot(dmag, kmax / gamma, "b.")
            dmags.append(dmag)
            if depth > kmax / gamma:
                # observed depth is too deep to have originated from the secondary star
                good.append(id)
            else:
                # uncertain signal source
                bad.append(id)
    ax.axhline(depth, 0, 1, c="k", ls="--")
    dmags = np.linspace(min(dmags), max(dmags), 100)
    gammas = 1 + 10 ** (0.4 * dmags)

    nbad = len(bad)
    ax.plot(dmags, kmax / gammas, "r-", label=f"potential NEBs={nbad}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta$Gmag")
    ax.set_ylabel("Eclipse depth")
    ax.legend()
    return ax


def plot_rotation_period(
    time,
    flux,
    err=None,
    mask=None,
    method="lombscargle",
    min_per=0.5,
    max_per=30,
    npoints=20,
    xlims=None,
    ylims=None,
    figsize=(10, 5),
    title=None,
):
    """
    method : str
        lombscargle or acf (autocorrelation function)
    """
    fig, ax = pl.subplots(1, 2, figsize=figsize, constrained_layout=True)
    if mask is not None:
        time, flux = time[~mask], flux[~mask]
        err = None if err is None else err[~mask]
    if method == "lombscargle":
        ls = LombScargle(time, flux, dy=err)
        frequencies, powers = ls.autopower(
            minimum_frequency=1.0 / max_per, maximum_frequency=1.0 / min_per
        )
        best_freq = frequencies[np.argmax(powers)]
        peak_period = 1.0 / best_freq
        periods = 1.0 / frequencies
    elif method == "acf":
        raise NotImplementedError("Method not yet available")
    else:
        raise ValueError("Use method='lombscargle'")
    # fit a gaussian to lombscargle power
    prot, prot_err = get_rotation_period(
        time,
        flux,
        min_per=min_per,
        max_per=max_per,
        npoints=npoints,
        plot=False,
    )

    # left: periodogram
    n = 0
    ax[n].plot(periods, powers, "k-")
    ax[n].axvline(
        peak_period, 0, 1, ls="--", c="r", label=f"peak={peak_period:.2f}"
    )
    ax[n].axvline(
        prot, 0, 1, ls="-", c="r", label=f"fit={prot:.2f}+/-{prot_err:.2f}"
    )
    ax[n].legend(title="Best period [d]")
    ax[n].set_xscale("log")
    ax[n].set_xlabel("Period [days]")
    ax[n].set_ylabel("Lomb-Scargle Power")

    # right: phase-folded lc and sinusoidal model
    n = 1
    offset = 0.5
    t_fit = np.linspace(0, 1, 100) - offset
    y_fit = ls.model(t_fit * peak_period - peak_period / 2, best_freq)
    ax[n].plot(
        t_fit * peak_period, y_fit, "r-", lw=3, label="sine model", zorder=3
    )
    # fold data
    phase = ((time / peak_period) % 1) - offset

    a = ax[n].scatter(
        phase * peak_period, flux, c=time, cmap=pl.get_cmap("Blues")
    )
    pl.colorbar(a, ax=ax[n], label="Time [BTJD]")
    ax[n].legend()
    if xlims is None:
        ax[n].set_xlim(-peak_period / 2, peak_period / 2)
    else:
        ax[n].set_xlim(*xlims)
    if ylims is not None:
        ax[n].set_ylim(*ylims)
    ax[n].set_ylabel("Normalized Flux")
    ax[n].set_xlabel("Phase [days]")
    fig.suptitle(title)
    return fig


def plot_tls(results, period=None, plabel=None, figsize=None):
    """

    Attributes
    ----------
    results : dict
        results of after running tls.power()
    * kwargs : dict
        plotting kwargs e.g. {'figsize': (8,8), 'constrained_layout': True}

    Returns
    -------
    fig : figure object
    """
    fig, ax = pl.subplots(2, 1, figsize=figsize)

    n = 0
    label = f"TLS={results.period:.3}"
    ax[n].axvline(results.period, alpha=0.4, lw=3, label=label)
    ax[n].set_xlim(np.min(results.periods), np.max(results.periods))

    for i in range(2, 10):
        ax[n].axvline(i * results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax[n].axvline(results.period / i, alpha=0.4, lw=1, linestyle="dashed")
    ax[n].set_ylabel(r"SDE")
    ax[n].set_xlabel("Period (days)")
    ax[n].plot(results.periods, results.power, color="black", lw=0.5)
    ax[n].set_xlim(0, max(results.periods))
    if period is not None:
        ax[n].axvline(period, 0, 1, ls="--", c="r", label=plabel)
    ax[n].legend(title="best period (d)")

    n = 1
    ax[n].plot(
        results.model_folded_phase - 0.5, results.model_folded_model, color="b"
    )
    ax[n].scatter(
        results.folded_phase - 0.5,
        results.folded_y,
        color="k",
        s=10,
        alpha=0.5,
        zorder=2,
    )
    xlim = 3 * results.duration / results.period
    ax[n].set_xlim(-xlim, xlim)
    ax[n].set_xlabel("Phase")
    ax[n].set_ylabel("Relative flux")
    fig.tight_layout()
    return fig


def plot_odd_even(
    flat, period, epoch, duration=None, yline=None, figsize=(8, 4)
):
    """
    """
    fig, axs = pl.subplots(
        1, 2, figsize=figsize, sharey=True, constrained_layout=True
    )
    fold = flat.fold(period=period, t0=epoch)

    ax = axs[0]
    fold[fold.even_mask].scatter(label="even", ax=ax)
    if yline is not None:
        ax.axhline(yline, 0, 1, lw=2, ls="--", c="k")

    ax = axs[1]
    fold[fold.odd_mask].scatter(label="odd", ax=ax)
    if yline is not None:
        ax.axhline(yline, 0, 1, lw=2, ls="--", c="k")
    if duration is not None:
        xlim = 3 * duration / 24 / period
        axs[0].set_xlim(-xlim, xlim)
        axs[1].set_xlim(-xlim, xlim)
    ax.set_ylabel("")
    fig.subplots_adjust(wspace=0)
    return fig


def plot_hrd_spectral_types(
    x=None,
    y=None,
    c=None,
    cmap="viridis",
    invert_xaxis=True,
    invert_yaxis=False,
    **plot_kwargs,
):
    """
    """
    df = get_mamajek_table()
    fig, ax = pl.subplots(1, 1, **plot_kwargs)

    if (x is not None) & (y is not None):
        _ = df.plot.scatter(x=x, y=y, c=c, ax=ax, cmap=cmap)
        # _ = df.plot.scatter(x='V-Ks', y='M_Ks', c='R_Rsun', cmap='viridis')
        # ax.axhline(6.7, 0, 1, ls='--', c='k')
        # ax.annotate(s='fully convective', xy=(7, 8), fontsize=12)
        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()
        ax.set_ylabel(y)
        ax.set_xlabel(x)
    else:
        classes = []
        for idx, g in df.assign(SpT2=df["#SpT"].apply(lambda x: x[0])).groupby(
            by="SpT2"
        ):
            classes.append(idx)
            x = g["logT"].astype(float)
            y = g["logL"].astype(float)
            ax.plot(x, y, label=idx)
        ax.set_ylabel(r"$\log_{10}$ (L/L$_{\odot}$)")
        ax.set_xlabel(r"$\log_{10}$ (T$_{\rm{eff}}$/K)")
        ax.legend()
        ax.invert_xaxis()
    return fig


def plot_cluster_kinematics(
    ticid=None,
    toiid=None,
    cluster_name=None,
    frac_err=0.5,
    rv=None,
    savefig=False,
):
    """
    """
    assert (ticid is not None) | (toiid is not None)

    t = Target(toiid=toiid, ticid=ticid)
    if cluster_name is None:
        cluster, idxs = t.get_cluster_membership(
            catalog_name="CantatGaudin2020",
            return_idxs=True,
            frac_err=frac_err,
            sigma=5,
        )
        cluster_name = cluster.Cluster

    c = Cluster(cluster_name=cluster_name)
    df_target = t.query_gaia_dr2_catalog(return_nearest_xmatch=True)

    if rv is not None:
        df_target.radial_velocity = rv
    else:
        if np.isnan(df_target.radial_velocity):
            rv = np.nanmean(list(t.query_vizier_param("RV").values()))
            if not np.isnan(rv):
                df_target.radial_velocity = rv
    try:
        fig1 = c.plot_xyz_uvw(
            target_gaiaid=t.gaiaid, df_target=df_target, match_id=False
        )
        fig1.suptitle(f"{t.target_name} in {c.cluster_name}")
        if savefig:
            fp1 = f"{t.target_name}_galactocentric.png"
            fig1.savefig(fp1, bbox_inches="tight")
    except Exception as e:
        print("Error: ", e)
    # ==============
    try:
        log10age = c.get_cluster_age()
        fig2 = c.plot_rdp_pmrv(
            target_gaiaid=t.gaiaid, df_target=df_target, match_id=False
        )
        fig2.suptitle(f"{t.target_name} in {c.cluster_name}")
        if savefig:
            fp2 = f"{t.target_name}_kinematics.png"
            fig2.savefig(fp2, bbox_inches="tight")
    except Exception as e:
        print("Error: ", e)
    # ==============
    try:
        # TODO: AG50 doesn't yield G consistent with cmd
        # if str(df_target.a_g_val) == "nan":
        #     vq = t.query_vizier_param("AG50")
        #     if "I/349/starhorse" in vq:
        #         df_target.a_g_val = vq["I/349/starhorse"]
        #         print("Using AG from starhorse.")
        log10age = c.get_cluster_age()
        ax = c.plot_cmd(
            target_gaiaid=t.gaiaid,
            df_target=df_target,
            match_id=False,
            log_age=log10age,
        )
        ax.set_title(f"{t.target_name} in {c.cluster_name}")
        if savefig:
            fp3 = f"{t.target_name}_cmd.png"
            ax.figure.savefig(fp3, bbox_inches="tight")
    except Exception as e:
        print("Error: ", e)

    try:
        ax = c.plot_hrd(
            target_gaiaid=t.gaiaid,
            df_target=df_target,
            match_id=False,
            log_age=log10age,
        )
        ax.set_title(f"{t.target_name} in {c.cluster_name}")
        if savefig:
            fp4 = f"{t.target_name}_hrd.png"
            ax.figure.savefig(fp4, bbox_inches="tight")
    except Exception as e:
        print("Error: ", e)


def plot_depth_dmag(gaia_catalog, gaiaid, depth, kmax=1.0, ax=None):
    """
    gaia_catalog : pandas.DataFrame
        gaia catalog
    gaiaid : int
        target gaia DR2 id
    depth : float
        observed transit depth
    kmax : float
        maximum depth
    """
    good, bad, dmags = [], [], []
    idx = gaia_catalog.source_id.isin([gaiaid])
    target_gmag = gaia_catalog.iloc[idx]["phot_g_mean_mag"]
    for _, row in gaia_catalog.iterrows():
        id, mag = row[["source_id", "phot_g_mean_mag"]]
        if int(id) != gaiaid:
            dmag = mag - target_gmag
            gamma = 1 + 10 ** (0.4 * dmag)
            pl.plot(dmag, kmax / gamma, "b.")
            dmags.append(dmag)
            if depth > kmax / gamma:
                # observed depth is too deep to have originated from the secondary star
                good.append(id)
            else:
                # uncertain signal source
                bad.append(id)
    if ax is None:
        fig, ax = pl.subplots(1, 1)
    ax.axhline(depth, 0, 1, c="k", ls="--", label="TESS depth")
    dmags = np.linspace(min(dmags), max(dmags), 100)
    gammas = 1 + 10 ** (0.4 * dmags)
    ax.plot(dmags, kmax / gammas, "r-")
    ax.set_yscale("log")
    return ax


def plot_out_of_transit(flat, per, t0, depth):
    """
    """
    fig, axs = pl.subplots(3, 1, figsize=(10, 10), gridspec_kw={"hspace": 0.1})
    dy = 5 if depth < 0.01 else 1.5
    ylim = (1 - dy * depth, 1 + 1.1 * depth)

    _ = plot_fold_lc(
        flat, period=per, epoch=t0 + per / 2, duration=None, ax=axs[0]
    )
    axs[0].axhline(1 - depth, 0, 1, c="C1", ls="--")
    pl.setp(axs[0], xlim=(-0.5, 0.5), ylim=ylim)

    _ = plot_fold_lc(
        flat, period=per, epoch=t0 + per / 2, duration=None, ax=axs[1]
    )
    axs[1].axhline(1 - depth, 0, 1, c="C1", ls="--")
    axs[1].legend("")
    pl.setp(axs[1], xlim=(-0.3, 0.3), title="", ylim=ylim)

    _ = plot_fold_lc(
        flat, period=per, epoch=t0 + per / 2, duration=None, ax=axs[2]
    )
    axs[2].axhline(1 - depth, 0, 1, c="C1", ls="--")
    axs[2].legend("")
    pl.setp(axs[2], xlim=(-0.1, 0.1), title="", ylim=ylim)
    return fig


def plot_interactive(
    catalog_name="CantatGaudin2020",
    min_parallax=1.5,
    thin=10,
    width=800,
    height=400,
):
    """show altair plots of TOI and clusters

    Parameters
    ----------
    plx_cut : float
        parallax cut in mas; default=2 mas < 100pc
    thin : integer
        thinning factor to use ony every nth cluster member
    """
    try:
        import altair as alt
    except ModuleNotFoundError:
        print("pip install altair")

    if sys.argv[-1].endswith("json"):
        print("import altair; altair.renderers.enable('notebook')")

    cc = ClusterCatalog(verbose=False)
    df0 = cc.query_catalog(catalog_name=catalog_name, return_members=False)
    df2 = cc.query_catalog(catalog_name=catalog_name, return_members=True)
    # add members count from df2 in df0
    # counts = df2.groupby('Cluster').size()
    # counts.name = 'Nstars'
    # counts = counts.reset_index()
    # df0 = pd.merge(df0, counts, how='outer')
    idx = df0.parallax >= min_parallax
    df0 = df0.loc[idx]
    df0["distance"] = Distance(parallax=df0["parallax"].values * u.mas).pc
    # plot catalog
    chart0 = (
        alt.Chart(df0)
        .mark_point(color="red")
        .encode(
            x=alt.X(
                "ra:Q",
                axis=alt.Axis(title="RA"),
                scale=alt.Scale(domain=[0, 360]),
            ),
            y=alt.Y(
                "dec:Q",
                axis=alt.Axis(title="Dec"),
                scale=alt.Scale(domain=[-90, 90]),
            ),
            tooltip=[
                "Cluster:N",
                "distance:Q",
                "parallax:Q",
                "pmra:Q",
                "pmdec:Q",
                "Nstars:Q",
            ],
        )
    )

    # get TOI list
    toi = get_tois(verbose=False, clobber=False)
    toi["TIC_ID"] = toi["TIC ID"]
    toi["RA"] = Angle(toi["RA"].values, unit="hourangle").deg
    toi["Dec"] = Angle(toi["Dec"].values, unit="deg").deg
    # plot TOI
    chart1 = (
        alt.Chart(toi, title="TOI")
        .transform_calculate(
            # FIXME: url below doesn't work in pop-up chart
            url="https://exofop.ipac.caltech.edu/tess/target.php?id="
            + alt.datum.TIC_ID
        )
        .mark_point(color="black")
        .encode(
            x=alt.X(
                "RA:Q",
                axis=alt.Axis(title="RA"),
                scale=alt.Scale(domain=[0, 360]),
            ),
            y=alt.Y(
                "Dec:Q",
                axis=alt.Axis(title="Dec"),
                scale=alt.Scale(domain=[-90, 90]),
            ),
            tooltip=[
                "TOI:Q",
                "TIC ID:Q",
                "url:N",
                "Stellar Distance (pc):Q",
                "PM RA (mas/yr):Q",
                "PM Dec (mas/yr):Q",
            ],
        )
        .properties(width=width, height=height)
        .interactive()
    )

    # plot cluster members
    idx = df2.parallax >= min_parallax
    df2 = df2.loc[idx]
    # skip other members
    df2 = df2.iloc[::thin, :]
    chart2 = (
        alt.Chart(df2)
        .mark_circle()
        .encode(
            x="ra:Q",
            y="dec:Q",
            color="Cluster:N",
            tooltip=[
                "source_id:O",
                "parallax:Q",
                "pmra:Q",
                "pmdec:Q",
                "phot_g_mean_mag:Q",
            ],
        )
    )

    return chart2 + chart1 + chart0


def df_to_gui(df, xaxis=None, yaxis=None):
    """
    turn df columns into interactive 2D plots
    """
    try:
        import panel as pn
        import hvplot.pandas
    except Exception:
        cmd = "pip install hvplot panel"
        raise ModuleNotFoundError(cmd)

    x = pn.widgets.Select(name="x", value=xaxis, options=df.columns.tolist())
    y = pn.widgets.Select(name="y", value=yaxis, options=df.columns.tolist())
    kind = pn.widgets.Select(
        name="kind", value="scatter", options=["bivariate", "scatter"]
    )

    plot = df.hvplot(x=x, y=y, kind=kind, colorbar=False, width=600)
    return pn.Row(pn.WidgetBox(x, y, kind), plot)
