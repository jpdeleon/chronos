#!/usr/bin/env python
"""
api to run triceratops to calculate fpp
and probabilities of other scenarios
"""
from pathlib import Path
import argparse
import matplotlib.pyplot as pl
import chronos as cr

parser = argparse.ArgumentParser()
parser.add_argument("-tic", type=int, default=None)
parser.add_argument("-toi", type=int, default=None)
parser.add_argument("-lc", "--lctype", default="pdcsap")
parser.add_argument("-c", "--cadence", default="long")
parser.add_argument("-sec", "--sector", type=int, default=None)
parser.add_argument("-per", type=float, default=None)
parser.add_argument("-t0", type=float, default=None)
parser.add_argument(
    "-dur", type=float, help="transit duration [hours]", default=None
)
parser.add_argument("-depth", type=float, default=None)
parser.add_argument(
    "-b",
    "--binsize",
    help="10 if short cadence else 1",
    type=int,
    default=None,
)
parser.add_argument(
    "-a", "--aper_mask", help="aperture mask shape", type=str, default=None
)
parser.add_argument("-o", "--outdir", type=str, default=".")
parser.add_argument("-v", "--verbose", action="store_true", default=False)
parser.add_argument("-s", "--save", action="store_true", default=False)

args = parser.parse_args()
per = args.per
t0 = args.t0
dur = args.dur

if args.cadence == "short":
    sc = cr.ShortCadence(
        ticid=args.tic, toiid=args.toi, sector=args.sector, calc_fpp=True
    )
    bin = 10 if args.binsize is None else args.binsize
    lc = sc.get_lc(args.lctype)
    if args.aper_mask is None:
        sap_mask = "pipeline"
else:
    sc = cr.LongCadence(
        ticid=args.tic, toiid=args.toi, sector=args.sector, calc_fpp=True
    )
    bin = 1 if args.binsize is None else args.binsize
    if args.aper_mask is None:
        sap_mask = "round"

    if args.lctype == "cdips":
        lc = sc.get_cdips_lc()
    elif args.lctype == "pathos":
        lc = sc.get_pathos_lc()
    else:
        lc = sc.make_custom_lc(sap_mask=sap_mask, aper_radius=1)

flat, trend = sc.get_flat_lc(
    lc, return_trend=True, period=per, epoch=t0, duration=dur
)
if (per is None) & (t0 is None) & (dur is None):
    _ = sc.run_tls(flat, plot=False)
    res = sc.tls_results
    per, t0, dur = res.period, res.T0 - cr.TESS_TIME_OFFSET, res.duration * 24
if args.depth is None:
    depth = 1 - res.depth
else:
    depth = args.depth

fig = sc.plot_trend_flat_lcs(
    lc, period=per, epoch=t0, duration=dur, binsize=bin
)
if args.save:
    fp = Path(args.outdir, sc.target_name + "_trend_flat.png")
    fig.savefig(fp, bbox_inches="tight")

ax = sc.plot_fold_lc(flat, period=per, epoch=t0)
if args.save:
    fp = Path(args.outdir, sc.target_name + "_fold.png")
    ax.figure.savefig(fp, bbox_inches="tight")

if args.cadence == "short":
    tpf = sc.get_tpf()
    mask = None
else:
    tpf = sc.get_tpf_tesscut()
    mask = sc.get_aper_mask(sap_mask=sap_mask, aper_radius=1)
# plot
sc.plot_field(mask=mask)
if args.save:
    fp = Path(args.outdir, sc.target_name + "_field.png")
    pl.savefig(fp, bbox_inches="tight")

stars = sc.get_NEB_depths(mask=mask, depth=depth)
df = sc.get_fpp(flat=flat, bin=bin, plot=True, period=per, epoch=t0)
if args.save:
    fp = Path(args.outdir, sc.target_name + "_plots.png")
    pl.savefig(fp, bbox_inches="tight")
if args.verbose:
    print(df)
fp = Path(args.outdir, sc.target_name + "_fpp.csv")
df.to_csv(fp, index=False)
print("Saved: ", fp)
