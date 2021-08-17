# Modeling Notes
## Transiting Exoplanet Vetting
* Exoplanet Detection Identification Vetter (EDI-Vetter; [Zink+2020](https://arxiv.org/pdf/2001.11515.pdf))
  - source pin-pointing; varying photometric aperture
  - binary blending 
  - even-odd depth difference
    - odd transit depth is a different from the even depths
  - check for secondary eclipse; note that some hot Jupiter have secondary eclipse too
  - period harmonic test; period aliasing
  - min Porb based on Roche lobe overflow; see Hippke+2019
  - duration and Porb consistency
    - tdur/P > 0.1 is orbiting close to the surface of the star; exception is ultra-hot Jupiters
* VESPA
* TRICERATOPS
* TERRA

## Stellar parameter estimation
* Kepler/K2 host stars 
  - M-dwarfs: [Hardegree-Ullman+2019](https://arxiv.org/abs/1905.05900)
  - C1-7: [Dressing+2017]()
  - C1-13: [Hardegree-Ullman+2020]()
  - C1-13: [Wittenmyer+2020](https://arxiv.org/abs/2005.10959)
* star populations in the Galactic context, chemo-kinematic properties: [Carillo+2019]()
* photometrically-derived stellar properties: Casagrande et al. (2019) and Deacon et al. (2019)
* Mann+
* extinction cal be estimated using `dustmaps`
* spectroscopic parameters (Teff, log g, [Fe/H]) from spectroscopy are used for isochrones fitting using `isochrones`
  - effective temperature (Teff),
  - surface gravity (log g)
  - metallicity ([M/H])
    - the assumption that the iron abundance [Fe/H] can be a proxy (or even equal) to [M/H] breaks down for metal-poor star
  - 2MASS (J, H, Ks) (Skrutskie et al. 2006)
    - adding infrared photometry does not automatically yield better results (Mayo+2018)
  - Gaia (G, GRP, GBP) photometric magnitudes
  - Gaia DR2 parallax (Gaia Collaboration et al. 2018) where available
    - inflate by adding 0.1 mas in quadrature to parallax error to account for systematics (Luri+201?)
* evidence of hidden binarity can be hinted from significant excess astrometric noise in Gaia DR2 and large absolute radial velocities--e.g. >3σ larger than the expected RV precision for stars of their temperature (Katz et al. 2019)
* 20% of binaries have feh<-1: https://arxiv.org/pdf/2002.00014.pdf
* giant host star candidates with log g <3.0 are more likely to be false positives, e.g. wherein a grazing eclipse by an M dwarf can produce the K2 transit-like signal, or where the transiting object orbits a different star, as postulated by the analysis of Kepler giants in Sliski, & Kipping (2014).
* evolutionary phase of star can be determined from the EEP number
* seismic detections indicate star's evolved nature
* Look for secondary set of spectral lines
* For sanity check, confirm none fall in unphysical regions of parameter of HRD/CMD

* stellar age can be constrained from color via `isochrones`, stellar rotation period, and rotation amplitude (Morris+2020)
* stellar rotation period can be measured using generalized Lomb-Scargle (GLS) periodogram (Zeichmeister+201?)
* AO/speckle imaging can reveal unresolved nearby companion (<1")
* compare with results from Hardegree-Ullman et al. (2020) (starhorse?) based on LAMOST spectra and Wittenmyer+ based on AAT/HERMES spectra for K2/C1-C13

## Transit light curve modeling
### Lightcurve modeling
* create light curve from tpf
* use SAP and PDCSAP lightcurves
* EVEREST (Luger et al. 2016), K2SFF light curves

### Systematics correction
* `biweight` smoothing filter available in `wotan` seems to be optimum for recovering shallow transits (Hippke+2019)
* flux dilution due to nearby stars within photometric aperture should be taken into account

### Transit fitting/ framework
* use quadratic limb-darkening using q-space parameterization (Kipping+201?)
* sample in impact parameter using transformation by Espinoza+201?
* stellar density can be used as prior in transit modeling
* physics-based contamination modeling (Parviainen+201?)
* `allesfitter` ()

### asteroseismology
* modeling by Stello et al. (2017), which uses the method by Huber et al. (2009) with the improvements described in Huber et al. (2011) and in Yu et al. (2018)
* derive physical parameters using the seismic ∆ν and νmax and the methods of Hon et al. (2018) and Sharma et al. (2016)


### Derived planet parameters
* planet radius
* insolation
* incident flux received by the planet in units of the solar constant using the semi-major axis and stellar luminosity
* equilibrium temperature for each planet (Teq) using both ”hot dayside” and well-mixed models, which assume that the planet re-radiates as a blackbody over 2π and 4π steradians respectively (Kane & Gelino 2011)
* HZ boundaries for each of the stars, using the formalism described by Kopparapu et al. (2013, 2014)
* ”runaway greenhouse” and ”maximum greenhouse” boundaries (referred to as the ”conservative” HZ) and the empirically derived ”recent Venus” and ”early Mars” boundaries (referred to as the ”optimistic” HZ)

## False positive scenarios
1. EB: manifestation of an EB system in light curve:
  - odd-even transit depth variation in phase-folded light curves: 
  - the presence of secondary eclipses: secondary eclipse denote the decrease of total flux of the system when the self-luminous companion eclipses the primary star; hot Jupiters and young brown dwarfs also show secondary eclipses
  - V-shaped transits: as a consequence of the small radius ratio of a binary system
  - RV measurements put a constraint on the minimum dynamical mass of the bound companion in case of non-detection of orbital period observed from photometry; if companion mass constraint is << Mjup then companion is likely either a brown dwarf or a planet  
2. NEB
  - wide field of view transit surveys usually suffer from small pixel size. Kepler have a ~4pix/a rcsec while TESS has ~21 pix/arcsec. Doing photometry with an aperture of a few pixels in radius would inevitably include more than 1 star especially when the field is near the galactic plane where stellar density is high. 
  - an NEB bright enough relative to the target can reproduce the detected transit; in particular m1-m2=dmag=2.5log10(f1/f2); nearby stars fainter than a certain dmag do not have enough flux to reproduce the observed transit; obs_depth=true_depth/1+10**(0.4*dmag); 
  - the transit shape can also help rule out EBs, see Seager & Ornalles-Mallen+2003
3. BEB
  - unresolved background EBs 
  - two archival images taken e.g. 50 years apart can show displacement stars across the sky by a few arcseconds for nearby stars that have large proper motions; demonstrating the absence of a bright enough star in the same location in the archival image rules out a BEB scenario 
4. HEB
  - hierarchical EB where the secondary is itself a binary system can produce otherwise large eclipses that have been diluted due to flux from the primary star and hence mimics a transit

* FPP calculation can be done using `triceratops` and/or `vespa`
* Shporer et al. (2017) identified three K2 validated Jupiter-sized planets as confirmed low-mass stellar secondaries. To explain the initial misclasification the authors state different possible causes as the indistinguishable radius distribution of the smaller stars and gas giants planets, the difficulty in detecting a secondary eclipse in eccentric orbits and a poor characterization of the host star. Another source of missidentification can be the presence of an unnoticed background eclipsing binary within the photometric Kepler aperture, as we discuss below for our targets with visible background stars in the Pan-STARRS-1 images and Gaia DR2. Cabrera et al. (2017) identified three K2 validated super-Earth-sized planets as background eclipsing binaries acquiring ground-based high-resolution images in which the binaries were left out of the photometric aperture. Although planet validation techniques are useful tools to get a quick approach to the goodness of planet candidates, there exist the possibility of missclasifications, so that detailed follow-up is necessary
to confirm them.

## Planet characterization
* RV modeling

## Target selection/follow-up
* which planets would be best suited for followup activities (Chandler et al. 2016; Kempton et al. 2018; Ostberg & Kane 2019)
* planet near or within Fulton gap is therefore unique and interesting
* hot super-Earth desert postulated by Lundkvist et al. (2016) exists in planet radius as a function of incident stellar flux
parameter space in the region between 2.2-3.8 R⊕ and Sinc >650 F⊕.

## Radius-gap
* stars hosting Kepler planet candidates revealed a “radius gap” (Fulton et al. 2017), with planets of 1.5-2.0R⊕ apparently depleted by more than a factor of two;
* radius gap was shown by Van Eylen et al. (2018) to have a slope dependent on orbital period, with a slope of dlogR
dlogP of approximately -1/9, a value corroborated by Gupta & Schlichting (2019) a
