"""
    Aperture photometry, this code covers finding sources, trimming the
    list for false positives (by matching against an exisitng catalog
    such as Gaia DR2), and then placing apertures and extracting
    instrumental magnitudes
"""

__all__ = ['readreducedfits', 'findsources', 'readcatalog', 'trimcatalog',
           'findnearestneighbors', 'trimsources', 'photometrize']

import numpy as np
import photutils

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Column, Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from clusterprime.helper import overwritecheck, trimchip

# SNLS mags taken from the CFHT Megacam page, used for filling in the
# appropriate values in the header's photometry block
snls_mags = {"u_SNLS": -0.232, "g_SNLS": -0.087, "r_SNLS": 0.065,
             "i_SNLS": 0.068, "z_SNLS": 0.065}

def readreducedfits(chipf, exts, mask=False, hdr=False):
    """Given the path to a fits file of reduced data this function
    returns the data that is requested, ranging from the different
    extensions to the header and relevant bad pixel mask. The order they
    are returned in is [1, 2, 3, mask, header] as requested (e.g. if you
    request 1, 3 and the mask you get [1, 3, mask]).

    Arguments:
    chipf (pathlib.Path) - the path to the reduced fits file
    exts (list of ints) - the extensions to be returned
    mask (bool, optional) - if True returns the bad pixel mask as well,
        default is False
    hdr (bool, optional) - if True returns the header of the chipf,
        default is False
    """
    out = []
    exts.sort()
    for ext in exts:
        out.append(fits.getdata(chipf, ext=ext, do_not_scale_image_data=True))
    if mask:
        # Extract the ccd index from the filename
        ccdi = int(chipf.name[3:5])
        # Read in the mask and convert to bool
        mask = fits.getdata("badpixel_mask.fits", ext=ccdi)
        mask = trimchip(mask)
        mask = mask < 1
        out.append(mask)
    if hdr:
        out.append(fits.getheader(chipf, ext=1))
    return tuple(out)

def findsources(chipdir, outdir, ccdi, overwrite=False):
    """Find all the potential sources in an image using DAOFind, no
    trimming aside from what is inherent to DAOFind is done. Matching
    against a catalog is done separately

    Arguments:
    chipdir (pathlib.Path) - the path in which the reduced image and its
        prime header files are found. (uses chipdir / "prime.fits.fz"
        and chipdir / f"ccd{ccdi:02}_red.fits.gz")
    outdir (pathlib.Path) - the path to the output directory, where the
        untrimmed catalog file will be written. This function makes the
        directory if it does not already exist
    ccdi (int) - the index of the ccd to find sources on
    overwrite (bool, optional) - if True quietly clobbers any files that
        are already in the directory, default is False which warns the
        user that no action was taken
    """
    # Create the directories if necessary, exist_ok to quietly skip if
    # they do exist
    outdir.mkdir(parents=True, exist_ok=True)
    # Standard overwrite check
    outf = outdir / f"ccd{ccdi:02}_untrimmed_catalog.ecsv"
    if not overwrite:
        if overwritecheck(outf): return
    # Get the prime header with data that's true for all the CCDs
    primef = chipdir / "prime.fits.fz"
    primehdr = fits.getheader(primef, ext=0)
    # And get the chip specific data+header
    chipf = chipdir / f"ccd{ccdi:02}_red.fits.gz"
    img, unc, mask, hdr = readreducedfits(chipf, [1, 3], mask=True, hdr=True)
    # PIXSCAL1 is the same as 2 in all the headers I have seen so far
    # so we use it to compute the FWHM in pixels
    seeing = primehdr["IQMEASUR"]/hdr["PIXSCAL1"]
    # Take the median uncertainty for our significance threshold
    sdv = np.median(unc)
    # Aggressive source finding for flexibility at later stages
    daofind = photutils.DAOStarFinder(3.*sdv, seeing)
    sources = daofind(img, mask=mask)
    # Trim sources with a negative peak
    peakcut = sources["peak"] > 0.
    sources = sources[peakcut]
    # Drop the columns I have no use for
    sources.remove_columns(["id", "npix", "sky", "flux", "mag"])
    # Build an array of pixel coords for generating ra/dec from the WCS
    # The 30 pixel offset is to account for the fact that the WCS 
    # believes the overscan has not been trimmed from the image.
    pixcoords = np.c_[sources["xcentroid"]+30., sources["ycentroid"]]
    w = WCS(hdr)
    radec = w.all_pix2world(pixcoords, 0)
    # Create ra/dec columns for the centroids and add them to my table
    ra = Column(radec[:,0], name="racentroid")
    dec = Column(radec[:,1], name="deccentroid")
    sources.add_column(ra)
    sources.add_column(dec)
    # Finally, write some relevant values I'll want later on to the
    # catalog's metadata
    sources.meta["Skymax"] = hdr["SKYMAX"]
    sources.meta["Gain"] = hdr["GAIN"]
    sources.meta["Maxlin"] = hdr["MAXLIN"]
    sources.meta["Seeing"] = seeing
    sources.write(outf, overwrite=True, format="ascii.ecsv")

def readcatalog(maxra, minra, maxdec, mindec, cluster, tolerance=20.):
    """Reads in and trims the prime catalog to match the bounds of the
    current field of view.
    
    Arguments:
    maxra (float) - the maximum RA value to trim the catalog to
    minra (float) - the minimum RA value to trim the catalog to
    maxdec (float) - the maximum Dec value to trim the catalog to
    mindec (float) - the minimum Dec value to trim the catalog to
    cluster (str) - the cluster for the catalog prefix
    tolerance (float) - the additional boundary around the edges to
        include in units of arcseonds. Useful for addressing concerns of
        precision in the astrometry, or offsets in the pointing
    """
    cat = Table.read(f"catalogs/{cluster}_gaiaedr3_plus_ps1.csv")
    # Adjust the min/max values for the tolerance account for the
    # conversion to degrees
    maxra += tolerance * 2.778e-4
    minra -= tolerance * 2.778e-4
    maxdec += tolerance * 2.778e-4
    mindec -= tolerance * 2.778e-4
    cat = cat[(cat["ra2k"] < maxra) & (cat["ra2k"] > minra)]
    cat = cat[(cat["de2k"] < maxdec) & (cat["de2k"] > mindec)]
    return cat

def trimcatalog(mycat, allcat):
    """Trims the sources found by findsources by crossmatching with the
    input catalog, ideally the crossmatched/combined Gaia and PS1 catalogs

    Arguments:
    mycat (astropy.table) - a catalog of sources found by findsources
        or something similar
    allcat (astropy.table) - the crossmatched/combined Gaia and PS1 catalog
    """
    # My coordinates as an array for the tree
    mycoords = np.c_[mycat["racentroid"], mycat["deccentroid"]]
    # Add id columns to table that we will populate as we find matches
    newcolumns = [Column(np.empty(len(mycat), dtype=int), name="gaia_id")]
    mycat.add_columns(newcolumns)
    # kD-Trees for easy nearest neighbor searches
    tree = cKDTree(mycoords)
    keeprows = np.zeros(len(mycat), dtype=bool)
    for i, source in enumerate(allcat):
        # Find the nearest neighbor in my catalog of found sources
        dist, index = tree.query([source["ra2k"], source["de2k"]], k=1)
        # Within 0.75 arcsecond is assumed to be close enough, this is
        # still a guess as of Aug 6th 2020.
        # NOTE: this distance metric is WRONG but at these small
        # separations it's close enough
        # Specifically, cKDTree uses Euclidean, this is the surface of
        # a sphere!
        if dist > 0.75*2.778e-4:
            # No match found, do nothing since this is what we assume
            pass
        else:
            # Match found, update values in table to reflect this
            keeprows[index] = True
            mycat["gaia_id"][index] = allcat["source_id"][i]
    # Drop sources that had no close matches in the gaia catalog since we
    # rely on Gaia EDR3 for membership
    return mycat[keeprows]

def findnearestneighbors(mycat, allcat):
    """Computes the distance to the nearest neighbor for every source
    in mycat, using their positions as denoted in allcat

    Arguments:
    mycat (astropy.table) - a catalog of sources found by findsources
        or something similar
    allcat (astropy.table) - the crossmatched/combined Gaia and PS1 catalog
    """
    # My coordinates as an array for the tree
    gaiacoords = np.c_[allcat["ra"], allcat["dec"]]
    # kD-Trees for easy nearest neighbor searches
    gaiatree = cKDTree(gaiacoords)
    nndist = np.zeros(len(mycat))
    for i, row in enumerate(mycat):
        sourceid = row["gaia_id"]
        # Select the gaia source and fetch it's ra/dec coordinates
        gaiarow = allcat["source_id"] == sourceid
        gaiara, gaiadec = allcat["ra"][gaiarow], allcat["dec"][gaiarow]
        # Get the two nearest neighbors for a nearby star check
        dist, index = gaiatree.query([gaiara[0], gaiadec[0]], k=2)
        # Instead of trimming we based on a fixed cutoff we now just stash
        # that separation value
        # HARDCODED "BUG": I am not using the right distance metric here
        nndist[i] = dist[1]*3600 # convert to arcsec
    mycat.add_column(nndist, name="nndist")
    return mycat

def trimsources(config, catdir, outdir, ccdi, overwrite=False):
    """Trims the sources for a given pointing and ccd index by matching
    against the Gaia DR2 catalog

    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    catdir (pathlib.Path) - the directory in which the untrimmed catalog
        exists
    outdir (pathlib.Path) - the path to the output directory, where the
        trimmed catalog file will be written. This function makes the
        directory if it does not already exist
    ccdi (int) - the index of the ccd to find sources on
    overwrite (bool, optional) - if True quietly clobbers any files that
        are already in the directory, default is False which warns the
        user that no action was taken
    """
    # Create the directories if necessary, exist_ok to quietly skip if
    # they do exist
    outdir.mkdir(parents=True, exist_ok=True)
    # Standard overwrite check
    outf = outdir / f"ccd{ccdi:02}_catalog.ecsv"
    if not overwrite:
        if overwritecheck(outf): return
    mycat = Table.read(catdir / f"ccd{ccdi:02}_untrimmed_catalog.ecsv",
                       format='ascii.ecsv')
    maxra, minra = np.max(mycat['racentroid']), np.min(mycat['racentroid'])
    maxdec, mindec = np.max(mycat['deccentroid']), np.min(mycat['deccentroid'])
    # Read in the allcat trimmed down to the current chip
    clust = config.cluster
    allcat = readcatalog(maxra, minra, maxdec, mindec, clust)
    mycat = trimcatalog(mycat, allcat)
    mycat = findnearestneighbors(mycat, allcat)
    # Remove now extraneous columns
    mycat.remove_columns(["sharpness", "roundness1", "roundness2", "peak"])
    # And write the results to a file
    mycat.write(outf, format="ascii.ecsv", overwrite=True)

def photometrize(config, catdir, outdir, ccdi, overwrite=False):
    """Computes the instrumental magnitudes from the counts in an
    aperture placed on each source in the trimmed catalog, for a given
    pointing and ccd index

    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    catdir (pathlib.Path) - the directory in which the trimmed catalog
        exists, and the directory to which the trimmed version will be
        written
    outdir (pathlib.Path) - the path to the output directory, where the
        photometry table file will be written. This function makes the
        directory if it does not already exist
    ccdi (int) - the index of the ccd to find sources on
    overwrite (bool, optional) - if True quietly clobbers any files that
        are already in the directory, default is False which warns the
        user that no action was taken
    """
    # Create the directories if necessary, exist_ok to quietly skip if
    # they do exist
    outdir.mkdir(parents=True, exist_ok=True)
    # Standard overwrite check
    outf = outdir / f"ccd{ccdi:02}_{config.apsize:0.2f}_phot.ecsv"
    if not overwrite:
        if overwritecheck(outf): return
    # Read in the trimmed catalog
    sources = Table.read(catdir / f"ccd{ccdi:02}_catalog.ecsv",
                         format="ascii.ecsv")
    pixcoords = np.c_[sources["xcentroid"], sources["ycentroid"]]
    aperr = config.apsize*sources.meta["Seeing"]
    apertures = photutils.CircularAperture(pixcoords, aperr)
    # Read in the image data so we can begin the photmetry extraction
    pointing = config.reduce / catdir.parent.stem / catdir.stem
    chipf = pointing / f"ccd{ccdi:02}_red.fits.gz"
    img, unc, mask = readreducedfits(chipf, [1, 3], mask=True, hdr=False)
    # Extract the photometry
    phot = photutils.aperture_photometry(img, apertures, error=unc, mask=mask)
    # Add the results to our table for when we write this out later
    sources.add_column(Column(phot["aperture_sum"], name="flux"))
    sources.add_column(Column(phot["aperture_sum_err"], name="flux_unc"))
    # Calculate a saturation cut off accounting for the fact that a sky
    # value was subtracted out but still contributed to whether or not
    # it was saturated
    satcutoff = 0.99*(sources.meta["Maxlin"] - sources.meta["Skymax"])
    maskflags = []
    peakflags = []
    cutouts = apertures.to_mask()
    for i, cutout in enumerate(cutouts):
        # Determine if there was a bad pixel in the aperture, or it ran
        # outside the detector
        xoutoflb = (apertures[i].positions[0]-apertures[i].r) < 0
        youtoflb = (apertures[i].positions[1]-apertures[i].r) < 0
        xoutofub = (apertures[i].positions[0]+apertures[i].r) > img.shape[1]-1
        youtofub = (apertures[i].positions[1]+apertures[i].r) > img.shape[0]-1
        mc = cutout.cutout(mask)
        if np.max(mc) or xoutoflb or xoutofub or youtoflb or youtofub:
            maskflags.append(True)
        else:
            maskflags.append(False)
        # Now we check if the source was saturated with our satcutoff
        ic = cutout.cutout(img)
        if np.max(ic) > satcutoff:
            peakflags.append(True)
        else:
            peakflags.append(False)
    # Add the pixel area and flag columns to the table
    sources.add_columns([Column(maskflags, name="MaskedPix?"),
                         Column(peakflags, name="SatPix?")])
    # Drop the columns that have no use beyond this point
    sources.remove_columns(["racentroid", "deccentroid"])
    # Extract photometry-related info from the fits header
    primehdr = fits.getheader(pointing / "prime.fits.fz", ext=0)
    sources.meta["ISPHOTOM"] = primehdr["ISPHOTOM"]
    sources.meta["APERR"] = aperr
    hdr = fits.getheader(chipf, ext=1)
    sources.meta["AIRMASS"] = hdr["AIRMASS"]
    sources.meta["EXPTIME"] = hdr["EXPTIME"]
    sources.meta["MJD-OBS"] = hdr["MJD-OBS"]
    sources.meta["DATE-OBS"] = hdr["DATE-OBS"]
    sources.meta["UTC-OBS"] = hdr["UTC-OBS"]
    sources.meta["EXPTIME"] = hdr["EXPTIME"]
    # Calculate the instrumental magnitudes, using the equation
    # m = -2.5*log_10(N_e-) + 2.5*log_10(exptime) (Magnier et al 1992)
    mags = -2.5*np.log10(phot["aperture_sum"]) + 2.5*np.log10(hdr["EXPTIME"])
    # From partial derivatives uncertainty on mag is approximately given
    # by sig_mag = -2.5 * sig_N / (N * ln(10)) and -2.5/ln(10) = 1.08574
    uncs = 1.08574 * np.abs(phot["aperture_sum_err"]/phot["aperture_sum"])
    # Add them to the catalog
    sources.add_column(Column(mags, name="mag_inst"))
    sources.add_column(Column(uncs, name="mag_unc"))
    # Write the output to a file
    sources.write(outf, format="ascii.ecsv", overwrite=True)
    