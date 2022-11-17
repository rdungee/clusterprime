"""
    Taking all the individual photometry results and collecting them
    into a light curve. This includes applying a zeropoint correction
    to each point in the series, does this by chip
"""

__all__ = ["getmedianmetavals", "collectepochvalues", "zeropointcorrect",
           "meanofpointings", "collectepochmeans", "buildtimeseries"]

import numpy as np
import numpy.ma as ma
import photutils

from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.table import Column, Table
from scipy.spatial import cKDTree

from clusterprime.helper import overwritecheck

def getmedianmetavals(catalogs):
    """Computes the median value for the metadata to associate with each
    lightcurve's data point. Specifically, the date/time (both as a UTC
    date/time and an MJD), and the FWHM

    Arguments:
    catalogs (list of astropy.tables) - A list containing the output
        catalog from running aperture photometry on each pointing in
        an epoch
    """
    # Arrays to store the magnitudes that we will average together
    mjds = np.empty(len(catalogs))
    fwhms = np.empty(len(catalogs))
    aperrs = np.empty(len(catalogs))
    date = np.empty(len(catalogs), dtype="<U10")
    time = np.empty(len(catalogs), dtype="<U11")
    for i, cat in enumerate(catalogs):
        mjds[i] = cat.meta["MJD-OBS"]
        date[i] = cat.meta["DATE-OBS"]
        time[i] = cat.meta["UTC-OBS"]
        fwhms[i] = cat.meta["Seeing"]
        aperrs[i] = cat.meta["APERR"]
    medidx = np.argsort(mjds)[len(mjds)//2]
    return mjds[medidx], date[medidx], time[medidx], np.mean(fwhms), np.mean(aperrs)

def collectepochvalues(uniqueids, catalogs):
    """Collects values from the different catalogs and creates arrays
    for each one. Values which are absent (off the chip, saturated, etc)
    have NaNs in the returned arrays for floats or -1 for integers.

    Arguments:
    uniqueids (numpy.ndarray of ints) - The array of ids that occur at
        least once in the catalogs, each id only occurring once in this
        array
    catalogs (list of astropy.Tables) - The catalog of sources for which
        photometry has been extracted, each one is a pointing in an
        epoch
    """
    mags = np.full((uniqueids.size, len(catalogs)), np.NaN)
    uncs = np.full((uniqueids.size, len(catalogs)), np.NaN)
    dist = np.full((uniqueids.size, len(catalogs)), np.NaN)
    for i in range(mags.shape[1]):
        cat = catalogs[i]
        for row in cat:
            if row['MaskedPix?'] or row['SatPix?']:
                continue
            else:
                idx = row['gaia_id'] == uniqueids
                mags[idx, i] = row['mag_inst']
                uncs[idx, i] = row['mag_unc']
                dist[idx, i] = row['nndist']
    # Apply an actual mask now
    mags = ma.masked_invalid(mags)
    uncs = ma.masked_invalid(uncs)
    Ninmean = np.sum(~uncs.mask, axis=1)
    return mags, uncs, Ninmean, dist

def zeropointcorrect(mags, completeness, nndist, cutoff=2.):
    """Given a batch of magnitudes measured at different times (i.e.
    subsequent pointings in an epoch, or different epochs), calculates
    the zero point correction to tie all of them to the same magnitude
    system.

    Arguments:
    mags (numpy.ndarray of floats) - the array of magnitudes to zero
        point correct. The shape should be 2D (N_stars, N_obs), i.e.,
        each row is treated as being all the magnitudes for a particular
        star.
    completeness (float or int) - the completeness cut to use as well.
        If a float, it is interpreted as a fraction of the total number
        of observations (mags.shape[1]). If a number it is treated as
        a number of observations (i.e. completeness = 3 will use any
        star with at least 3 observations)
    nndist (numpy.ndarray of floats) - the array of the distance to the
        nearest neighbor of that source, in units of arcsec
    cutoff (optional, float) - the distance cutoff to use in combo with
        nndist, default is 2
    """
    mags = mags.copy()
    if type(completeness) == int:
        N = np.sum(~mags.mask, axis=1)
    else:
        N = np.sum(~mags.mask, axis=1)/mags.shape[1]
    Nmask = N >= completeness
    distmask = nndist > cutoff
    # Can't let the sources that are too close bias the scatter calculation
    mags4scatter = mags[distmask, :]
    scatter = ma.std(mags4scatter, axis=1)
    medscatter = np.nanquantile(scatter.filled(np.NaN), 0.5)
    scattermask = (ma.std(mags, axis=1).filled(np.NaN) < medscatter) & Nmask & distmask
    zps = np.zeros(mags.shape[1])
    for i in range(1, mags.shape[1]):
        zps[i] = ma.median(mags[scattermask,0] - mags[scattermask,i])
        mags[:,i] += zps[i]
    return zps, mags

def meanofpointings(config, obsdir, outdir, ccdi, overwrite=False):
    """Takes the photometry results from each pointing in an epoch and
    produces a new catalog which is the mean of the 5 pointings, masking
    out any that happened to fall on bad pixels or became saturated

    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    obsdir (pathlib.Path) - the directory in which the pointing subdirs
        exist, each pointing subdir needs to contain the _phot.ecsv
        table for that pointing/ccdi, specifically it looks for:
        "obsdir.glob('*p') / ccd{ccdi:02}_phot.ecsv"
    outdir (pathlib.Path) - the path to the output directory, where the
        epoch mean photometry table file will be written. This function
        makes the directory if it does not already exist
    ccdi (int) - the index of the ccd to find sources on
    overwrite (bool, optional) - if True quietly clobbers any files that
        are already in the directory, default is False which warns the
        user that no action was taken
    """
    # Create the directories if necessary, exist_ok to quietly skip if
    # they do exist
    outdir.mkdir(parents=True, exist_ok=True)
    outf = outdir / f"ccd{ccdi:02}_{config.apsize:0.2f}_phot.ecsv"
    # Standard overwrite check
    if not overwrite:
        if overwritecheck(outf): return
    # Need to read in each of the photometry catalogs
    catalogs = []
    pointings = sorted(obsdir.glob('*p'))
    for pointing in pointings:
        photf = pointing / f"ccd{ccdi:02}_{config.apsize:0.2f}_phot.ecsv"
        catalogs.append(Table.read(photf, format="ascii.ecsv"))
    # Find the set of IDs that appear at least once in each catalog
    uniqueids = []
    for cat in catalogs:
        uniqueids = uniqueids + list(cat["gaia_id"])
    uniqueids = np.unique(np.array(uniqueids))
    # Collect the values for each source into arrays, each row (0th
    # index) is a source, and there are 5 columns
    mags, uncs, Ninmean, nndist = collectepochvalues(uniqueids, catalogs)
    nndist = np.nanmin(nndist, axis=1)
    zps, mags = zeropointcorrect(mags, 4, nndist)
    magavgs = ma.mean(mags, axis=1)
    magsdvs = ma.std(mags, axis=1, ddof=1)
    uncs = ma.power(uncs/Ninmean[:, np.newaxis], 2)
    maguncs = np.sqrt(ma.sum(uncs, axis=1))
    # Now prep and then write the table to disk
    epochcat = Table()
    mjd, date, time, fwhm, aperr = getmedianmetavals(catalogs)
    epochcat.meta["MJD-OBS"] = mjd
    epochcat.meta["DATE-OBS"] = str(date)
    epochcat.meta["UTC-OBS"] = str(time)
    epochcat.meta["Seeing"] = fwhm
    epochcat.meta["APERR"] = aperr
    epochcat.meta["Zero Points"] = zps
    epochcat.add_column(Column(uniqueids, name="gaia_id"))
    epochcat.add_column(Column(magavgs.filled(np.NaN), name="magavg"))
    epochcat.add_column(Column(magsdvs.filled(np.NaN), name="magsdv"))
    epochcat.add_column(Column(maguncs.filled(np.NaN), name="magunc"))
    epochcat.add_column(Column(nndist, name="nndist"))
    epochcat.add_column(Column(Ninmean, name="Ninmean"))
    epochcat.write(outf, format="ascii.ecsv", overwrite=True)

def collectepochmeans(uniqueids, fitsgroups, catalogs):
    """Collects values from the different epoch mean catalogs and
    creates arrays for each one. Values which are absent have NaNs in
    the returned arrays for floats or -1 for integers.

    Arguments:
    uniqueids (numpy.ndarray of ints) - The array of ids that occur at
        least once in the catalogs, each id only occurring once in this
        array
    fitsgroups (list of strs) - the list of epochs, in order. An epoch
        is denoted by the str "obs###" where ### is the zero-padded int
        number for the observation in the sequence (e.g. obs000)
    catalogs (list of astropy.Tables) - The catalog of sources for which
        photometry has been extracted, each one is a pointing in an
        epoch
    """
    mags = np.full((uniqueids.size, len(fitsgroups)), np.NaN)
    sdvs = np.full((uniqueids.size, len(fitsgroups)), np.NaN)
    uncs = np.full((uniqueids.size, len(fitsgroups)), np.NaN)
    over = np.full((uniqueids.size, len(fitsgroups)), np.NaN)
    Ninavgs = np.zeros((len(uniqueids), len(fitsgroups)), dtype=int)
    for i in range(mags.shape[0]):
        uid = uniqueids[i]
        for j in range(mags.shape[1]):
            cat = catalogs[fitsgroups[j]]
            idx = uid == cat["gaia_id"]
            if np.sum(idx):
                mags[i,j] = cat["magavg"][idx]
                sdvs[i,j] = cat["magsdv"][idx]
                uncs[i,j] = cat["magunc"][idx]
                over[i,j] = cat["nndist"][idx]
                Ninavgs[i,j]= cat["Ninmean"][idx]
    # Apply an actual mask now
    mags = ma.masked_invalid(mags)
    sdvs = ma.masked_invalid(sdvs)
    uncs = ma.masked_invalid(uncs)
    return mags, sdvs, uncs, Ninavgs, over

def buildtimeseries(config, obsdirs, outdir, ccdi, overwrite=False):
    """Put together the timeseries data for a particular ccd index.

    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    obsdirs (list of pathlib.Path) - the list of directories in which
        the epoch mean photometry tables exist, each obsdir needs to
        contain the _phot.ecsv table for that epoch's mean/ccdi,
        specifically it looks for: "obsdir / ccd{ccdi:02}_phot.ecsv"
    outdir (pathlib.Path) - the path to the output directory, where the
        chip specific time series table file will be written. This
        function makes the directory if it does not already exist
    ccdi (int) - the index of the ccd to find sources on
    overwrite (bool, optional) - if True quietly clobbers any files that
        are already in the directory, default is False which warns the
        user that no action was taken
    """
    # Create the directories if necessary, exist_ok to quietly skip if
    # they do exist
    outdir.mkdir(parents=True, exist_ok=True)
    outf = outdir / f"ccd{ccdi:02}_{config.apsize:0.2f}_ts.ecsv"
    # Standard overwrite check
    if not overwrite:
        if overwritecheck(outf): return
    # Need to read in each of the photometry catalogs
    catalogs = {}
    for obsdir in obsdirs:
        catf = obsdir / f"ccd{ccdi:02}_{config.apsize:0.2f}_phot.ecsv"
        catalogs[obsdir.stem] = Table.read(catf, format="ascii.ecsv")
    # Make a master list of source_ids to account for differences in
    # each epoch's mean pointing
    uniqueids = []
    for obs in catalogs:
        uniqueids = uniqueids + list(catalogs[obs]["gaia_id"])
    uniqueids = np.unique(np.array(uniqueids))
    # No need for indivdual pointings so now generate an ordered list of
    # the epochs
    obsnums = sorted([od.stem for od in obsdirs])
    # Place to store all the mean magnitudes
    mags, sdvs, uncs, Ninavgs, over = collectepochmeans(uniqueids, obsnums, catalogs)
    # Possible hard-coded bug here, may need to not do min but instead do a mask per epoch
    # type of deal for the ZPC
    over = np.nanmin(over, axis=1)
    # Store some results of the processing
    zps, mags = zeropointcorrect(mags, 0.5, over)
    # Generate the timeseries table for output
    tscat = Table()
    tscat.add_column(Column(uniqueids, name="gaia_id"))
    for j in range(mags.shape[1]):
        tscat.add_column(Column(mags[:,j], name=f"{obsnums[j]}_avg"))
        tscat.add_column(Column(sdvs[:,j], name=f"{obsnums[j]}_sdv"))
        tscat.add_column(Column(uncs[:,j], name=f"{obsnums[j]}_unc"))
        # tscat.add_column(Column(over[:,j], name=f"{obsnums[j]}_nndist"))
        tscat.add_column(Column(Ninavgs[:,j], name=f"{obsnums[j]}_Ninavg"))
    tscat.add_column(Column(over, name="nndist"))
    tscat.write(outf, format="ascii.ecsv", overwrite=True)
    # Now make a time table, which matches a column name in the above
    # table to an MJD and Date/UTC time from the median of the dates 
    mjds = []
    dates = []
    times = []
    fwhms = []
    aperrs = []
    for fg in obsnums:
        mjds.append(catalogs[fg].meta["MJD-OBS"])
        dates.append(catalogs[fg].meta["DATE-OBS"])
        times.append(catalogs[fg].meta["UTC-OBS"])
        fwhms.append(catalogs[fg].meta["Seeing"])
        aperrs.append(catalogs[fg].meta["APERR"])
    # Generate and writeout the time table
    metatab = Table()
    metatab.add_column(Column(obsnums, name="Obs"))
    metatab.add_column(Column(mjds, name="MJD-OBS"))
    metatab.add_column(Column(dates, name="DATE-OBS"))
    metatab.add_column(Column(times, name="UTC-OBS"))
    metatab.add_column(Column(fwhms, name="Seeing"))
    metatab.add_column(Column(zps, name="ZP_Corr"))
    metatab.add_column(Column(aperrs, name="aperrs"))
    metaf = outf.parent / f"ccd{ccdi:02}_metatable.ecsv"
    metatab.write(metaf, format="ascii.ecsv", overwrite=True)
