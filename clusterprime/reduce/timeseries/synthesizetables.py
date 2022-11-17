"""
    Functions for taking the generated light curves from each individual
    chip and turning them into:
        1) an alltab which contains all the light curves for everything
        in the field
        2) .lc files, where one file contains all of the info specific
        to that star, incl. the time series photometry, zero point
        corrections, and pulled gaia/ps1 data
"""

__all__ = ["jointables", "generatealltab", "generatelcfs"]

import numpy as np
import numpy.ma as ma

from astropy.io import fits
from astropy.table import Column, Table, join, vstack
from astropy.stats import sigma_clipped_stats

from clusterprime.helper import overwritecheck

def jointables(alltab, newtab, Nobs):
    """Merges new table into the alltab by a vstack. A special exception
    is made for the case where a source appears in alltab and newtab. In
    this case it chooses whichever one is more complete and drops the
    other from the resulting table.

    Arguments:
    alltab (astropy.Table) - the table which is being put together that
        is to contain all the sources in the entire field
    newtab (astropy.Table) - the table which is to be appended to alltab
    Nobs (int) - the number of obsercations to combine together into one
        table
    """
    overlap, alli, newi = np.intersect1d(alltab["gaia_id"], newtab["gaia_id"],
                                         return_indices=True)
    # Start from the bottom and work our way up to make sure i always
    # refers to the right row, and doesn't fall outside the table
    alli = np.sort(alli)[::-1]
    newi = np.sort(newi)[::-1]
    if overlap.size:
        avgs = [f"obs{i:03}_avg" for i in range(Nobs)]
        for i in range(overlap.size):
            # alli are the indices of alltab which have a duplicate so
            # pull the row which has a match in new tab
            allrow = alltab[alli[i]]
            # Pull the magnitudes from that row and compute how complete
            # it is
            allmags = np.array(list(allrow[avgs]))
            allcomp = np.sum(~np.isnan(allmags)) / allmags.size
            # Do this all again but now for newtab
            newrow = newtab[newi[i]]
            newmags = np.array(list(newrow[avgs]))
            newcomp = np.sum(~np.isnan(newmags)) / newmags.size
            # Keep the version with the higher completeness
            if newcomp > allcomp:
                alltab.remove_row(alli[i])
            else:
                newtab.remove_row(newi[i])
    return vstack([alltab, newtab])

def generatealltab(config, tsfs, outdir, Nobs, overwrite=False):
    """Takes all the individual light curve tables and combines them
    into one large, all inclusive table. Writes the results to a file
    named all_ts.ecsv.

    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    tsfs (list of pathlib.Paths) - the list of paths to the tables that
        are to be combined into the all inclusive table
    outdir (pathlib.Path) - the path to the output directory, where the
        all table file will be written. This function makes the
        directory if it does not already exist
    Nobs (int) - the number of obsercations to combine together into one
        table
    overwrite (bool, optional) - if True quietly clobbers any files that
        are already in the directory, default is False which warns the
        user that no action was taken
    """
    # Create the directories if necessary, exist_ok to quietly skip if
    # they do exist
    outdir.mkdir(parents=True, exist_ok=True)
    outf = outdir / "all_ts.ecsv"
    # Standard overwrite check
    if not overwrite:
        if overwritecheck(outf): return
    tstables = [Table.read(tsf, format="ascii.ecsv") for tsf in tsfs]
    # Pop the first one off the list as a starting point
    alltab = tstables.pop()
    while tstables:
        # Go through the list, pop it off and join it with alltab
        alltab = jointables(alltab, tstables.pop(), Nobs)
    # For a column containing the completeness of each light curve
    completenessvals = np.zeros(len(alltab))
    # Calculate a completeness for every light curve
    for i, row in enumerate(alltab):
        avgs = [f"obs{j:03}_avg" for j in range(Nobs)]
        mags = np.array(list(row[avgs]))
        completenessvals[i] = np.sum(~np.isnan(mags)) / mags.size
    completenesscol = Column(completenessvals, name="Completeness")
    alltab.add_column(completenesscol)
    alltab.write(outf, format="ascii.ecsv", overwrite=overwrite)

def mag2ppms(mags, err):
    """Converts an instrumental magnitude (or magnitudes) into flux in
    units of parts per million about the median value, to match the
    kepler light curves that we have.

    mags (numpy.ndarray) - an array containing the instrumental
        magnitudes
    """

    flux = np.power(10., (-1/2.5)*(mags - 2.5*np.log10(121)))
    fluxerr = (flux * err) / 1.08574
    avg, med, sdv = sigma_clipped_stats(flux)
    frac = (flux - med) / med
    return 1.0e6 * frac, 1.0e6 * (fluxerr / med)

def generatelcfs(config, tabfs, outdir):
    """Takes an all table and generates light curve files for every
    source in that table.

    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    tabfs (list of tuples of pathlib.Paths) - a list of paired paths
        the first in the pair should be a data table, containing the
        time series photometry for a CCD, the second in the pairing
        should be the path to the corresponding meta table
    outdir (pathlib.Path) - the path to the output directory, where the
        all table file will be written. This function makes the
        directory if it does not already exist
    """
    # Create the directories if necessary, exist_ok to quietly skip if
    # they do exist
    outdir.mkdir(parents=True, exist_ok=True)
    clust = config.cluster
    allcat = Table.read(f"catalogs/{clust}_gaiaedr3_plus_ps1.csv")
    # Define the map from table column name to FITS header name
    keymap = {'ra':'RA', 'ra_error':'RA_ERR',
              'dec':'DEC', 'dec_error':'DEC_ERR',
              'parallax':'PARA', 'parallax_error':'PARA_ERR',
              'pmra':'PMRA', 'pmra_error':'PMRA_ERR',
              'pmdec':'PMDE', 'pmdec_error':'PMDE_ERR',
              'phot_g_mean_mag':'GAIA_G', 'phot_g_mean_mag_error':'G_G_ERR',
              'phot_bp_mean_mag':'GAIA_BP', 'phot_bp_mean_mag_error':'G_BP_ERR',
              'phot_rp_mean_mag':'GAIA_RP', 'phot_rp_mean_mag_error':'G_RP_ERR',
              'g':'PS1_G', 'g_err':'P_G_ERR',
              'r':'PS1_R', 'r_err':'P_R_ERR',
              'i':'PS1_I', 'i_err':'P_I_ERR',
              'z':'PS1_Z', 'z_err':'P_Z_ERR'}
    for datatabf, metatabf in tabfs:
        # Read in the paired tables
        metatab = Table.read(metatabf)
        datatab = Table.read(datatabf)
        # Get relevant data from the metatable that is common to all
        # sources on this chip
        obsnum = np.array([int(_[3:]) for _ in metatab['Obs']])
        obsmax = obsnum.size
        magcols = [f"obs{i:03}_avg" for i in range(obsmax)]
        unccols = [f"obs{i:03}_unc" for i in range(obsmax)]
        sdvcols = [f"obs{i:03}_sdv" for i in range(obsmax)]
        for row in datatab:
            # Pull the data from datatab and put it into numpy arrays for
            # easier manipulation
            mags = np.array(list(row[magcols]))
            uncs = np.array(list(row[unccols]))
            sdvs = np.array(list(row[sdvcols]))
            # Calculate flux and flux error as well
            ppms, ppmerr = mag2ppms(mags, uncs)
            # Initialize and fill the data array
            dataarr = np.zeros((obsmax, 10))
            dataarr[:,0] = metatab['MJD-OBS'].data
            dataarr[:,1] = ppms
            dataarr[:,2] = ppmerr
            dataarr[:,3] = obsnum
            dataarr[:,4] = mags
            dataarr[:,5] = uncs
            dataarr[:,6] = sdvs
            dataarr[:,7] = metatab['Seeing'].data
            dataarr[:,8] = metatab['ZP_Corr'].data
            dataarr[:,9] = metatab['aperrs'].data
            # Get some of the gaia parameters and add them to the table's
            # metadata
            allrow = allcat[row['gaia_id'] == allcat["source_id"]]
            # The catalog I'm matching against was trimmed for galaxies,
            # but the one used in source finding wasn't! So if I don't
            # find a match it's probably a galaxy, so skip it
            if len(allrow) < 1: continue
            else: allrow = allrow[0]
            hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU()])
            hdul[0].header['GAIA_ID'] = row['gaia_id']
            hdul[0].header['PS1_ID'] = allrow['ps1_id']
            for key in keymap:
                value = float(allrow[key])
                hdul[0].header[keymap[key]] = value if not np.isnan(value) else -999.
            hdul[0].header['KINEPROB'] = allrow['kine_prob']
            hdul[0].header['MAINSEQ'] = allrow['main_seq']
            hdul[0].header['MULTI'] = allrow['multiple']
            hdul[0].header['BRIGHT'] = allrow['bright']
            mask = np.isnan(dataarr).sum(axis=1).astype(bool)
            dataarr = dataarr[~mask,:]
            hdul[0].data = dataarr
            hdul[0].header['COMPLETE'] = dataarr.shape[0] / obsmax
            dist = row['nndist']
            if np.isnan(dist):
                print("***WARNING***")
                print(f"{row['gaia_id']} has a nearest neighbor distance of {dist}")
                print(f"Completeness: {dataarr.shape[0] / obsmax}")
                dist = -999
            hdul[0].header['NNDIST'] = dist
            hdul[0].header['INDEX0'] = 'Time MJD'
            hdul[0].header['INDEX1'] = 'Flux [ppm]'
            hdul[0].header['INDEX2'] = 'Err [ppm]'
            hdul[0].header['INDEX3'] = 'Obs Num'
            hdul[0].header['INDEX4'] = 'Mags [instr.]'
            hdul[0].header['INDEX5'] = 'Err [mags]'
            hdul[0].header['INDEX6'] = 'Sdv [mags]'
            hdul[0].header['INDEX7'] = 'Seeing [pix]'
            hdul[0].header['INDEX8'] = 'ZPC [mags]'
            hdul[0].header['INDEX9'] = 'Aper Rs [pix]'
            lcf = outdir / f"{row['gaia_id']}.fits"
            hdul.writeto(lcf, overwrite=True)