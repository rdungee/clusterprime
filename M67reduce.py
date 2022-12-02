"""
    This is the reduction script that was used to reduce the M67 data,
    it is included here as a worked example of how to run the pipeline
    from beginnging to end.
"""

import sys

from os import getcwd

import numpy as np

from astropy.io import fits
from astropy.table import Column, Table

import clusterprime as clp

"""
****************************
***** BEFORE YOU START *****
****************************

You need to have a catalog file, and a bad pixel mask from CFHT for
the MegaPrime instrument. This file should be in a subdirectory of this
package. (e.g., this code is in /home/rdungee/cluster/clusterprime and so
the catalogs are located in /home/rdungee/cluster/clusterprime/catalogs,
note that the code which is run, like this script, is assumed to be run
in the code directory of /home/rdungee/cluster/clusterprime).

At the moment, all the relevant catalog information should be in one file
located in that directory, and so you should put care into making sure that
catalog has all the proper information. It is expected to have all of the
following columns listed as:
name - description
Note that all units are as provided by the respective data archives, I did not
do any conversions, names are mostly the same as well

source_id - the Gaia source_id for whichever data release you are using
ra - Gaia's reported RA
ra_error - Gaia's RA error
dec - Gaia's reported dec
dec_error - error of ^
parallax - Gaia's parallax
parallax_error - Gaia's parallax error
pmra - Gaia's proper motion in RA
pmra_error - error of ^
pmdec - Gaia's proper motion in Dec
pmdec_error - error of ^
ruwe - Gaia's calculated RUWE value (started in EDR3)
phot_g_mean_mag - Gaia's G mag
phot_bp_mean_mag - Gaia's BP mag
phot_rp_mean_mag - Gaia's RP mag
phot_g_mean_mag_error - error on G
phot_bp_mean_mag_error - error on BP
phot_rp_mean_mag_error - error on RP
ps1_id - the PS1 id of the source, found by crossmatching between the PS1 and Gaia catalogs
g - PS1 g mag
r - PS1 r mag
i - PS1 i mag
z - PS1 z mag
g_err - error on g
r_err - error on r
i_err - error on i
z_err - error on z
kine_prob - probability of cluster membership using the proper motions and parallax
main_seq - considered to be a main sequence, single member
multiple - considered to be a member but to bright or too blue to be main sequence
bright - considered to be a member, but too bright for PS1 photometry
ra2k - the Gaia RA convereted to epoch J2000.0 for crossmatch with PS1 and MegaPrime astrometry
de2k - the Gaia Dec convereted to epoch J2000.0 for crossmatch with PS1 and MegaPrime astrometry


The 2nd thing of note is the bad pixel mask, this is provided by CFHT to mask out
known bad pixels in the megaprime detector, it should be downloaded and put into
the code base directory with the name badpixel_mask.fits
(e.g., /home/ryan/thesis/cluster/clusterprime/badpixel_mask.fits)
"""

# Setting overwrite to False to prevent overwriting any data that might
# already exist
overwrite = False
# Setting debug to False since this was only used during the development
debug = False
# Aperture size of 2 is what we settled on, this sets the aperture
# _radius_ at 2 times the seeing FWHM of the image
apsize = 2.00
# Create the config file, we use the default for the various directories
# that means the directory structure is as follows:
"""
     -- Directory structure --

     basedir (holds everything)
        -> clusterprime (holds the pipeline code, including this script)
        -> datadir (hold all the data, reduced or unreduced, can be a symbolic
                    link to a drive with lots of volume)
            -> clustername (holds the unreduced data for a particular cluster
                            ***NOTE*** this data must be sorted into
                            subdirectories corresponding to the visits to the
                            cluster, and they must be named with the scheme
                            obsNNN, where NNN is a zero-padded number starting
                            from 000. Each visit to a cluster consisted of 5
                            seperate images, and so each obsNNN should have 5
                            fits files in it. Anything directly in clustername
                            is ignored, only the files inside of a obsNNN
                            subdirectory are used.)
                -> obsNNN (directories for each epoch of data, should contain
                           5 fits files corresponding to the 5 images in a
                           dither pattern that was used)
            -> clusternamereduce (holds all of the reduced data, everything in
                                  this directory is created by the pipeline
                                  itself)
                -> analysis (contains the output of a specific pipeline run,
                             including the tables of all the timeseries data)
                    -> lightcurves (contains the final output of the pipeline,
                                    a fits file for each source, containing the
                                    lightcurve itself)
                    -> obsNNN (the reduced version of obsNNN above, contains the
                               per image/per chip catalogs, photometry, sky sub
                               versions etc.)

"""
config = clp.Config(getcwd(), 'M67', 'added_overlap_flag', apsize)
"""
     -- Example Directory structure --
     in parenthesis now is the directories that existed given the particular
     config created in the line above this comment

     basedir (/home/rdungee/cluster)                                      ***NOTE USER MADE***
        -> clusterprime (/home/rdungee/cluster/clusterprime)              ***NOTE USER MADE***
        -> datadir (/home/rdungee/cluster/data)                           ***NOTE USER MADE***
            -> clustername (/home/rdungee/cluster/data/M67)               ***NOTE USER MADE***
                -> obsNNN (/home/rdungee/cluster/data/M67/obsNNN)         ***NOTE USER MADE***
            -> clusternamereduced (/home/rdungee/cluster/data/M67reduce)  ***NOTE USER MADE***
                -> analysis (/home/rdungee/cluster/data/M67reduce/added_overlap_flag)
                    -> obsNNN (/home/rdungee/cluster/data/M67reduce/added_overlap_flag/obsNNN)
                    -> lightcurves (/home/rdungee/cluster/data/M67reduce/added_overlap_flag/lightcurves)

"""

# We had 126 visits to M67 in total, meaning there were observations 000
# through 125 (obs000 to obs125), to "parallelize" reducing the data we
# can focus it on a subset of the observations set by this range. This
# line is doing that by user input (e.g. calling
#  'python m67reduce.py 0 20' will run it for obs000 to obs019)
obsis = range(int(sys.argv[1]), int(sys.argv[2]))
# Version that runs it without user input, commented out for now
# obsis = range(0, 20)
# Can also technically split on CCDs since there are 40 chips in the megaprime
# detector. These numbers are the .fits file extensions for each one, they do
# not put image data in the 0th extension, hence the use of [1,41)
ccdis = range(1, 41)

# ***********************
# *** SKY SUBTRACTION ***
# ***********************
# This step runs the sky subtraction on all obsis and ccdis specified above
# Start by constructing a list input directories
# e.g., ['/home/rdungee/cluster/data/M67/obs000',
#        '/home/rdungee/cluster/data/M67/obs001']
# which will pull data from those directories, run the sky subtraction, and then
# save the results to the matching reduce directories:
# /home/rdungee/cluster/data/M67reduce/obs000
#  and /home/rdungee/cluster/data/M67reduce/obs001
newobsdirs = []
for i in obsis:
    newobsdirs.append(config.data / f'obs{i:03}')
# Gather the actual file names for each directory, use a dictionary so that each
# directory is mapped directly to the list of files it contains, making it easy
# to confirm it found the right files
# e.g.,: {'obs000': ['/full/path/0000000.fits.fz', '/full/path/0000001.fits.fz']}
skysub = {}
for obs in newobsdirs:
    pointings = sorted(obs.glob('*.fits.fz'))
    skysub[obs.stem] = pointings
# Add whats been done to the logger for quick read through later
logger = clp.Logger()
# Run the skysubtraction
for fitsgroup in skysub:
    print("Running skysubtract for group: ", fitsgroup)
    clp.reduce.skysubtract(config, skysub[fitsgroup], fitsgroup, logger, 
                           overwrite=overwrite)

# **********************
# *** SOURCE FINDING ***
# **********************
# Find all the sources in each chip of each fits files using DAOStarFinder from
# photutils

# Start by creating a list of all the directories with data we need to process
# e.g., ['/home/rdungee/cluster/data/M67reduce/obs000/0000000p']
# where inside that directory (which was created automatically by the previous
# step) is the sky background subtracted images
chipdirs = []
for i in obsis:
    chipdirs.extend(sorted((config.reduce / f'obs{i:03}').glob('*p')))
# Run source finding for each ccd in ccdis, for each directory from the step
# above
for chipdir, outdir in zip(chipdirs, chipdirs):
    print(f"Find sources for {chipdir.parent.stem}, {chipdir.stem}")
    for ccdi in ccdis:
        print(f"CCD: {ccdi:02}")
        clp.reduce.findsources(chipdir, outdir, ccdi, overwrite=overwrite)
# This produces the untrimmed catalogs, which contain every "source" real or not
# these catalogs are stored with the sky subtracted image data, in the the
# reduced data directory subdirs: obsNNN/MMMMMMMp where MMMMMMMp was the filename
# of the fits file before sky subtraction (this subdir contains a file per ccd
# that has been reduced so far)
# The output is simply the table that DAOStarFinder returns, see the photutils
# docs on this for details on what it contains

# ************************
# *** CATALOG TRIMMING ***
# ************************
# We now trim the catalogs by cross matching against Gaia EDR3.
# ***NOTE*** if you want to change the catalog, you must change the function
# readcatalog contained in aperture.py (clusterprime/reduce/photometry/aperture.py)
# In there the catalog is hardcoded in the first line of the function as
# cat = Table.read(f"catalogs/{cluster}_gaiaedr3_plus_ps1.csv")
# a catalog file that you must construct yourself, it needs to have RA/Dec in
# epoch J2000.0 for comparison with megaprime data, it currently assumes the
# column names are "ra2k" and "de2k" for them respectively

# First build the input and output directory lists to cycle through
catdirs = []
for i in obsis:
    catdirs.extend(sorted((config.reduce / f'obs{i:03}').glob('*p')))
outdirs = []
for cd in catdirs:
    outdirs.append(config.analysis / cd.parent.name / cd.name)
trimsource = zip(catdirs, outdirs)
# Run the trimming, which crossmatches the list from the previous step against
# the Gaia EDR3 catalog, anything without a match to Gaia is discarded, real or
# not it will not have astrometry data
for catdir, outdir in trimsource:
    print(f"Crossmatching sources for {catdir.parent.stem}, {catdir.stem}")
    for ccdi in ccdis:
        print(f"CCD: {ccdi:02}")
        clp.reduce.trimsources(config, catdir, outdir, ccdi,
                               overwrite=overwrite)
# This step creates trimmed catalogs for each ccd in each obs subdir
# Columns include:
#  - xcentroid: x pixel centroid of source
#  - ycentroid: y pixel centroid of source
#  - racentroid: same but now in RA (deg)
#  - deccentroid: same but now in Dec (deg)
#  - gaia_id: the gaia identifier for the source, taken from crossmatching step
#  - nndist: the distance in arcsec to the nearest neighboring source
# The file is a astropy.table .ecsv format so it also has commented header lines,
# which contain metadata:
#  - Skymax: the peak sky value for the chip
#  - Gain: the detector gain
#  - Maxlin: the maxlinearity of the detector, for saturation cutoff
#  - Seeing: the seeing FWHM in pixels of that observation

# ******************
# *** PHOTOMETRY ***
# ******************
# Now compute the aperture photometry, using photutils this is pretty
# straight forward to do, most of the work now is keeping track of everything

# Start with directory lists again, looking for the trimmed catalogs and their
# matching reduced images for computing photometry
catdirs = []
for i in obsis:
   catdirs.extend(sorted((config.analysis / f'obs{i:03}').glob('*p')))
phot = zip(catdirs, catdirs)
# Run the photometry
for catdir, outdir in phot:
   print(f"Photometrizing {catdir.parent.stem}, {catdir.stem}")
   for ccdi in ccdis:
       print(f"CCD: {ccdi:02}")
       clp.reduce.photometrize(config, catdir, outdir, ccdi, 
                       overwrite=overwrite)
# This step creates photometry catalogs in the same directory as the catalogs
# Columns include:
#  - xcentroid: x pixel centroid of source
#  - ycentroid: y pixel centroid of source
#  - gaia_id: the gaia identifier for the source, taken from crossmatching step
#  - nndist: the distance in arcsec to the nearest neighboring source
#  - flux: the flux in the aperture
#  - flux_unc: the uncertainty of the flux from error propagation (Poisson, read noise, sky)
#  - MaskedPix?: if true, bad pixels were in the aperture
#  - SatPix?: if true, saturated pixels were in aperture
#  - mag_inst: instrumental magnitude of flux
#  - mag_unc: uncertainty on mag
# The file is a astropy.table .ecsv format so it also has commented header lines,
# which contain metadata:
#  - Skymax: the peak sky value for the chip
#  - Gain: the detector gain
#  - Maxlin: the maxlinearity of the detector, for saturation cutoff
#  - Seeing: the seeing FWHM in pixels of that observation
#  - AIRMASS: the airmass of the observation
#  - EXPTIME: the exposure time of the obs
#  - MJD-OBS: exposure start time in MJD format
#  - DATE-OBS: the date of the observation in UTC
#  - UTC-OBS: the exposure start time in UTC

# *******************
# *** EPOCH MEANS ***
# *******************
# Now we compute the mean magnitude for each source in a given epoch (i.e., the
# mean of up to 5 magnitudes, creating a point in the light curve)

# Directory set up as usual
obsnums = [config.analysis / f'obs{i:03}' for i in obsis]
epochmeans = zip(obsnums, obsnums)
# Run the epoch mean generating
for obsdir, outdir in epochmeans:
   print(f"Computing Epoch Means for {obsdir.stem}")
   for ccdi in ccdis:
       print(f"CCD: {ccdi:02}")
       clp.reduce.meanofpointings(config, obsdir, outdir, ccdi,
                                  overwrite=overwrite)
# This step generates a table per ccd that sit in the epoch reduced data
# directory (e.g. /home/rdungee/cluster/data/M67reduce/analysisname/obs000) for legacy
# reasons it includes the aperture size (2.00 in this example script) in
# the filename
# Columns include:
#  - gaia_id: the gaia identifier for the source, taken from crossmatching step
#  - magavg: mean mag_inst value
#  - magsdv: the standard deviation of the N measurements
#  - magunc: the estimated uncertainty using error propagation
#  - nndist: the distance in arcsec to the nearest neighboring source
#  - Ninmean: the number of mag_inst values used in computing the above (i.e.,
#             how many didn't have bad pixels/weren't partially on chip/no saturation)
# The file is a astropy.table .ecsv format so it also has commented header lines,
# which contain metadata:
#  - MJD-OBS: exposure start time in MJD format
#  - DATE-OBS: the date of the observation in UTC
#  - UTC-OBS: the exposure start time in UTC
#  - Seeing: the seeing FWHM in pixels of that observation


# ****************************
# *** BY CHIP LIGHT CURVES ***
# ****************************
# Now we run through step of collecting all the indiviual epochs of photometry
# into one table that covers the full time series. This step creates two tables
# per chip in the detector, one that contains the time series photometry itself
# and one that contains the time series of the "meta" values, e.g., the median
# seeing for each epoch, the zero-point correction used for that epoch, etc.
# This step also applies the zero-point correction to put all the epochs in the
# same system before any furhter analysis.
bychiplcs = sorted(config.analysis.glob('obs'+3*'[0-9]'))
outdir = config.analysis
print("Making per chip time series tables")
for ccdi in ccdis:
  print(f"CCD: {ccdi:02}")
  clp.reduce.buildtimeseries(config, bychiplcs, outdir, ccdi, 
                             overwrite=overwrite)
# First output table is ccdMM_apsize_ts.ecsv, it has the following columns:
# - gaia_id - the source's gaia EDR3 id
# - obsNNN_avg - the mean magnitude for obsNNN one column per obs that exists
# - obsNNN_sdv - the standard deviation for obsNNN one column per obs that exists
# - obsNNN_unc - the estimated uncertainty for obsNNN one column per obs that exists
# - obsNNN_Ninavg - the N used in the avg for obsNNN one column per obs that exists
# - nndist - nearest neighbor in arcseconds for that source
# Second output is table ccdMM_metatable.ecsv, it has the following columns:
# - Obs - the obs number (for M67 this was the integer 0-125)
# - MJD-OBS - MJD of the observation, taken as the median of the values for the
#             pointings that went into the epoch
# - DATE-OBS - UT Date of the epoch
# - UTC-OBS - UT time of the epoch, taken as the median of the exposure start
#             times for each pointing in the epoch
# - Seeing - The median seeing of the pointings in that epoch
# - ZP_Corr - the zero-point correction applied to this epoch
# - aperrs - the radius of the apertures used for this epoch

# ***************
# *** ALL TAB ***
# ***************
# This step simply stacks all the by chip tables into one table that contains
# all the time series photometry for all the sources, as such it has the same
# columns
# - gaia_id - the source's gaia EDR3 id
# - obsNNN_avg - the mean magnitude for obsNNN one column per obs that exists
# - obsNNN_sdv - the standard deviation for obsNNN one column per obs that exists
# - obsNNN_unc - the estimated uncertainty for obsNNN one column per obs that exists
# - obsNNN_Ninavg - the N used in the avg for obsNNN one column per obs that exists
# - nndist - nearest neighbor in arcseconds for that source
alltab = sorted(config.analysis.glob('ccd*_ts.ecsv'))
print("Generating all table")
# ***NOTE*** this call needs to be modified if there are not 126 observations in total
clp.reduce.generatealltab(config, alltab, config.analysis, 126,
                        overwrite=overwrite)

# ****************
# *** LC FILES ***
# ****************
# The final step of the reduction pipeline: generating an lc file for each source
# lc files are fits files, the 0th extension contains the light curve and
# current analysis steps add the periodogram as the 1st extension. The data on
# the 0th extension are shaped (N, 10) where N is the number of observations
# so each slice are as follows:
# given hdul = astropy.io.fits.open("examplelc.fits")
# hdul[0].data[:,0] = time of obs in MJD
# hdul[0].data[:,1] = flux [ppm]
# hdul[0].data[:,2] = estimated uncertainty [ppm]
# hdul[0].data[:,3] = obs number (integer value)
# hdul[0].data[:,4] = magnitude [instrumental]
# hdul[0].data[:,5] = estimated uncertainty [mag]
# hdul[0].data[:,6] = standard deviation [mag]
# hdul[0].data[:,7] = seeing [pixels]
# hdul[0].data[:,8] = zero-point correction [mag]
# hdul[0].data[:,9] = aperture radius [pixels]
# All of the above is included in the header, as well as the following additional
# information, as long as it was all available in the catalogs used during the
# data reduction
# GAIA_ID, PS1_ID, RA, RA_ERR, DEC, DEC_ERROR - self explanatory
# PARA, PARA_ERR - the parallax of the target and the parallax error
# PMRA, PMRA_ERR, PMDE, PMDE_ERR - proper motions
# GAIA_G, GAIA_BP, GAIA_RP, G_G_ERR, G_BP_ERR, G_RP_ERR - gaia phot + errors
# PS1_G, PS1_R, PS1_I, PS1_Z, P_G_ERR, P_R_ERR, P_I_ERR, P_Z_ERR - ps1 phot + errors
# KINEPROB - kinematic cluster membership probability
# MAINSEQ - tagged as a main sequence member
# MULTI - tagged as a potential binary
# BRIGHT - too bright for PS1 photometry
# COMPLETE - fraction of the light curve that is filled in
# NNDIST - distance to nearest neighbor in arcsec

datatabfs = sorted(config.analysis.glob('ccd*_ts.ecsv'))
metatabfs = sorted(config.analysis.glob('ccd*_metatable.ecsv'))
indivlcs = list(zip(datatabfs, metatabfs))
print("Generating Light Curve files")
clp.reduce.generatelcfs(config, indivlcs, config.analysis / 'lightcurves')
# output files are located in the lightcurves subdir,
# e.g., /home/rdungee/cluster/M67reduce/analysisname/lightcurves
# file names are the Gaia ID, e.g., 604896498115959296.fits