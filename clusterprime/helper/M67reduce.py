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
            -> clusternamereduce (holds all of the reduced data, everything in
                                  this directory is created by the pipeline
                                  itself)
                -> analysis (contains the output of a specific pipeline run,
                             including the tables of all the timeseries data)

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
            -> clusternamereduced (/home/rdungee/cluster/data/M67reduce)  ***NOTE USER MADE***
                -> analysis (/home/rdungee/cluster/data/M67reduce/added_overlap_flag)

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
#  - ps1_id: the ps1 identifier if found in the ps1 catalog, -1 otherwise
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
#  - ps1_id: the ps1 identifier if found in the ps1 catalog, -1 otherwise
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
obsnums = [config.analysis / f'obs{i:03}' for i in obsis]
epochmeans = zip(obsnums, obsnums)
# Run the epoch mean generating
for obsdir, outdir in epochmeans:
   print(f"Computing Epoch Means for {obsdir.stem}")
   for ccdi in ccdis:
       print(f"CCD: {ccdi:02}")
       clp.reduce.meanofpointings(config, obsdir, outdir, ccdi,
                                  overwrite=overwrite)

# ****************************
# *** BY CHIP LIGHT CURVES ***
# ****************************
bychiplcs = sorted(config.analysis.glob('obs'+3*'[0-9]'))
outdir = config.analysis
print("Making per chip time series tables")
for ccdi in ccdis:
  print(f"CCD: {ccdi:02}")
  clp.reduce.buildtimeseries(config, bychiplcs, outdir, ccdi, 
                             overwrite=overwrite)

# ***************
# *** ALL TAB ***
# ***************
alltab = sorted(config.analysis.glob('ccd*_ts.ecsv'))
print("Generating all table")
clp.reduce.generatealltab(config, alltab, config.analysis, 106,
                        overwrite=overwrite)

# ****************
# *** LC FILES ***
# ****************
datatabfs = sorted(config.analysis.glob('ccd*_ts.ecsv'))
metatabfs = sorted(config.analysis.glob('ccd*_metatable.ecsv'))
indivlcs = list(zip(datatabfs, metatabfs))
print("Generating Light Curve files")
clp.reduce.generatelcfs(config, indivlcs, config.analysis / 'lightcurves')

# ********************
# *** PERIODOGRAMS ***
# ********************
# maskout = []
# lcfs = list((config.analysis / 'lightcurves').glob('*.fits'))
# # lcfs = [config.analysis / 'lightcurves/604705870286863872.fits',
#         # config.analysis / 'lightcurves/604705904646604032.fits']

# outdir = config.analysis / 'periodograms_w_subseries'
# outdir.mkdir(exist_ok=True)
# (outdir / "unphased").mkdir(exist_ok=True)
# (outdir / "mags").mkdir(exist_ok=True)
# (outdir / "seeing").mkdir(exist_ok=True)
# (outdir / "zpc").mkdir(exist_ok=True)

# outf = outdir / 'found_peaks.ecsv'
# outtab = Table(names=("gaia_id"
#                       , "period0_whole", "FAP0_whole", "period1_whole", "period2_whole"
#                       , "period0_ZPC", "FAP0_ZPC"
#                       , "period0_see", "FAP0_see"
#                       , "period0_sub0", "FAP0_sub0", "period1_sub0", "period2_sub0"
#                       , "period0_sub1", "FAP0_sub1", "period1_sub1", "period2_sub1"
#                       , "period0_sub2", "FAP0_sub2", "period1_sub2", "period2_sub2"),
#                dtype=("i8"
#                       , "f8", "f8", "f8", "f8"
#                       , "f8", "f8"
#                       , "f8", "f8"
#                       , "f8", "f8", "f8", "f8"
#                       , "f8", "f8", "f8", "f8"
#                       , "f8", "f8", "f8", "f8"))

# for lcf in lcfs:
#     print(f"Generating periodograms for {lcf.stem}...")

#     hdul = fits.open(lcf)
#     # Completeness check, being more strict this time to mitigate
#     # problems with pulling out the subseries
#     if hdul[0].header['COMPLETE'] < 0.75:
#         continue
#     time, flux, err = hdul[0].data[:,1], hdul[0].data[:,2], hdul[0].data[:,3]
#     see, zpc = hdul[0].data[:,5], hdul[0].data[:,6]

#     # Full light curve
#     period, power, fap_full, oneperlevel = clp.analyze.gen_periodogram(time, flux, err)
#     hdu_full = fits.ImageHDU()
#     hdu_full.header['TYPE'] = 'Full light curve'
#     hdu_full.data = np.c_[period, power]
#     hdul.append(hdu_full)
#     top3_full, peakratios = clp.analyze.N_maxima(power, 3, period)

#     clp.analyze.gen_figure(time, flux, period, power, oneperlevel
#                            , fap_full, hdul[0].header['GAIA_ID'], hdul[0].header['COMPLETE']
#                            , 'Inst. Mag.', 'Time [MJD]', 'mags'
#                            , (outdir / 'unphased'), err)
#     phs, foldflux, folderr = clp.analyze.phase_fold(time, flux, err, top3_full[-1])
#     clp.analyze.gen_figure(phs, foldflux, period, power, oneperlevel
#                            , fap_full, hdul[0].header['GAIA_ID'], hdul[0].header['COMPLETE']
#                            , 'Inst. Mag.', 'Phase', 'folded'
#                            , (outdir / 'mags'), folderr)

#     minper, maxper = 2.5, 150
#     maxfreq, minfreq = 1/minper, 1/maxper
#     freqgrid = np.linspace(minfreq, maxfreq, 10000)

#     # block 1: time < 58680
#     sub0_mask = time < 58680
#     sub0_time, sub0_flux, sub0_err = time[sub0_mask], flux[sub0_mask], err[sub0_mask]
#     if sub0_flux.size <= 2:
#         period = 1/freqgrid
#         power_sub0 = np.zeros(period.size)
#         top3_sub0 = np.zeros(3)
#         fap_sub0 = 100.
#     else:
#         period, power_sub0, fap_sub0, oneperlevel = clp.analyze.gen_periodogram(sub0_time, sub0_flux, sub0_err, freq=freqgrid)
#         top3_sub0, peakratios = clp.analyze.N_maxima(power_sub0, 3, period)

#         clp.analyze.gen_figure(sub0_time, sub0_flux, period, power_sub0, oneperlevel
#                             , fap_sub0, hdul[0].header['GAIA_ID'], hdul[0].header['COMPLETE']
#                             , 'Inst. Mag.', 'Time [MJD]', 'mags_sub0'
#                             , (outdir / 'unphased'), sub0_err)
#         phs, foldflux, folderr = clp.analyze.phase_fold(sub0_time, sub0_flux, sub0_err, top3_sub0[-1])
#         clp.analyze.gen_figure(phs, foldflux, period, power_sub0, oneperlevel
#                             , fap_sub0, hdul[0].header['GAIA_ID'], hdul[0].header['COMPLETE']
#                             , 'Inst. Mag.', 'Phase', 'folded_sub0'
#                             , (outdir / 'mags'), folderr)

#     # block 2: 58680 < time < 59060
#     sub1_mask = (time > 58680) & (time < 59060)
#     sub1_time, sub1_flux, sub1_err = time[sub1_mask], flux[sub1_mask], err[sub1_mask]
#     if sub1_flux.size <= 2:
#         power_sub1 = np.zeros(period.size)
#         top3_sub1 = np.zeros(3)
#         fap_sub1 = 100.
#     else:
#         period, power_sub1, fap_sub1, oneperlevel = clp.analyze.gen_periodogram(sub1_time, sub1_flux, sub1_err, freq=freqgrid)
#         top3_sub1, peakratios = clp.analyze.N_maxima(power_sub1, 3, period)

#         clp.analyze.gen_figure(sub1_time, sub1_flux, period, power_sub1, oneperlevel
#                             , fap_sub1, hdul[0].header['GAIA_ID'], hdul[0].header['COMPLETE']
#                             , 'Inst. Mag.', 'Time [MJD]', 'mags_sub1'
#                             , (outdir / 'unphased'), sub1_err)
#         phs, foldflux, folderr = clp.analyze.phase_fold(sub1_time, sub1_flux, sub1_err, top3_sub1[-1])
#         clp.analyze.gen_figure(phs, foldflux, period, power_sub1, oneperlevel
#                             , fap_sub1, hdul[0].header['GAIA_ID'], hdul[0].header['COMPLETE']
#                             , 'Inst. Mag.', 'Phase', 'folded_sub1'
#                             , (outdir / 'mags'), folderr)

#     # block 3: time > 59060
#     sub2_mask = time > 59060
#     sub2_time, sub2_flux, sub2_err = time[sub2_mask], flux[sub2_mask], err[sub2_mask]
#     if sub2_flux.size <= 2:
#         power_sub2 = np.zeros(period.size)
#         top3_sub2 = np.zeros(3)
#         fap_sub2 = 100.
#     else:
#         period, power_sub2, fap_sub2, oneperlevel = clp.analyze.gen_periodogram(sub2_time, sub2_flux, sub2_err, freq=freqgrid)
#         hdu_sub = fits.ImageHDU()
#         hdu_sub.header['TYPE'] = 'Subsets of LC'
#         hdu_sub.data = np.c_[period, power_sub0, power_sub1, power_sub2]
#         hdul.append(hdu_sub)
#         top3_sub2, peakratios = clp.analyze.N_maxima(power_sub2, 3, period)

#         clp.analyze.gen_figure(sub2_time, sub2_flux, period, power_sub2, oneperlevel
#                             , fap_sub2, hdul[0].header['GAIA_ID'], hdul[0].header['COMPLETE']
#                             , 'Inst. Mag.', 'Time [MJD]', 'mags_sub2'
#                             , (outdir / 'unphased'), sub2_err)
#         phs, foldflux, folderr = clp.analyze.phase_fold(sub2_time, sub2_flux, sub2_err, top3_sub2[-1])
#         clp.analyze.gen_figure(phs, foldflux, period, power_sub2, oneperlevel
#                             , fap_sub2, hdul[0].header['GAIA_ID'], hdul[0].header['COMPLETE']
#                             , 'Inst. Mag.', 'Phase', 'folded_sub2'
#                             , (outdir / 'mags'), folderr)

#     # Seeing values
#     period, power, fap_see, oneperlevel = clp.analyze.gen_periodogram(time, see)
#     top1_see, peakratios = clp.analyze.N_maxima(power, 1, period)
#     hdu_full = fits.ImageHDU()
#     hdu_full.header['TYPE'] = 'Full seeing curve'
#     hdu_full.data = np.c_[period, power]
#     hdul.append(hdu_full)

#     # ZPC values
#     period, power, fap_zpc, oneperlevel = clp.analyze.gen_periodogram(time, zpc)
#     top1_zpc, peakratios = clp.analyze.N_maxima(power, 1, period)
#     hdu_full = fits.ImageHDU()
#     hdu_full.header['TYPE'] = 'Full ZPC curve'
#     hdu_full.data = np.c_[period, power]
#     hdul.append(hdu_full)

#     newrow = (hdul[0].header['GAIA_ID']
#              , top3_full[-1], fap_full, top3_full[-2], top3_full[-3]
#              , top1_see[0], fap_see
#              , top1_zpc[0], fap_zpc
#              , top3_sub0[-1], fap_sub0, top3_sub0[-2], top3_sub0[-3]
#              , top3_sub1[-1], fap_sub1, top3_sub1[-2], top3_sub1[-3]
#              , top3_sub2[-1], fap_sub2, top3_sub2[-2], top3_sub2[-3])
#     outtab.add_row(newrow)
#     outtab.write(outf, format="ascii.ecsv", overwrite=True)
#     hdul.writeto(lcf, overwrite=True)
