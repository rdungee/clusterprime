"""
    The first step in the pipeline is sky subtraction, since CFHT does
    the flat fielding and fringe corrections before delivering the data.
    The default method is based on DAOPhot, specifically the photutils
    package's implementation, but ideally can be adapted to fit other
    methods easily (especially anything with the same or similar syntax)
"""
__all__ = ["makedirectories", "writeprimefits", "estimatesky",
           "skysubtractchip", "skysubtract"]

import astropy.io.fits as fits
import numpy as np
import photutils.background as bkg

from clusterprime.helper import trimchip, overwritecheck, Logger

def makedirectories(config, fitsfiles, obsnum):
    """Makes the directory structure for a set of reduced fits files,
    where there are subdirectories "obs[obsnum]" w/ obsnum formatted to
    3 digits (e.g. obs000). Inside this directory each pointing gets a
    subdirectory (5 total). In each of these pointing subdirectories are
    the reduced fits images. Each chip will not be a separate single
    extension fits file, which reduces overall RAM usage.

    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    fitsfiles (list of str) - the path to the fits files which are
        going to be reduced and thus potentially need a new reduced
        data subdirectory
    obsnum (str) - the observation number, formatted as obs### where ###
        is the zero-padded, zero-indexed number in the sequence
    """
    # Start with the obs group's directory, indiciated by its number
    obsdir = config.reduce / obsnum
    obsdir.mkdir(exist_ok=True)
    # Now make directory for each pointing
    fitsdirs = []
    for fitsf in fitsfiles:
        # Fetch the part of the file name that uniquely identifies the
        # pointing for use as the directory name
        pointing = fitsf.name
        pointing = pointing.split('.')[0]
        pointingdir = obsdir / pointing
        pointingdir.mkdir(exist_ok=True)
        # Save the paths in a list so I have both the fits file and the 
        # directory to save it's processed output to
        fitsdirs.append(pointingdir)
    return fitsdirs

def writeprimefits(fitsfiles, fitsdirs, overwrite=False):
    """Writes the primary header for each of the files into a fits file
    with no image data, this is so the primary header data is colocated
    with all the reduced images.

    Arguments:
    fitsfiles (list of str) - the path to the fits files which are
        going to be reduced and thus potentially need a new reduced
        data subdirectory
    fitsdirs (list of pathlib.Path) - the pathlib.Path paths to the
        reduced data directories where these fits headers are to be
        stored
    overwrite (bool, optional) - if True quietly clobbers any files that
        are already in the directory, default is False which warns the
        user that no action was taken
    """
    for fitsf, fitsd in zip(fitsfiles, fitsdirs):
        # Output file name, for overwrite check
        outf = fitsd / "prime.fits.fz"
        # Standard if not overwriting and it exists, warn and skip
        if not overwrite:
            if overwritecheck(outf): continue
        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdr = fits.getheader(fitsf, ext=0)
        hdul[0].header = hdr
        hdul.writeto(outf, overwrite=True)
        hdul.close()

def estimatesky(fitsf, badpixelmask, ccdi, gain=1.):
    """A wrapper to the photutils MMMBackground procedure for sky
    background estimation. The default behavior is to generate a grid
    of estimated values and interpolate uisng a 3rd order polynomial

    Arguments:
    fitsf (str or pathlib.Path) - the path to the fits file to perform
        a sky estimation/subtraction on
    badpixelmask (numpy.ndarray) - the bad pixel mask, True indicates
        that the pixel is bad
    ccdi (int) - the ccd index indicating the extension to pull from
        fitsf (the ext keyword arg to astropy.io.fits.getdata())
    gain (float, optional) - the gain to be used, default is 1, set this
        if you want the noise estimate to be more accurate
    """
    bkgval = bkg.MMMBackground()
    bkgrms = bkg.StdBackgroundRMS()
    bkgint = bkg.BkgZoomInterpolator()

    img = fits.getdata(fitsf, ext=ccdi)
    # Apply the gain so we are now in e- units, meaning we can get the
    # Poisson noise from the square root of the img
    img = gain * trimchip(img)
    bkggen = bkg.Background2D(img, (461, 256), mask=badpixelmask, 
                              bkg_estimator=bkgval, bkgrms_estimator=bkgrms,
                              interpolator=bkgint)
    img = img - bkggen.background
    # Retun the sky subtracted image, the sky which was subtracted out
    # and the variance (for ease of uncertainty propagation)
    return img, bkggen.background, np.power(bkggen.background_rms, 2.)

def skysubtractchip(config, fitsfile, fitsdir, ccdi, logger, 
                    overwrite=False):
    """Process a single ccd by subtracting the sky and writing out a
    multi-extension fits file which contains the sky-subtracted image,
    the sky background itself, and the noise map
    
    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    fitsfile (str or pathlike.Path) - the path to the fits file to
        process
    fitsdir (pathlike.Path) - the output directory
    ccdi (int) - the ccd index indicating the extension to pull from
        fitsf (the ext keyword arg to astropy.io.fits.getdata())
    logger (clusterprime.Logger) - the logger to store basic info in
    overwrite (bool, optional) - if True quietly clobbers any files that
        are already in the directory, default is False which warns the
        user that no action was taken
    """
    outf = fitsdir / f"ccd{ccdi:02}_red.fits.gz"
    # Standard check to see if we skip
    if not overwrite:
        if overwritecheck(outf): return
    print(f"Sky subtracting chip {ccdi}")
    # Bad pixel mask
    badpixelmask = fits.getdata('badpixel_mask.fits', ext=ccdi)
    badpixelmask = trimchip(badpixelmask)
    badpixelmask = badpixelmask < 1
    # Extension specific header
    hdr = fits.getheader(fitsfile, ext=ccdi)
    gain = hdr['GAIN']
    rnmap = np.empty(badpixelmask.shape)
    rnmap[:,:1024] = np.power(hdr['RDNOISEA'], 2.)
    rnmap[:,1024:] = np.power(hdr['RDNOISEB'], 2.)
    img, sky, var = estimatesky(fitsfile, badpixelmask, ccdi, gain)
    # Propagate uncertainties
    var += rnmap
    var += img
    unc = np.sqrt(np.abs(var))
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(), fits.ImageHDU(),
                         fits.ImageHDU()])
    hdul[1].data = img
    hdul[1].header = hdr
    hdul[1].header['SKYMAX'] = np.max(sky)
    hdul[2].data = sky
    hdul[3].data = unc
    hdul.writeto(outf, overwrite=True)
    hdul.close()
    logf = config.reduce / "sky_log.ecsv"
    logger.log(sky, fitsfile.name, ccdi, '', logname=logf)
    return

def skysubtract(config, fitsfiles, obsnum, logger, overwrite=False):
    """Processes a group of fits files by sky subtracting them and
    writing out the resuling data to the reduced data directory
    specified by the config object. Creates all the relevant directories
    and then calls skysubtractchip on every ccd of every fits file in
    the group.

    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    fitsfiles (list of str) - the path to the fits files which are
        going to be reduced and thus potentially need a new reduced
        data subdirectory
    obsnum (str) - the observation number, formatted as obs### where ###
        is the zero-padded, zero-indexed number in the sequence
    logger (clusterprime.Logger) - the logger to store basic info in
    overwrite (bool, optional) - if True quietly clobbers any files that
        are already in the directory, default is False which warns the
        user that no action was taken
    """
    # Setup the reduced data directory structure
    fitsdirs = makedirectories(config, fitsfiles, obsnum)
    # Then we write all those primary headers out
    writeprimefits(fitsfiles, fitsdirs, overwrite=overwrite)
    # Do the sky subtraction for every chip
    for fitsf, fitsd in zip(fitsfiles, fitsdirs):
        print(f"Sky subtracting for file {fitsf.name}")
        for ccdi in range(1, 41):
            skysubtractchip(config, fitsf, fitsd, ccdi, logger,
                            overwrite=overwrite)
    return