"""
    A simple logging object for tracking basic information from the sky
    subtraction process.
"""

import numpy as np

from astropy.table import Table

class Logger:
    """A wrapper to simplify creating an astropy table that serves as
    the log. The table columns' names and dtypes are preset by this
    object. Takes no arguments to set up.
    """

    def __init__(self):
        self.tab = Table(names=("Minimum", "Maximum", "Median",
                                "Standard Deviation", "Filename", "CCD Index",
                                "Flag"),
                         dtype=("f8", "f8", "f8", "f8", "S16", "i4", "S32"))
        return

    def log(self, ccdchip, fitsname, ccdi, flag, logname="log.ecsv"):
        """Adds a row to the table, specifically it calculates a few
        stats and adds them to the table. Saves the file as a .ecsv
        which means it ***will overwrite any currently exisiting log***
        which by default should be a simple update from the previous
        call

        Arguments:
        ccdchip (numpy.ndarray) - the image itself as a numpy array
        fitsname (string) - the filename the ccdchip was pulled from
        ccdi (int) - the index of the ccd in the fits file indicated by
            fitsname
        flag (string) - a flag to be added to the row, can be any string
            limited to 32 characters in length.
        logname (string, optional) - the file name to write the log to,
            default is 'log.ecsv'
        """
        self.tab.add_row((np.min(ccdchip), np.max(ccdchip), np.median(ccdchip),
                          np.std(ccdchip), fitsname, ccdi, flag))
        self.tab.write(logname, format="ascii.ecsv", overwrite=True)
        return
    
    def updatemeta(self, key, value):
        """A simple method to update the meta data for the table stored
        in the object

        Arguments:
        key (string) - the key to add or modify in the table's metadata
        value (anything) - the value to associate with that key
        """
        self.tab.meta[key] = value
        return