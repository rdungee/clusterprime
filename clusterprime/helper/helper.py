"""
    A collection of helper functions that are frequently used through
    out the reduction process
"""
from os.path import isfile
from warnings import warn

def overwritecheck(fname):
    """Function that performs a simple isfile check, if it finds a file
    then it returns a warning to the user to let them know that file
    was skipped in processing!

    Arguments:
    fname (string) - the path of the file to check
    """
    doesexist = isfile(fname)
    if doesexist:
        warn(f"File {fname} exists already, you are seeing this because"
              " overwrite is false and so this step is being skipped for this"
              " file!")
    return doesexist

def trimchip(img):
    """Trims an image of the overscan regions, these have been thus far
    (August 5 2020) seen to be constant, so the numbers are hardcoded
    here as opposed to taken from the headers.

    Arguments:
    img (numpy.ndarray) - the image to trim
    """
    return img[1:4611, 32:2080]
