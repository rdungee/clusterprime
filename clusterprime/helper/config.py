"""
    A configuration object to hold path information, so that I can stop
    storing that info at the top of every file
"""
from warnings import warn

from pathlib import Path

class Config:
    """Object that holds the paths to the relevant directories,
    specifically the base, data, and reduce directories

    Arguments:
    currdir (str) - the path to the current directory from which the
        code is being run.
    cluster (str) - the cluster for which to run the analysis, options
        are 'M67' and 'Rup147' note these are case sensitive (I think)
    analysis (str) - the name for the experiment, such as an aperture
        size, or masking parts of a periodogram. The name should be
        unique as it will go inside redudir
    apsize (float) - the aperture size to be used
    basedir (str, optional) - the "base directory," meaning the
        directory that ultimately contains the current working directory
        and the data directory. If None then this is constructed from
        the current directory assuming a particular layout
    datadir (str, optional) - the "data directory," meaning the
        directory that contains the raw data. If None then this is
        constructed from the current directory assuming a particular
        layout
    redudir (str, optional) - the "reduced data directory," meaning the
        directory that contains the reduced data. If None then this is
        constructed from the current directory assuming a particular
        layout
    test (bool, optional) - if True uses "testdata" instead of "data" in
        the assumed directory layouts, default is False
    """
    
    def __init__(self, currdir, cluster, analysis, apsize, basedir=None,
                 datadir=None, redudir=None, test=False):
        if test: d = 'testdata'
        else: d = 'data'
        # Current directory, str or path object
        if type(currdir) is str:
            self.current = Path(currdir)
        else:
            self.current = currdir
        # Assume base directory is 1 level higher
        if basedir is None:
            self.base = self.current.parent
        elif type(basedir) is str:
            self.base = Path(basedir)
        else:
            self.base = basedir
        # Datadir is assumed to be inside data and M67
        if datadir is None:
            self.data = self.base / d / cluster
        elif type(datadir) is str:
            self.data = Path(datadir)
        else:
            self.data = datadir
        # redudir is assumed to be inside data and M67
        if redudir is None:
            self.reduce = self.base / d / f'{cluster}reduce'
        elif type(redudir) is str:
            self.reduce = Path(redudir)
        else:
            self.reduce = redudir
        # experiment creates a subdirectory in redudir containing all
        # the post sky subtraction information, such as source lists
        # and photometry tables, things that can change frequently!
        # esp. compared to the sky subtraction
        self.analysis = self.reduce / analysis
        self.analysis.mkdir(exist_ok=True)

        if (cluster is not "M67") or (cluster is not "Rup147"):
            warn(f"cluster {cluster} is not one of the standard forms. Please" 
                 " double check your config instantiation.")
        self.cluster = cluster
        
        self.apsize = apsize