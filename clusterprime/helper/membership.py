"""
    Helper function for selecting cluster members from datasets.
"""

from warnings import simplefilter

simplefilter("ignore", category=UserWarning)

import hdbscan
import numpy as np

from astropy.table import Table

def find_members(config, gids, probcut=None, return_probs=False):
    """Given an array (or column) of Gaia IDs, finds the rows which are
    likely to be members of a cluster. This cluster is simply the
    clusterdix, and is not necessarily a real cluster like M67. Returns
    the indices of everything to keep in gids.

    Arguments:
    config (clusterprime.Config) - the config object which contains path
        information
    gids (numpy.ndarray) - the set of Gaia IDs to check for their
        cluster membership likelihood
    probcut (float, optional) - The value to use as the minimum cluster
        membership probabilty, ranging from 0 to 1. If 0, or None (the
        default) instead of using a probability cut membership is
        determined using HDBscans's returned labels
    return_probs (bool, optional) - If True also returns the likelihood
        of cluster membership for each source. Default is False
    """
    clust = config.cluster
    gaiacat = config.current / f"catalogs/{clust}_gaia.vot"
    gaiacat = Table.read(gaiacat, format="votable")
    # Pull the kinematic information and the source ids
    catids = np.array(gaiacat["source_id"])
    pmra = np.array(gaiacat['pmra'])
    pmdec = np.array(gaiacat['pmdec'])
    para = np.array(gaiacat['parallax'])
    kinematics = np.c_[pmra, pmdec, para]
    # Filter for sources missing one of their pieces of kinematic info
    # so they don't break the clustering algo
    mask = np.sum(~np.isnan(kinematics), axis=1).astype(bool)
    kinematics = kinematics[mask,:]
    catids = catids[mask]
    # Perform clustering on the whole catalog to get better membership
    # probabilities
    clusterer = hdbscan.HDBSCAN(min_samples=200, prediction_data=True)
    clusterer.fit(kinematics)
    uniqueclusters = np.unique(clusterer.labels_)
    clustdiffs = 1.0e32*np.ones(uniqueclusters.size - 1)
    if clust == "M67":
        litkine = np.array([-10.97, -2.93, 1.13])
    elif clust == "Rup147":
        litkine = np.array([-0.97, -26.65, 3.25])
    # Find which cluster index corresponds to the best match for what I
    # am looking for (M67 or Rup 147)
    for uc in uniqueclusters:
        if uc == -1: continue
        clusterkine = kinematics[clusterer.labels_ == uc,:]
        medkine = np.median(clusterkine, axis=0)
        clustdiffs[uc] = np.sum(np.abs(litkine - medkine))

    bestmatch = np.argmin(clustdiffs)
    clusterkine = kinematics[clusterer.labels_ == bestmatch,:]
    if clust == "M67":
        print("                PMRA,   PMDec, Para")
        print("M67 Kinematics: -10.97, -2.93, 1.13")
        medkine = np.median(clusterkine, axis=0)
        print(f"Best Match    : {medkine[0]:0.2f},"
              f" {medkine[1]:0.2f}, {medkine[2]:0.2f}")
    elif clust == "Rup147":
        print("                PMRA,   PMDec, Para")
        print("Rup147 Kinematics: -0.97, -26.65, 3.25")
        medkine = np.median(clusterkine, axis=0)
        print(f"Best Match       : {medkine[0]:0.2f},"
              f" {medkine[1]:0.2f}, {medkine[2]:0.2f}")
    # Compute the membership likelihood
    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    membershipprobs = soft_clusters[:,bestmatch]

    if probcut:
        membermask = (membershipprobs >= probcut)
    else:
        membermask = (clusterer.labels_ == bestmatch)
    catids = catids[membermask]
    membershipprobs = membershipprobs[membermask]
    # Cross match the ids I have probabilities for
    matches, catmatch, gidsmatch = np.intersect1d(catids, gids, 
                                                  return_indices=True)
    if return_probs:
        return gidsmatch, membershipprobs[catmatch]
    else:
        return gidsmatch