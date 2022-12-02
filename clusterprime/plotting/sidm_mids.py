import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from astropy.table import Table
from matplotlib.ticker import MultipleLocator

fonts = {'family': 'sans',
         'weight': 'normal',
         'size'  : 14}
mpl.rc('font', **fonts)

# *****************************************
# *** GENERATE TABLE OF CLUSTER MEMBERS ***
# *****************************************
"""These two lines below read in all the relevant data for doing the SIDM/MIDS
plot (Fig. 2 of the M67 paper, Dungee et al. 2022), the first of the two: read
in as tstab is the output of the generatealltab function towards the end of the
pipeline detailed in m67reduce.py, the 2nd is the standard all inclusive catalog
file that is needed to run the pipeline, the path to these files (and their names)
will be different on your system, so make sure to update these two!
"""
tstab = Table.read("sidm_mids/complete_ts.ecsv", format="ascii.ecsv")
allcat = Table.read("sidm_mids/M67_gaiaedr3_plus_ps1.csv", format="ascii.csv")

allcat= allcat[(allcat["prob_kine"] > 0.5)]

overlap, tsi, alli = np.intersect1d(np.array(tstab["gaia_id"]), np.array(allcat["source_id"]), return_indices=True)

tstab = tstab[tsi]
allcat = allcat[alli]

gids = np.array(allcat["source_id"])
gmag = np.array(allcat["i"])

idm_obs = np.empty((len(tstab), 126))
ids_obs = np.empty((len(tstab), 126))
ids_the = np.empty((len(tstab), 126))
Ninavg = np.empty((len(tstab), 126))
for obsi in range(126):
    idm_obs[:,obsi] = tstab[f"obs{obsi:03}_avg"]
    ids_obs[:,obsi] = tstab[f"obs{obsi:03}_sdv"]
    ids_the[:,obsi] = tstab[f"obs{obsi:03}_unc"]
    Ninavg[:,obsi] = tstab[f"obs{obsi:03}_Ninavg"]
    
idm_obs = ma.masked_invalid(idm_obs)
ids_obs = ma.masked_invalid(ids_obs)
ids_the = ma.masked_invalid(np.sqrt(Ninavg)*ids_the)
sidm_obs = ma.std(idm_obs, axis=1, ddof=1)
sidm_the = np.sqrt(ma.sum(np.power(ids_the, 2), axis=1)) / np.sqrt(np.sum(~ids_the.mask, axis=1))
mids_obs = ma.median(ids_obs, axis=1)
mids_the = ma.median(ids_the, axis=1)

fig = plt.figure(figsize=(12, 6), dpi=250)
ax1 = fig.add_subplot(131)
ax1.set_position([0.08,0.13,0.36,0.8])
ax0 = fig.add_subplot(132)
ax0.set_position([0.53,0.13,0.36,0.8])
cax = fig.add_subplot(133)
cax.set_position([0.90,0.13,0.02,0.8])

ax0.set_xscale("log")
ax0.set_yscale("log")
ax1.set_xscale("log")
ax1.set_yscale("log")

ax0.tick_params(which='major', length=6)
ax0.tick_params(which='minor', length=4)
ax0.tick_params(which='both', direction='in')
ax1.tick_params(which='major', length=6)
ax1.tick_params(which='minor', length=4)
ax1.tick_params(which='both', direction='in')

ax0.plot([0,1], [0,1], 'k--', zorder=3, lw=2)
ax0.plot([0,1], [0,1], 'w-', zorder=2, lw=2)
im = ax0.scatter(sidm_the, sidm_obs, c=gmag, cmap='magma_r', s=12, zorder=1)
ax0.set_ylim((0.9*np.min(sidm_obs)), 1.1*np.max(sidm_obs))
ax0.set_xlim((0.9*np.min(sidm_the), 1.1*np.max(sidm_the)))
ax0.set_ylabel("Observed Scatter in a Light Curve [mag]")
ax0.set_xlabel("Predicted Scatter in a Light Curve [mag]")
#ax0.set_aspect("equal")

ax1.plot([0,1], [0,1], 'k--', zorder=3, lw=2, label="1:1 Line")
ax1.plot([0,1], [0,1], 'w-', zorder=2, lw=2)
im = ax1.scatter(mids_the, mids_obs, c=gmag, cmap='magma_r', s=12, zorder=1)
ax1.set_xlim((0.9*np.min(mids_the), 1.1*np.max(mids_the)))
ax1.set_ylim((0.9*np.min(mids_obs), 1.1*np.max(mids_obs)))
ax1.set_ylabel("Observed Scatter in an Epoch [mag]")
ax1.set_xlabel("Predicted Scatter in an Epoch [mag]")
ax1.legend(loc="lower right")
#ax1.set_aspect("equal")
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("PS1 i [mag]")

# *** SAVE THE FILE ***
# saves it to whatever directory you are running this script from as written
plt.savefig("sidm_mids.png")