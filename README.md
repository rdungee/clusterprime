# clusterprime
A repo for the data pipeline that takes CFHT/MegaPrime data and produces light curves.

Set this up by cloning into a directory, and then in that directory run 'pip install ./ -e'

Only tested in python 3.6.8, and the package versions are specified in the setup script so definitely only do this in a python virtual environment made for specifically this!!

There are a few supplementary filess needed for the pipeline, they can be found [here](https://www.dropbox.com/sh/6oqx2xka8dky9xe/AACaZWgwHfW8mCkZEv_JX7b6a?dl=0). Note that the bad pixel mask is needed by the pipeline no matter what, and the M67 catalog is meant
to serve as an example of the formatting needed for the catalog file required by the pipeline (details are also laid out in the M67reduce.py example file).