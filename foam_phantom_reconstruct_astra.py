import astra
import h5py
import sys
import numpy as np
import imageio

# Open HDF5 file
f = h5py.File(sys.argv[1],'r')

# Load sinogram of middle slice
sino = f['projs'][:,f['projs'].shape[1]//2]

# Load angles
angs = f['angs'][:]

# Set up ASTRA geometry
pg = astra.create_proj_geom('parallel', 1, sino.shape[1], angs)
vg = astra.create_vol_geom(sino.shape[1])
pid = astra.create_projector('cuda', pg, vg)
w = astra.OpTomo(pid)

# Reconstruct using FBP with Ram-Lak filter
rec = w.reconstruct('FBP_CUDA', sino)

# Save result
imageio.imsave('slice_rec.tiff', rec)
