import h5py
import numpy as np
import tqdm

def prepare_submission(vol, filename, description):
    if vol.shape == (1100, 1440, 1440):
        vol = vol[240:840,:,:]
    elif vol.shape != (600, 1440, 1440):
        raise ValueError('Volume shape not (1100, 1440, 1440) or (600, 1440, 1440)')

    mult = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    out = np.zeros((300, 720, 720), dtype=np.uint8)
    out[vol[::2,::2,::2] > 0.5] = 1
    out[vol[::2,::2,1::2] > 0.5] += 2
    out[vol[::2,1::2,::2] > 0.5] += 4
    out[vol[::2,1::2,1::2] > 0.5] += 8
    out[vol[1::2,::2,::2] > 0.5] += 16
    out[vol[1::2,::2,1::2] > 0.5] += 32
    out[vol[1::2,1::2,::2] > 0.5] += 64
    out[vol[1::2,1::2,1::2] > 0.5] += 128

    f = h5py.File(filename, 'w')
    dset = f.create_dataset('vol', (300, 720, 720), dtype='u1', compression="gzip", compression_opts=9)
    dset.attrs['description'] = description
    dset[:] = out
    f.close()

def load_submission(filename):
    f = h5py.File(filename, 'r')
    vol = f['vol'][:]
    descr = f['vol'].attrs['description']
    f.close()

    out = np.zeros((600, 1440, 1440), dtype=np.uint8)
    out[::2, ::2, ::2][(vol & 1)>0] = 1
    out[::2, ::2, 1::2][((vol>>1) & 1)>0] = 1
    out[::2, 1::2, ::2][((vol>>2) & 1)>0] = 1
    out[::2, 1::2, 1::2][((vol>>3) & 1)>0] = 1
    out[1::2, ::2, ::2][((vol>>4) & 1)>0] = 1
    out[1::2, ::2, 1::2][((vol>>5) & 1)>0] = 1
    out[1::2, 1::2, ::2][((vol>>6) & 1)>0] = 1
    out[1::2, 1::2, 1::2][((vol>>7) & 1)>0] = 1
    return out, descr
