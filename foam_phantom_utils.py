import h5py
import numpy as np
import tqdm
import foam_ct_phantom

def prepare_submission(vol, filename, description):
    if vol.shape != (1080, 1280, 1280):
        raise ValueError('Volume shape not (1080, 1280, 1280)')

    mult = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    out = np.zeros((540, 640, 640), dtype=np.uint8)
    out[vol[::2,::2,::2] > 0.5] = 1
    out[vol[::2,::2,1::2] > 0.5] += 2
    out[vol[::2,1::2,::2] > 0.5] += 4
    out[vol[::2,1::2,1::2] > 0.5] += 8
    out[vol[1::2,::2,::2] > 0.5] += 16
    out[vol[1::2,::2,1::2] > 0.5] += 32
    out[vol[1::2,1::2,::2] > 0.5] += 64
    out[vol[1::2,1::2,1::2] > 0.5] += 128

    f = h5py.File(filename, 'w')
    dset = f.create_dataset('vol', (540, 640, 640), dtype='u1', compression="gzip", compression_opts=1)
    dset.attrs['description'] = description
    for i in tqdm.trange(540):
        dset[i] = out[i]
    f.close()

def load_submission(filename):
    f = h5py.File(filename, 'r')
    vol = f['vol'][:]
    descr = f['vol'].attrs['description']
    f.close()

    out = np.zeros((1080, 1280, 1280), dtype=np.uint8)
    out[::2, ::2, ::2][(vol & 1)>0] = 1
    out[::2, ::2, 1::2][((vol>>1) & 1)>0] = 1
    out[::2, 1::2, ::2][((vol>>2) & 1)>0] = 1
    out[::2, 1::2, 1::2][((vol>>3) & 1)>0] = 1
    out[1::2, ::2, ::2][((vol>>4) & 1)>0] = 1
    out[1::2, ::2, 1::2][((vol>>5) & 1)>0] = 1
    out[1::2, 1::2, ::2][((vol>>6) & 1)>0] = 1
    out[1::2, 1::2, 1::2][((vol>>7) & 1)>0] = 1
    return out, descr


def compute_scores(ground_truth_file, vol, cx=0, cy=0):
    f = h5py.File(ground_truth_file, 'r')
    s = f['spheres'][:]
    f.close()

    # Background is labeled 0, foam is labeled 1

    # Set large voids to label 2
    s[4::5][s[3::5]>=0.1]=2
    # Set medium voids to label 3
    s[4::5][np.logical_and(s[3::5]<0.1,s[3::5]>=0.025)]=3
    # Set small voids to label 4
    s[4::5][s[3::5]<0.025]=4

    # Generate ground truth volume
    gt = np.zeros((1080, 1280, 1280), dtype=np.float32)
    for i in tqdm.trange(1080):
        foam_ct_phantom.ccode.genvol(s, gt, 1280, 1280, 1080, 3/1280, i, cx=cx, cy=cy)
    gt = gt[:,::-1].astype(np.uint8) # Flip to make identical to ASTRA

    return metrics(vol, gt)


def metrics(vol, gt):
    # Compute true positives, false negatives, etcetera
    tp_foam = vol[gt==1].sum()
    tp_large = (vol[gt==2]==0).sum()
    tp_medium = (vol[gt==3]==0).sum()
    tp_small = (vol[gt==4]==0).sum()

    fp_foam = vol[gt>1].sum()

    fn_foam = (vol[gt==1]==0).sum()
    fn_large = vol[gt==2].sum()
    fn_medium = vol[gt==3].sum()
    fn_small = vol[gt==4].sum()

    dice_foam = 2*tp_foam/(2*tp_foam + fp_foam + fn_foam)
    sens_large = tp_large/(tp_large + fn_large)
    sens_medium = tp_medium/(tp_medium + fn_medium)
    sens_small = tp_small/(tp_small + fn_small)

    gt[gt>1] = 0

    msk = np.zeros_like(gt)
    tmp = gt[:-1]!=gt[1:]
    msk[:-1][tmp] = 1
    msk[1:][tmp] = 1
    tmp = gt[:,:-1]!=gt[:,1:]
    msk[:,:-1][tmp] = 1
    msk[:,1:][tmp] = 1
    tmp = gt[:,:,:-1]!=gt[:,:,1:]
    msk[:,:,:-1][tmp] = 1
    msk[:,:,1:][tmp] = 1

    tp_edge = 0
    fp_edge = 0
    fn_edge = 0
    for i in tqdm.trange(1080):
        edge_gt = gt[i][msk[i]>0]
        edge_vol = vol[i][msk[i]>0]

        tp_edge += edge_vol[edge_gt==1].sum()
        fp_edge += edge_vol[edge_gt==0].sum()
        fn_edge += (edge_vol[edge_gt==1]==0).sum()

    dice_edge = 2*tp_edge/(2*tp_edge + fp_edge + fn_edge)

    scores = np.array([dice_foam, sens_large, sens_medium, sens_small, dice_edge])
    harm_mean = scores.size/np.sum(1/scores)

    return harm_mean, dice_foam, sens_large, sens_medium, sens_small, dice_edge
    


def compute_scores_dyn(ground_truth_file, vol, cx=0, cy=0):
    f = h5py.File(ground_truth_file, 'r')
    s = f['spheres'][:]
    sz = f['sizes'][:]
    f.close()

    # Background is labeled 0, foam is labeled 1

    # Set large voids to label 2
    s[4::5][s[3::5]>=0.1]=2
    # Set medium voids to label 3
    s[4::5][np.logical_and(s[3::5]<0.1,s[3::5]>=0.025)]=3
    # Set small voids to label 4
    s[4::5][s[3::5]<0.025]=4

    ftmp = h5py.File('tmpchallenge.h5', 'w')
    ftmp['spheres'] = s
    ftmp['sizes'] = sz
    ftmp.close()
    
    # Evaluate scores for sample at the time of projection 2688
    time = np.linspace(0, 1, 3072)[2688]

    # Generate ground truth volume
    vol_geom = foam_ct_phantom.VolumeGeometry(1280,1280,1080,3/1280)
    foam_ct_phantom.expand.genvol_expand(time, 'tmpchallengevol.h5', 'tmpchallenge.h5', vol_geom)
    ftmp = h5py.File('tmpchallengevol.h5','r')
    gt = ftmp['volume'][:]
    ftmp.close()
    gt = gt[:,::-1].astype(np.uint8) # Flip to make identical to ASTRA
    
    return metrics(vol, gt)
