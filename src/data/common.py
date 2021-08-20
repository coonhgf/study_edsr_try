import random

import numpy as np
import skimage.color as sc

import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1  # [y] 1    [y] 1
        tp = p * patch_size  # [y] 1*48    [y] 1*96(baseline)    target.patch
        ip = tp // scale  # [y] 48//2    [y] 96//2(baseline)    input.patch
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)  # [y] input.x(low reso)
    iy = random.randrange(0, ih - ip + 1)  # [y] input.y(low reso)

    if not input_large:
        tx, ty = scale * ix, scale * iy  # [y] target.x, target.y(high reso)
    else:
        tx, ty = ix, iy

    if len(args[0].shape) == 3:
        # [y] (y, x, c)
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]
    else:
        # [y] only (y, x)
        ret = [
            args[0][iy:iy + ip, ix:ix + ip],
            *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
        ]
        

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            ####print("img.ndim == 2, now, img.shape={0}".format(img.shape))

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        ####print("at end set_channel(), img.shape={0}".format(img.shape))
        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        ####print("tensor={0}".format(tensor.size()))
        return tensor

    return [_np2Tensor(a) for a in args]

def np2Tensor_dicom(*args, rgb_range=5119):
    def _np2Tensor(img):
        # [y] shift from -2048~3071 to 0~5119, [move this shift to data's common py]
        #img_shift = img + 2048
        
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        ####tensor.mul_(rgb_range / 5119)

        ####print("tensor={0}".format(tensor.size()))
        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img
    
    def _augment_only_yx(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        if rot90: img = img.transpose(1, 0)    
        return img
    
    if len(args[0].shape) == 3:
        return [_augment(a) for a in args]
    else:
        return [_augment_only_yx(a) for a in args]

