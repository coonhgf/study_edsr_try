import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data


### [y]
from utility import log_initialize
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut
import matplotlib.pyplot as plt
import time
import datetime



def apply_lung_window(np_hu_img):
    set_lung_window = np.array([-1200.0, 600.0])  # [y] from hu to hu, not (window_center, window_length)
    np_lw_img = (np_hu_img-set_lung_window[0]) / (set_lung_window[1]-set_lung_window[0])
    np_lw_img[np_lw_img < 0]=0
    np_lw_img[np_lw_img > 1]=1
    np_lw_img = (np_lw_img*255).astype('uint8')
    return 0, np_lw_img


class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        ### [y]
        print("[y] run srdata's [exp3] __init__()")
        
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)
            
            ### [y]
            print("[y] srdata, path_bin={0}".format(path_bin))

        list_hr, list_lr = self._scan()
        
        ### [y]
        print("[y] srdata, len of list_hr={0}".format(len(list_hr)))
        print("[y] srdata, len of list_lr={0}".format(len(list_lr)))
        
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            ### [y] ori
            #os.makedirs(
            #    self.dir_hr.replace(self.apath, path_bin),
            #    exist_ok=True
            #)
            # =>
            tmp_dp1 = self.dir_hr.replace(self.apath, path_bin)
            os.makedirs(
                tmp_dp1,
                exist_ok=True
            )
            ### [y]
            print("[y] see, srdata-sep, tmp_dp1={0}".format(tmp_dp1))
            
            for s in self.scale:
                ### [y] ori
                #os.makedirs(
                #    os.path.join(
                #        self.dir_lr.replace(self.apath, path_bin),
                #        'X{}'.format(s)
                #    ),
                #    exist_ok=True
                #)
                # =>
                tmp_dp2 = self.dir_lr.replace(self.apath, path_bin)
                os.makedirs(
                    os.path.join(
                        tmp_dp2,
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )
                ### [y]
                print("see, srdata-sep, tmp_dp2={0}".format(tmp_dp2))
            
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True, hr0_or_lr1=0) 
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True, hr0_or_lr1=1)
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)
        
        ### [y]
        print("[y] end srdata's [exp3] __init__()")
        
    # Below functions as used to prepare images
    def _scan(self):
        ### [y]
        print("[y] start srdata's [exp3] _scan()")
        
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))
                
        ### [y]
        print("[y] end srdata's [exp3] _scan()")

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        ### [y]
        print("[y] now in srdata's [exp3] _set_filesystem()")
        
        self.apath = os.path.join(dir_data, self.name)  # [y] "../../../../Eclipse_ws_data/edsr/dataset/yh_sr_exp1"
        self.dir_hr = os.path.join(self.apath, 'HR')  # [y] will be overrided
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')  # [y] will be overrided
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.dcm', '.dcm')
        
        ### [y]
        print("[y]srdata, self.apath={0}".format(self.apath))
        print("[y]srdata, self.dir_hr={0}".format(self.dir_hr))
        print("[y]srdata, self.dir_lr={0}".format(self.dir_lr))
        ### [y]
        print("[y] end srdata's [exp3] _set_filesystem()")
        

    def _check_and_load(self, ext, img, f, verbose=True, hr0_or_lr1=0):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                ###pickle.dump(imageio.imread(img), _f)
                # =>
                # dcm_data = dcmread(img)
                # dcm_img = dcm_data.pixel_array.astype(np.float32)
                # dcm_img_clip = np.clip(dcm_img, -2048, 3071)
                # dcm_img_shift = dcm_img_clip + 2048
                # pickle.dump(dcm_img_shift, _f)
                # => should be to hu:
                dcm_data = dcmread(img)
                dcm_img = dcm_data.pixel_array.astype(np.float32)
                #the_intercept = dcm_data.RescaleIntercept
                #the_slope = dcm_data.RescaleSlope
                #dcm_img_hu = dcm_img * the_slope + the_intercept
                dcm_img_hu = apply_modality_lut(dcm_img, dcm_data)
                #dcm_img_hu_clip = np.clip(dcm_img_hu, -2048.0, 3071.0)
                ####dcm_img_shift = dcm_img_hu_clip + 2048
                
                ### [y] to lung win, save image
                # save_img_dp = "/home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/yh_debug_at_save_bin"
                # a_fn = os.path.basename(img)
                # tmp_list = os.path.splitext(a_fn)
                # only_fn = tmp_list[0]
                # print("\n\ndebug, only_fn={0}".format(only_fn))
                # if only_fn in ["1113017_038-1__0039", "2335572_o80__0027", "2376137_o94__0006", \
                #                "1113017_038-1__0039x2", "2335572_o80__0027x2", "2376137_o94__0006x2"]:
                #     print("found : {0} !!!!!".format(only_fn))
                #     if hr0_or_lr1 == 0:
                #         save_img_fp = os.path.join(save_img_dp, "{0}__{1}__hr.png".format(only_fn, "at_save_bin"))
                #     else:
                #         save_img_fp = os.path.join(save_img_dp, "{0}__{1}__lr.png".format(only_fn, "at_save_bin"))
                #     print("save_img_fp={0}".format(save_img_fp))
                #     tmpv, np_lung_win_img = apply_lung_window(dcm_img_hu)
                #     fig = plt.figure()
                #     ax = fig.add_subplot(1, 1, 1)
                #     ax.imshow(np_lung_win_img, cmap='gray')
                #     plt.savefig(save_img_fp)
                ###
                
                pickle.dump(dcm_img_hu, _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        ####print("lr.shape={0}, hr.shape={1}, fn={2}".format(lr.shape, hr.shape, filename))        
        
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        
        # ### [y] to lung win, save image
        # for dbidx, a_np_img in enumerate(pair):
        #     save_img_dp = "/home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/yh_debug"
        #     time_stmp = datetime.datetime.utcnow().strftime('%Y%m%d.%H%M%S')  # [y] UTC time
        #     save_img_fp = os.path.join(save_img_dp, "{0}__{1}.png".format(time_stmp, dbidx))
        #     print("filename={0}".format(filename))
        #     print("shape of a_np_img={0}".format(a_np_img.shape))
        #     print("save_img_fp={0}".format(save_img_fp))
        #     tmpv, np_lung_win_img = apply_lung_window(a_np_img)
        #     fig = plt.figure()
        #     ax = fig.add_subplot(1, 1, 1)
        #     ax.imshow(np_lung_win_img, cmap='gray')
        #     plt.savefig(save_img_fp)
        # ###
        # time.sleep(4)
        # print("\n\n\n")
        
        pair_t = common.np2Tensor_dicom(*pair)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            ###hr = imageio.imread(f_hr)
            ###lr = imageio.imread(f_lr)
            print("should not run into here")
            exit(-1)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        
        ####print("get_patch(), lr.shape={0}, hr.shape={1}".format(lr.shape, hr.shape))
        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

