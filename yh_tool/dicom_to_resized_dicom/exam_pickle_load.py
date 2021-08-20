#-*- coding: utf-8 -*-
import sys
import os
import numpy as np
import shutil
import traceback
import hashlib
import time
import datetime
import tarfile
import hashlib
#import SimpleITK as sitk
import pandas
from PIL import Image

###
#import pydicom
from pydicom import dcmread
from scipy.ndimage.interpolation import zoom
import shutil
import pickle
import matplotlib.pyplot as plt
import time
import datetime


def csv_mapping_get_seri_id_by_folder_name(csv_fp, folder_name):
    #
    # [y] read name mapping csv
    # format of csv : (shorter_id, seri_id, pat_id)
    #
    np_mapping = np.array(pandas.read_csv(csv_fp))
    got_seri_id = None
    
    for idx in range(np_mapping.shape[0]):
        # [y] if each row starting of #, skip this line
        the_row_string_col0 = str(np_mapping[idx][0])
        if "#" in the_row_string_col0:
            continue
        
        if str(np_mapping[idx][2]).strip() == folder_name:
            got_seri_id = np_mapping[idx][1]
            break
        
    if got_seri_id == None:
        return -1, got_seri_id
    else:
        return 0, got_seri_id


def clear_dir(the_dp):
    for filename in os.listdir(the_dp):
        file_path = os.path.join(the_dp, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except:
            retm = traceback.format_exc()
            print(retm)
            return -1, retm
    return 0, ""


def create_dir(the_dp):
    try:
        os.makedirs(the_dp, exist_ok=True)
    except:
        retm = traceback.format_exc()
        print(retm)
        return -1, retm
    return 0, ""


def apply_lung_window(np_hu_img):
    set_lung_window = np.array([-1200.0, 600.0])  # [y] from hu to hu, not (window_center, window_length)
    np_lw_img = (np_hu_img-set_lung_window[0]) / (set_lung_window[1]-set_lung_window[0])
    np_lw_img[np_lw_img < 0]=0
    np_lw_img[np_lw_img > 1]=1
    np_lw_img = (np_lw_img*255).astype('uint8')
    return 0, np_lw_img


if __name__ == '__main__':
    print("convert dicom to resized dicom start!")
    #
    # read dicom by seri_id(will need csv mapping file), 
    # => save each slice dicom to png, filename will be folder_name__4digits_number.png, as HighResolution
    # => resize X2(i.e. 512 to 256), as LowResolution X2
    #
    
    #
    # setting, usually fix
    #
    
    #
    # setting, usually modified
    #
    src_hr_pickle_dp = "/home/v5/yh/Eclipse_ws_data/edsr/dataset/yh_sr_exp3/bin/yh_edsr_csh_axial_exp3_train_HR"
    src_lr_pickle_dp = "/home/v5/yh/Eclipse_ws_data/edsr/dataset/yh_sr_exp3/bin/yh_edsr_csh_axial_exp3_train_LR_bicubic/X2"
    list_pickle_part_fn = ["1113017_038-1__0039", "2335572_o80__0027", "2376137_o94__0006"]
    hr_rest_fn = ".pt"
    lr_rest_fn = "x2.pt"
    save_load_pickle_dp = "/home/v5/yh/Eclipse_ws_data/edsr/dataset/yh_sr_exp3/bin/debug"
    
    #
    # auto set
    #
    
    
    #
    # checking
    #
    
    
    #
    # main process
    #
    for a_fn in list_pickle_part_fn:
        hr_fp = os.path.join(src_hr_pickle_dp, "{0}{1}".format(a_fn, hr_rest_fn))
        with open(hr_fp, 'rb') as _f:
            np_hr = pickle.load(_f)
            print("shape of np_hr={0}".format(np_hr.shape))
            
        lr_fp = os.path.join(src_lr_pickle_dp, "{0}{1}".format(a_fn, lr_rest_fn))
        with open(lr_fp, 'rb') as _f:
            np_lr = pickle.load(_f)
            print("shape of np_lr={0}".format(np_lr.shape))
            
        ### [y] to lung win, save image
        save_img_dp = save_load_pickle_dp
        time_stmp = datetime.datetime.utcnow().strftime('%Y%m%d.%H%M%S')  # [y] UTC time
        save_img_fp = os.path.join(save_img_dp, "{0}__{1}.png".format(time_stmp, 0))
        tmpv, np_lw_hr = apply_lung_window(np_hr)
        print("type of np_lw_hr={0}".format(type(np_lw_hr[0][0])))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(np_lw_hr, cmap='gray')
        plt.savefig(save_img_fp)
        #
        save_img_fp = os.path.join(save_img_dp, "{0}__{1}.png".format(time_stmp, 1))
        tmpv, np_lw_lr = apply_lung_window(np_lr)
        print("type of np_lw_lr={0}".format(type(np_lw_lr[0][0])))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(np_lw_lr, cmap='gray')
        plt.savefig(save_img_fp)
        ###
        
        time.sleep(2)
    
    
    print("convert dicom to resized dicom end")