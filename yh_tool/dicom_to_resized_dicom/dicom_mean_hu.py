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
from collections import Counter
from pydicom.pixel_data_handlers.util import apply_modality_lut


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
    print("calc hu mean start!")
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
    src_dcm_root_dp = "/media/sdc1/home/yh_dataset/edsr/yh_edsr_csh_axial/original/train"
    src_dcm_folder_by_file_fp = "/media/sdc1/home/yh_dataset/edsr/tool_txt/copy_folder_by_file__210707_train.txt"  # [y] txt檔案, 裡面每一行表示一個folder name
    
    #
    # auto set
    #
    list_src_dcm_folder = []
    with open(src_dcm_folder_by_file_fp, "r") as infile:
        for a_line in infile:
            content = a_line.strip()
            list_src_dcm_folder.append(content)
            
            
    #
    # checking
    #
    # check each folder exist
    not_exist_folder = []
    for a_folder in list_src_dcm_folder:
        tmp_dp = os.path.join(src_dcm_root_dp, a_folder)
        if not os.path.isdir(tmp_dp):
            not_exist_folder.append(a_folder)
    if len(not_exist_folder) >= 1:
        print("src folder not exist:{0}".format(not_exist_folder))
        exit(1)
    
           
    #
    # main process
    #
    
    # read each dicom folder's dicom and convert to png
    # png naming is FolderName___001.png etc.
    list_all_mean = []  # all mean
    list_all_hu_max = []
    list_all_hu_min = []
    for a_dcm_fd in list_src_dcm_folder:
        print("processing : {0}".format(a_dcm_fd))
        tmp_src_dp = os.path.join(src_dcm_root_dp, a_dcm_fd)
        
        # list files in this folder
        list_filename = []
        for tmp_fn in os.listdir(tmp_src_dp):
            tmp = os.path.join(tmp_src_dp, tmp_fn)
            if os.path.isfile(tmp):
                list_filename.append(tmp_fn)
        list_filename.sort()
        
        # process
        list_scan_hu_mean = []
        scan_hu_max = -1000
        scan_hu_min = 1000
        for sidx, tmp_dcm_fn in enumerate(list_filename):
            tmp_dcm_fp = os.path.join(tmp_src_dp, tmp_dcm_fn)
            #print("now dcm fp : {0}".format(tmp_dcm_fp))
            
            # HR
            # read hu and calc mean
            dcm_data = dcmread(tmp_dcm_fp)
            dcm_img = dcm_data.pixel_array.astype(np.float64)
            
            #
            #
            #
            if sidx == 0:
                print("dcm_data.BitsAllocated={0}".format(dcm_data.BitsAllocated))
                print("dcm_data.BitsStored={0}".format(dcm_data.BitsStored))
                print("dcm_data.HighBit={0}".format(dcm_data.HighBit))
                print("dcm_data.WindowCenter={0}".format(dcm_data.WindowCenter))
                print("dcm_data.WindowWidth={0}".format(dcm_data.WindowWidth))
                print("dcm_data.RescaleIntercept={0}".format(dcm_data.RescaleIntercept))
                print("dcm_data.RescaleSlope={0}".format(dcm_data.RescaleSlope))
                print("dcm_data.PixelRepresentation={0}".format(dcm_data.PixelRepresentation))
                # print("============")
                # win_c = dcm_data.WindowCenter
                # win_w = dcm_data.WindowWidth
                # win_min = win_c - (win_w/2)
                # win_max = win_c + (win_w/2)
                # print("win_min={0}".format(win_min))
                # print("win_max={0}".format(win_max))
                # print("============")
            
            #
            # convert to HU value
            #
            #the_intercept = dcm_data.RescaleIntercept
            #the_slope = dcm_data.RescaleSlope
            #dcm_img_hu = dcm_img * the_slope + the_intercept
            #=>
            dcm_img_hu = apply_modality_lut(dcm_img, dcm_data)
            #print("type of dcm_img_hu:{0}".format(type(dcm_img_hu)))
            
            # calc mean of this slice
            a_mean_of_slice = np.mean(dcm_img_hu)
            list_scan_hu_mean.append(a_mean_of_slice)
            tmp_max_val = np.max(dcm_img_hu)
            tmp_min_val = np.min(dcm_img_hu)
            if tmp_max_val > scan_hu_max:
                scan_hu_max = tmp_max_val
            if tmp_min_val < scan_hu_min:
                scan_hu_min = tmp_min_val
            
        # calc and save the mean of this scan
        a_mean_of_scan = sum(list_scan_hu_mean)/len(list_scan_hu_mean)
        list_all_mean.append(a_mean_of_scan)
        
        # rec max and min hu of this scan
        list_all_hu_max.append(scan_hu_max)
        list_all_hu_min.append(scan_hu_min)
        print("scan_hu_max={0}".format(scan_hu_max))
        print("scan_hu_min={0}".format(scan_hu_min))
        print()
    
    
    #
    # calc all scan's mean and show
    #
    a_mean_of_all = sum(list_all_mean)/len(list_all_mean)
    print("mean_per_scan={0}".format(list_all_mean))
    print("len of mean_per_scan={0}".format(len(list_all_mean)))
    print("a_mean_of_all={0}".format(a_mean_of_all))
    print("")

    
    # hu value ana
    dict_hu_max_counter = Counter(list_all_hu_max)
    dict_hu_min_counter = Counter(list_all_hu_min)
    cvt_sorted_list_hu_max = sorted(dict_hu_max_counter.items(), key=lambda x:x[1], reverse=True)
    cvt_sorted_list_hu_min = sorted(dict_hu_min_counter.items(), key=lambda x:x[1], reverse=True)
    print("top [all] of cvt_sorted_list_hu_max=\n{0}".format(cvt_sorted_list_hu_max))
    print("top 10 of cvt_sorted_list_hu_min=\n{0}".format(cvt_sorted_list_hu_min[0:10]))
    print("")
    
    
    print("calc hu mean end")