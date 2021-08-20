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
    src_dcm_root_dp = "/media/sdc1/home/yh_dataset/edsr/yh_edsr_csh_axial/original/train"
    src_dcm_folder_by_file_fp = "/media/sdc1/home/yh_dataset/edsr/tool_txt/copy_folder_by_file__210707_yh_ana.txt"  # [y] txt檔案, 裡面每一行表示一個folder name, 有列在裡面就會copy
    dst_png_HR_root_dp = "/media/sdc1/home/yh_dataset/edsr/yh_edsr_csh_axial/original_to_resized_dicom/bit12_test"
    dst_png_LR_X2_root_dp = "/media/sdc1/home/yh_dataset/edsr/yh_edsr_csh_axial/original_to_resized_dicom/bit12_test/X2"
    
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
    # create destination dir
    list_check_dst_dp = [dst_png_HR_root_dp, dst_png_LR_X2_root_dp]
    for a_dp in list_check_dst_dp:
        if os.path.isdir(a_dp):
            retv, retm = clear_dir(a_dp)
            if retv != 0:
                exit(-1)
        else:
            retv, retm = create_dir(a_dp)
            if retv != 0:
                exit(-1)
    print("clean or create destination folder : OK")
    
    
    # read each dicom folder's dicom and convert to png
    # png naming is FolderName___001.png etc.
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
        for sidx, tmp_dcm_fn in enumerate(list_filename):
            tmp_dcm_fp = os.path.join(tmp_src_dp, tmp_dcm_fn)
            print("now dcm fp : {0}".format(tmp_dcm_fp))
            
            # HR
            # just copy dicom
            dst_fn = "{0}__{1}.dcm".format(a_dcm_fd, "%04d" % sidx)
            dst_fp = os.path.join(dst_png_HR_root_dp, dst_fn)
            shutil.copyfile(tmp_dcm_fp, dst_fp)
            
            #
            # LR of X2
            #
            dcm_data = dcmread(tmp_dcm_fp)
            
            # modify seri_id, append ".yh_mdf
            seri_id = dcm_data.SeriesInstanceUID
            dcm_data.SeriesInstanceUID = "{0}.{1}".format(seri_id, "yh_mdf")
            #dcm_data[0x10, 0x10].value = "{0}.{1}".format(seri_id, "yh_mdf")  # => can not work
            #dcm_data[0x28, 0x10].value = 256  # rows => can not work
            #dcm_data[0x28, 0x11].value = 256  # columns => can not work
            
            dcm_img = dcm_data.pixel_array.astype(np.float32)
            print("shape of the_dcm_img={0}".format(dcm_img.shape))
            
            # clip
            the_intercept = dcm_data.RescaleIntercept
            the_slope = dcm_data.RescaleSlope
            dcm_img_hu = dcm_img * the_slope + the_intercept
            dcm_img_hu_clip = np.clip(dcm_img_hu, -2048, 3071)
            # back to its value by using intercept and slope
            if the_slope == 0:
                print("error slope is 0, quit now")
                exit(-1)
            dcm_rst = (dcm_img_hu_clip - the_intercept) / the_slope
            dcm_rst_i16 = dcm_rst.astype(np.int16)
            dcm_data.PixelData = dcm_rst_i16.tostring()
            print("shape of dcm_rst_i16={0}".format(dcm_rst_i16.shape))
            dcm_data.Rows, dcm_data.Columns = dcm_rst_i16.shape
            
            # save 
            slice_fn = "{0}__{1}x2.dcm".format(a_dcm_fd, "%04d" % sidx)
            print("saved x2 filename={0}\n\n".format(slice_fn))
            slice_fp = os.path.join(dst_png_LR_X2_root_dp, slice_fn)
            dcm_data.save_as(slice_fp)
        
    print("convert dicom to resized dicom end")