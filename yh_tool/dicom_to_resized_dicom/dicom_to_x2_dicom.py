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
from pydicom.pixel_data_handlers.util import apply_modality_lut
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
    src_dcm_root_dp = "/media/sdc1/home/yh_dataset/edsr/yh_edsr_csh_axial/original/val2"
    src_dcm_folder_by_file_fp = "/media/sdc1/home/yh_dataset/edsr/tool_txt/copy_folder_by_file__210813_val2.txt"  # [y] txt檔案, 裡面每一行表示一個folder name, 有列在裡面就會copy
    dst_png_HR_root_dp = "/media/sdc1/home/yh_dataset/edsr/yh_edsr_csh_axial/original_to_resized_dicom_exp3/yh_edsr_csh_axial_exp3_val2_HR"
    dst_png_LR_X2_root_dp = "/media/sdc1/home/yh_dataset/edsr/yh_edsr_csh_axial/original_to_resized_dicom_exp3/yh_edsr_csh_axial_exp3_val2_LR_bicubic/X2"
    
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
            #print("now dcm fp : {0}".format(tmp_dcm_fp))
            
            # HR
            # just copy dicom
            dst_fn = "{0}__{1}.dcm".format(a_dcm_fd, "%04d" % sidx)
            dst_fp = os.path.join(dst_png_HR_root_dp, dst_fn)
            shutil.copyfile(tmp_dcm_fp, dst_fp)
            
            #
            # [y] debug
            #
            # a_fn = os.path.basename(dst_fn)
            # tmp_list = os.path.splitext(a_fn)
            # only_fn = tmp_list[0]
            # if only_fn not in ["1113017_038-1__0039", "2335572_o80__0027", "2376137_o94__0006", \
            #                "1113017_038-1__0039x2", "2335572_o80__0027x2", "2376137_o94__0006x2"]:
            #     continue
            
            
            #
            # LR of X2
            #
            dcm_data = dcmread(tmp_dcm_fp)
            
            # modify seri_id, append ".yh_mdf
            seri_id = dcm_data.SeriesInstanceUID
            dcm_data.SeriesInstanceUID = "{0}.{1}".format(seri_id, "ds_exp3")
            #dcm_data[0x10, 0x10].value = "{0}.{1}".format(seri_id, "yh_mdf")  # => can not work
            #dcm_data[0x28, 0x10].value = 256  # rows => can not work
            #dcm_data[0x28, 0x11].value = 256  # columns => can not work
            
            dcm_img = dcm_data.pixel_array.astype(np.float32)
            print("shape of the_dcm_img={0}".format(dcm_img.shape))
            
            #
            # [y]
            #
            hr_max_val = np.max(dcm_img)
            hr_min_val = np.min(dcm_img)
            print("hr_max_val={0}".format(hr_max_val))
            print("hr_min_val={0}".format(hr_min_val))
            
            #
            # [y] convert to lung window, save png
            #
            save_img_dp = "/media/sdc1/home/yh_dataset/edsr/yh_edsr_csh_axial/yh_debug_when_gen_x2"
            a_fn = os.path.basename(dst_fn)
            tmp_list = os.path.splitext(a_fn)
            only_fn = tmp_list[0]
            print("\n\ndebug, only_fn={0}".format(only_fn))
            #if only_fn in ["1113017_038-1__0039", "2335572_o80__0027", "2376137_o94__0006", \
            #               "1113017_038-1__0039x2", "2335572_o80__0027x2", "2376137_o94__0006x2"]:
            if only_fn in ["1157515_o40__0018", "2322964_o76__0034", "2369217_o91__0048", \
                           "1157515_o40__0018x2", "2322964_o76__0034x2", "2369217_o91__0048x2"]:
                save_img_fp = os.path.join(save_img_dp, "{0}__{1}__hr.png".format(only_fn, "gen_data"))
                dcm_img_hu = apply_modality_lut(dcm_img, dcm_data)
                tmpv, np_lung_win_img = apply_lung_window(dcm_img_hu)
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(np_lung_win_img, cmap='gray')
                plt.savefig(save_img_fp)
            
            
            # resize image
            resize_factor = [0.5, 0.5]  # ex : 512 to 256
            dcm_img_x2 = zoom(dcm_img, resize_factor, mode='nearest', order=3)
            #print("[120:128, 120:128]")
            #print("dcm_img_x2:\n{0}".format(dcm_img_x2[120:128, 120:128]))
            dcm_img_x2 = np.round(dcm_img_x2, 0)
            dcm_img_x2 = np.clip(dcm_img_x2, hr_min_val, hr_max_val)
            # use PixelRepresentation to decide convertion type
            if dcm_data.PixelRepresentation == 0:
                dcm_img_x2_typecvt = dcm_img_x2.astype(np.uint16)
            elif dcm_data.PixelRepresentation == 1:
                dcm_img_x2_typecvt = dcm_img_x2.astype(np.int16)
            else:
                print("dcm_data.PixelRepresentation not valid, now is : {0}".format(dcm_data.PixelRepresentation))
                exit(-1)
            dcm_data.PixelData = dcm_img_x2_typecvt.tobytes()
            dcm_data.Rows, dcm_data.Columns = dcm_img_x2_typecvt.shape
            
            #
            # [y]
            #
            lr_max_val = np.max(dcm_img_x2)
            lr_min_val = np.min(dcm_img_x2)
            print("lr_max_val={0}".format(lr_max_val))
            print("lr_min_val={0}".format(lr_min_val))
            
            #
            # [y] save lr to png
            #
            #if only_fn in ["1113017_038-1__0039", "2335572_o80__0027", "2376137_o94__0006", \
            #               "1113017_038-1__0039x2", "2335572_o80__0027x2", "2376137_o94__0006x2"]:
            if only_fn in ["1157515_o40__0018", "2322964_o76__0034", "2369217_o91__0048", \
                           "1157515_o40__0018x2", "2322964_o76__0034x2", "2369217_o91__0048x2"]:
                save_img_fp = os.path.join(save_img_dp, "{0}__{1}__lr.png".format(only_fn, "gen_data"))
                dcm_img_hu_x2 = apply_modality_lut(dcm_img_x2, dcm_data)
                tmpv, np_lung_win_img_x2 = apply_lung_window(dcm_img_hu_x2)
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(np_lung_win_img_x2, cmap='gray')
                plt.savefig(save_img_fp)
            
            
            # save 
            slice_fn = "{0}__{1}x2.dcm".format(a_dcm_fd, "%04d" % sidx)
            print("saved x2 filename={0}\n\n".format(slice_fn))
            slice_fp = os.path.join(dst_png_LR_X2_root_dp, slice_fn)
            dcm_data.save_as(slice_fp)
            
            
            #
            # [y] read x2 dicom after saving
            #
            #if only_fn in ["1113017_038-1__0039", "2335572_o80__0027", "2376137_o94__0006", \
            #               "1113017_038-1__0039x2", "2335572_o80__0027x2", "2376137_o94__0006x2"]:
            if only_fn in ["1157515_o40__0018", "2322964_o76__0034", "2369217_o91__0048", \
                           "1157515_o40__0018x2", "2322964_o76__0034x2", "2369217_o91__0048x2"]:
                dcm_data_rback = dcmread(slice_fp)
                dcm_img_rback = dcm_data_rback.pixel_array.astype(np.float32)
                dcm_img_hu_rback = apply_modality_lut(dcm_img_rback, dcm_data_rback)
                tmpv, np_lung_win_img_rback = apply_lung_window(dcm_img_hu_rback)
                
                #
                # [y]
                #
                lr_rback_max_val = np.max(dcm_img_rback)
                lr_rback_min_val = np.min(dcm_img_rback)
                print("lr_rback_max_val={0}".format(lr_rback_max_val))
                print("lr_rback_min_val={0}".format(lr_rback_min_val))
                
                save_img_fp = os.path.join(save_img_dp, "{0}__{1}__lr_rback.png".format(only_fn, "gen_data"))
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(np_lung_win_img_rback, cmap='gray')
                plt.savefig(save_img_fp)
            
            print("\n\n\n")
            
        
    print("convert dicom to resized dicom end")