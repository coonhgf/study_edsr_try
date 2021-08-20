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



def simple_copytree(src, dst, symlinks=False, ignore=None):
    try:
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)
    except:
        retm = traceback.format_exc()
        print(retm)
        return -1, retm
    return 0, ""

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


if __name__ == '__main__':
    print("copy folder start!")
    
    #
    # setting, usually fix
    #
    
    #
    # setting, usually modified
    #
    src_root_dp = "/media/sdc1/Linda/Dataset/CSH_Chest/5mm/dicom"
    src_folder_by_file_fp = "/media/sdc1/home/yh_dataset/edsr/tool_txt/copy_folder_by_file__210813_val2.txt"  # [y] txt檔案, 裡面每一行表示一個folder name, 有列在裡面就會copy
    dst_root_dp = "/media/sdc1/home/yh_dataset/edsr/yh_edsr_csh_axial/original/val2"
    
    
    #
    # auto set
    #
    list_src_folder = []
    with open(src_folder_by_file_fp, "r") as infile:
        for a_line in infile:
            content = a_line.strip()
            list_src_folder.append(content)
            
            
    #
    # checking
    #
    # check each folder exist
    not_exist_folder = []
    for a_folder in list_src_folder:
        tmp_dp = os.path.join(src_root_dp, a_folder)
        if not os.path.isdir(tmp_dp):
            not_exist_folder.append(a_folder)
    if len(not_exist_folder) >= 1:
        print("src folder not exist:{0}".format(not_exist_folder))
        exit(1)
    
           
    #
    # main process
    #
    # create destination dir
    if os.path.isdir(dst_root_dp):
        retv, retm = clear_dir(dst_root_dp)
        if retv != 0:
            exit(-1)
    else:
        retv, retm = create_dir(dst_root_dp)
        if retv != 0:
            exit(-1)
    print("clean or create destination folder : OK")
    
    
    # copy folder
    for a_src_folder in list_src_folder:
        tmp_src_dp = os.path.join(src_root_dp, a_src_folder)
        tmp_dst_dp = os.path.join(dst_root_dp, a_src_folder)
        
        retv, retm = create_dir(tmp_dst_dp)
        if retv != 0:
            print("fail at copy {0} to dst".format(a_src_folder))
            exit(-1)
        
        retv, retm = simple_copytree(tmp_src_dp, tmp_dst_dp)
        if retv != 0:
            print("fail at simple_copytree with {0}".format(a_src_folder))
            exit(-1)
    
    
    print("copy folder end")