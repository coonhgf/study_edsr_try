# EDSR baseline model (x2) + JPEG augmentation
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75
# 37.209
#python main.py --model EDSR --scale 2 --patch_size 96 --save yh_edsr_baseline_x2 --reset --data_train DIV2K --data_test DIV2K

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]




# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset
# 37.209, med_exp1, dataset:yh_edsr_csh_axial(yh_sr_exp1)
#python main.py --model EDSR --data_train yh_sr_exp1 --data_test yh_sr_exp1 --scale 2 --patch_size 48 --save med_exp1_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --reset
# yh do summary
#python main.py --model EDSR --data_train DIV2K --scale 2 --patch_size 96 --save yh_see_summary --reset --data_test DIV2K
# 37.209, med_exp2, dataset:yh_edsr_csh_axial_exp2(yh_sr_exp2)
#python main.py --model EDSR --data_train yh_sr_exp2 --data_test yh_sr_exp2 --scale 2 --patch_size 48 --save med_exp2_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --rgb_range 5119 --reset
# 37.209, med_exp3, dataset:yh_edsr_csh_axial_exp3(yh_sr_exp3)
#python main.py --model EDSR --data_train yh_sr_exp3 --data_test yh_sr_exp3 --scale 2 --patch_size 48 --save med_exp3_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --rgb_range 5119 --reset
# 37.209, med_exp3, dataset:yh_edsr_csh_axial_exp3(yh_sr_exp3)
#python main.py --model EDSR --data_train yh_sr_exp3 --data_test yh_sr_exp3 --scale 2 --patch_size 96 --save med_exp3_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --rgb_range 5119 --reset
# 37.209, med_exp3, dataset:yh_edsr_csh_axial_exp3(yh_sr_exp3) with loss:MSE
#python main.py --model EDSR --data_train yh_sr_exp3 --data_test yh_sr_exp3 --scale 2 --patch_size 96 --save med_exp3_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --rgb_range 5119 --reset --decay 100 --loss 1*MSE --save_results
# 37.209, med_exp4(but ds name is exp3, for not mdf code), dataset:yh_edsr_csh_axial_exp3(yh_sr_exp3) with loss:L1
#python main.py --model EDSR --data_train yh_sr_exp3 --data_test yh_sr_exp3 --scale 2 --patch_size 96 --save med_exp3_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --rgb_range 5119 --reset --decay 80 --loss 1*L1 --save_results --epochs 300



# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]




# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble




# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results
#python main.py --data_test Demo --scale 2 --pre_train /home/v5/yh/edsr/EDSR-PyTorch/yh_doownload_model/EDSR_x2.pt --test_only --save_results
# 37.209
#python main.py --data_test Demo --scale 2 --pre_train download --test_only --save_results
# 37.209, test my trained model
#python main.py --data_test Demo --scale 2 --pre_train /home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/yh_edsr_baseline_x2/model --test_only --save_results
# 37.209, test med_exp1 val data, with auther's baseline edsr model
#python main.py --data_test Demo --scale 2 --pre_train download --test_only --save_results --dir_demo "../test_med_exp1"
# 37.209, test med_exp1 val data, with my trained baseline edsr model by DIV2K
#python main.py --data_test Demo --scale 2 --pre_train /home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/yh_edsr_baseline_x2_tr1/model/model_best.pt --test_only --save_results --dir_demo "../test_med_exp1"
# 37.209, test with med_exp1 val data, with mdf 1ch code model
#python main.py --data_test Demo --scale 2 --pre_train /home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/med_exp1_x2/model/model_best.pt --test_only --save_results --dir_demo "../test_med_exp1" --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --patch_size 48
# 37.209, test with med_exp2 val data, with hu training, and save as dicom code
#python main.py --data_test yh_sr_exp2 --scale 2 --pre_train /home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/med_exp2_x2/model/model_best.pt --test_only --save_results --dir_demo "../test_med_exp2" --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --rgb_range 5119 --patch_size 48
# 37.209, test with med_exp3 val data, with hu training, and save as dicom code
#python main.py --data_test yh_sr_exp3 --scale 2 --pre_train /home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/med_exp3_x2/model/model_best.pt --test_only --save_results --dir_demo "../test_med_exp3" --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --rgb_range 5119 --patch_size 48
#python main.py --data_test yh_sr_exp3 --scale 2 --pre_train /home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/med_exp3_x2__u_model4_no_shift2048/model/model_best.pt --test_only --save_results --dir_demo "../test_med_exp3" --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --rgb_range 5119 --patch_size 48
python main.py --data_test yh_sr_exp3 --scale 2 --pre_train /home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/med_exp3_x2/model/model_best.pt --test_only --save_results --dir_demo "../test_med_exp3" --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 1 --rgb_range 5119 --patch_size 96




# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

