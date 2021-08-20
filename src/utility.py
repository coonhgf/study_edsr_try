import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs



### [y]
import logging
import traceback
from pydicom import dcmread


def log_initialize(log_name, log_dp, flag_by_day=False):
    # exam log_dp exist
    if not os.path.isdir(log_dp):
        try:
            os.makedirs(log_dp, exist_ok=True)
        except:
            retm = traceback.format_exc()
            return None
    
    mlog = logging.getLogger(log_name)
    mlog.setLevel(logging.DEBUG)
    if flag_by_day:
        mlog_fh = logging.TimedRotatingFileHandler(os.path.join(log_dp, "{0}.log".format(log_name)), \
                                                   when='midnight', interval=1, backupCount=10, \
                                                   encoding='utf-8', utc=True)
        mlog_fh.suffix = "%Y%m%d"
    else:
        mlog_fh = logging.FileHandler(os.path.join(log_dp, "{0}.log".format(log_name)), 'w', 'utf-8')
    mlog_fh.setLevel(logging.DEBUG)
    mlog_ch = logging.StreamHandler()
    mlog_ch.setLevel(logging.DEBUG)
    mlog_fmat = logging.Formatter('%(asctime)s - [%(threadName)s][%(name)s][%(levelname)s] - %(funcName)s() :: %(message)s')
    mlog_fh.setFormatter(mlog_fmat)
    mlog_ch.setFormatter(mlog_fmat)
    mlog.addHandler(mlog_fh)
    mlog.addHandler(mlog_ch)
    return mlog



class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            ### [y]
            print("[y] checkpoint-init, i think is not give(def) args.load")
            
            if not args.save:  # [y] if not given save name, code will use timestamp as save name
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
            
            ### [y]
            print("[y] self.dir = {0}".format(self.dir))
        else:
            ### [y]
            print("[y] checkpoint-init, give(def) args.load")
            
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            if self.args.rgb_range == 255:
                while True:
                    if not queue.empty():
                        filename, tensor = queue.get()
                        if filename is None: break
                        imageio.imwrite(filename, tensor.numpy())
            elif self.args.rgb_range == 5119:
                while True:
                    if not queue.empty():
                        filename, tensor, ori_dcm_fp = queue.get()
                        if filename is None: break
                        #
                        # save result as dcm
                        #
                        # read original dcm for meta data
                        dcm_data = dcmread(ori_dcm_fp)
            
                        # modify seri_id, append ".yh_save_testing
                        seri_id = dcm_data.SeriesInstanceUID
                        dcm_data.SeriesInstanceUID = "{0}.{1}".format(seri_id, "exp3_testing")
                        
                        # update image data
                        np_rst = np.squeeze(tensor.numpy(), axis=2)
                        ###
                        #tmp_min_val = np.min(np_rst)
                        #if tmp_min_val < -1:
                        #    print("\n\n\n <-1 found \n\n\n")
                        ###
                        # np_rst = np_rst - 2048.0
                        # np_rst_round = np.round(np_rst, 0)
                        # np_rst_round_i16 = np_rst_round.astype(np.int16)
                        # np_rst_clip = np.clip(np_rst_round_i16, -2048, 3071)
                        # dcm_data.PixelData = np_rst_clip.tostring()
                        # print("shape of dcm_img_x2_clip={0}".format(np_rst_clip.shape))
                        # dcm_data.Rows, dcm_data.Columns = np_rst_clip.shape
                        ###
                        # np_rst = np_rst - 2048.0
                        # np_rst_round = np.round(np_rst, 0)
                        # np_rst_clip = np.clip(np_rst_round, -2048.0, 3071.0)
                        # # convert back to ori-style
                        # the_intercept = dcm_data.RescaleIntercept
                        # the_slope = dcm_data.RescaleSlope
                        # if the_slope == 0:
                        #     print("\n\n\n Error, the_slope=0 in file:{0}".format(ori_dcm_fp))
                        # np_rst_oristyle = (np_rst_clip - the_intercept) / the_slope
                        # np_rst_oristyle_i16 = np_rst_oristyle.astype(np.int16)
                        # dcm_data.PixelData = np_rst_oristyle_i16.tostring()
                        # print("shape of np_rst_oristyle_i16={0}".format(np_rst_oristyle_i16.shape))
                        # dcm_data.Rows, dcm_data.Columns = np_rst_oristyle_i16.shape
                        ### =>
                        # np_rst_round = np.round(np_rst, 0)
                        # np_rst_clip = np.clip(np_rst_round, -2048.0, 3071.0)
                        # ####np_rst_clip = np.clip(np_rst_round, -1024.0, 3071.0)
                        # # convert back to ori-style
                        # the_intercept = dcm_data.RescaleIntercept
                        # the_slope = dcm_data.RescaleSlope
                        # if the_slope == 0:
                        #     print("\n\n\n Error, the_slope=0 in file:{0}".format(ori_dcm_fp))
                        # np_rst_oristyle = (np_rst_clip - the_intercept) / the_slope
                        # np_rst_oristyle_i16 = np_rst_oristyle.astype(np.int16)
                        # ####dcm_data.PixelData = np_rst_oristyle_i16.tostring()
                        # dcm_data.PixelData = np_rst_oristyle_i16.tobytes()
                        # print("shape of np_rst_oristyle_i16={0}".format(np_rst_oristyle_i16.shape))
                        # dcm_data.Rows, dcm_data.Columns = np_rst_oristyle_i16.shape
                        # dcm_data.BitsStored = 16
                        # dcm_data.HighBit = 15
                        # #dcm_data.SmallestImagePixelValue = 0
                        # #dcm_data.LargestImagePixelValue = 4095
                        # # save
                        # dcm_data.save_as(filename)
                        ### =>
                        # np_rst_round = np.round(np_rst, 0)
                        # np_rst_clip = np.clip(np_rst_round, -2048.0, 3071.0)
                        # ####np_rst_clip = np.clip(np_rst_round, -1024.0, 3071.0)
                        # # convert back to ori-style
                        # the_intercept = dcm_data.RescaleIntercept
                        # the_slope = dcm_data.RescaleSlope
                        # if the_slope == 0:
                        #     print("\n\n\n Error, the_slope=0 in file:{0}".format(ori_dcm_fp))
                        # np_rst_oristyle = (np_rst_clip - the_intercept) / the_slope
                        # np_rst_oristyle_i16 = np_rst_oristyle.astype(np.int16)
                        # ####dcm_data.PixelData = np_rst_oristyle_i16.tostring()
                        # dcm_data.PixelData = np_rst_oristyle_i16.tobytes()
                        # print("shape of np_rst_oristyle_i16={0}".format(np_rst_oristyle_i16.shape))
                        # dcm_data.Rows, dcm_data.Columns = np_rst_oristyle_i16.shape
                        # dcm_data.BitsStored = 16
                        # dcm_data.HighBit = 15
                        # dcm_data.PixelRepresentation = 0
                        # #dcm_data.SmallestImagePixelValue = 0
                        # #dcm_data.LargestImagePixelValue = 4095
                        # ### => bef sleep, test 12bit to 16bit, should to intercept=0, slope=1
                        # np_rst_round = np.round(np_rst, 0)
                        # #np_rst_clip = np.clip(np_rst_round, -2048.0, 3071.0)
                        # np_rst_clip_i16 = np_rst_round.astype(np.int16)
                        # dcm_data.PixelData = np_rst_clip_i16.tobytes()
                        # #print("shape of np_rst_oristyle_i16={0}".format(np_rst_clip_i16.shape))
                        # dcm_data.Rows, dcm_data.Columns = np_rst_clip_i16.shape
                        # dcm_data.BitsStored = 16
                        # dcm_data.HighBit = 15
                        # dcm_data.PixelRepresentation = 1
                        # dcm_data.RescaleIntercept = 0
                        # dcm_data.RescaleSlope = 1
                        # #dcm_data.SmallestImagePixelValue = 0
                        # #dcm_data.LargestImagePixelValue = 4095
                        # #del dcm_data["SmallestImagePixelValue"]
                        # #del dcm_data["LargestImagePixelValue"]
                        ### => fix uint int issue, can work 1024
                        # dcm_img = dcm_data.pixel_array.astype(np.float32)
                        # src_max_val = np.max(dcm_img)
                        # src_min_val = np.min(dcm_img)
                        # print()
                        # print("src_max_val={0}".format(src_max_val))
                        # print("src_min_val={0}".format(src_min_val))
                        #
                        # max_val = np.max(np_rst)
                        # min_val = np.min(np_rst)
                        # print("max_val={0}".format(max_val))
                        # print("min_val={0}".format(min_val))
                        # # from hu back to ori format
                        # the_intercept = dcm_data.RescaleIntercept
                        # the_slope = dcm_data.RescaleSlope
                        # if the_slope == 0:
                        #     np_rst_back = np.zeros(np_rst.shape)
                        # else:
                        #     np_rst_back = (np_rst - the_intercept) / the_slope
                        #
                        # # see max and min again
                        # max_back_val = np.max(np_rst_back)
                        # min_back_val = np.min(np_rst_back)
                        # print("max_back_val={0}".format(max_back_val))
                        # print("min_back_val={0}".format(min_back_val))
                        #
                        # # clip protect here
                        # np_rst_clip = np.clip(np_rst_back, src_min_val, src_max_val)
                        # np_rst_clip_uint16 = np_rst_clip.astype(np.uint16)
                        # dcm_data.PixelData = np_rst_clip_uint16.tobytes()
                        # #print("shape of np_rst_oristyle_i16={0}".format(np_rst_clip_i16.shape))
                        # dcm_data.Rows, dcm_data.Columns = np_rst_clip_uint16.shape
                        # => 1024, 2048
                        #
                        #
                        #
                        dcm_img = dcm_data.pixel_array.astype(np.float32)
                        src_max_val = np.max(dcm_img)
                        src_min_val = np.min(dcm_img)
                        print()
                        print("src_max_val={0}".format(src_max_val))
                        print("src_min_val={0}".format(src_min_val))
                        
                        max_val = np.max(np_rst)
                        min_val = np.min(np_rst)
                        print("max_val={0}".format(max_val))
                        print("min_val={0}".format(min_val))
                        # from hu back to ori format
                        the_intercept = dcm_data.RescaleIntercept
                        the_slope = dcm_data.RescaleSlope
                        if the_slope == 0:
                            np_rst_back = np.zeros(np_rst.shape)
                        else:
                            np_rst_back = (np_rst - the_intercept) / the_slope
                        
                        # see max and min again
                        max_back_val = np.max(np_rst_back)
                        min_back_val = np.min(np_rst_back)
                        print("max_back_val={0}".format(max_back_val))
                        print("min_back_val={0}".format(min_back_val))
                        
                        # clip protect here
                        np_rst_clip = np.clip(np_rst_back, src_min_val, src_max_val)
                        if dcm_data.PixelRepresentation == 0:
                            np_rst_typecvt = np_rst_clip.astype(np.uint16)
                        elif dcm_data.PixelRepresentation == 1:
                            np_rst_typecvt = np_rst_clip.astype(np.int16)
                        else:
                            print("dcm_data.PixelRepresentation not valid, now is : {0}".format(dcm_data.PixelRepresentation))
                            exit(-1)
                        dcm_data.PixelData = np_rst_typecvt.tobytes()
                        #print("shape of np_rst_oristyle_i16={0}".format(np_rst_clip_i16.shape))
                        dcm_data.Rows, dcm_data.Columns = np_rst_typecvt.shape
                        
                        # save
                        dcm_data.save_as(filename)
                        
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        if self.args.rgb_range == 255:
            for _ in range(self.n_processes): self.queue.put((None, None))
            while not self.queue.empty(): time.sleep(1)
            for p in self.process: p.join()
        elif self.args.rgb_range == 5119:
            for _ in range(self.n_processes): self.queue.put((None, None, None))
            while not self.queue.empty(): time.sleep(1)
            for p in self.process: p.join() 

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
                
    def save_results_dicom(self, dataset, filename, save_list, scale, HR_dp):
        if self.args.save_results:
            ori_dcm_fp = os.path.join(HR_dp, "{0}.dcm".format(filename))
            #print("ori_dcm_fp={0}".format(ori_dcm_fp))
            
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )
            
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(5119 / self.args.rgb_range)
                #tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                tensor_cpu = normalized.short().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.dcm'.format(filename, p), tensor_cpu, ori_dcm_fp))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def quantize_dicom(img, rgb_range):
    ####pixel_range = 5119 / rgb_range
    ####return img.mul(pixel_range).clamp(0, 5119).round().div(pixel_range)
    #=>
    return img

def calc_psnr_dicom(sr, hr, scale, rgb_range=5119, dataset=None):
    if hr.nelement() == 1:
        print("hr.nelement() == 1, just return result:0")
        return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        print("calc psnr, in if part")
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        #print("calc psnr, in else part")
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    ori_code_psnr = -10 * math.log10(mse)
    
    #
    # my psnr
    #
    #yh_diff = (sr - hr)
    #yh_valid = yh_diff[..., shave:-shave, shave:-shave]
    #yh_mse = yh_valid.pow(2).mean()
    #yh_psnr = 10 * math.log10((5119**2)/yh_mse)
    
    #print("ori psnr={0}, yh_psnr={1}".format(ori_code_psnr, yh_psnr))
    return ori_code_psnr

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

