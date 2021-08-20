import os
from data import srdata
import glob

class yh_sr_exp1(srdata.SRData):
    def __init__(self, args, name='yh_sr_exp1', train=True, benchmark=False):
        ### [y]
        print("[y] run yh_sr_exp1's __init__()")
        
        self.train = train
        
        super(yh_sr_exp1, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        
        ### [y]
        print("[y] end yh_sr_exp1's __init__()")
        

    def _scan(self):
        ### [y]
        print("[y] start yh_sr_exp1's _scan()")
        
        names_hr = glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        names_hr.sort()
        print("len of names_hr={0}".format(len(names_hr)))
        
        names_lr = [[] for _ in self.scale]
        for hr_fp in names_hr:
            hr_fn = os.path.basename(hr_fp)
            hr_fn_no_ext = os.path.splitext(hr_fn)[0]
            for si, s in enumerate(self.scale):
                lr_fn = "{0}x{1}{2}".format(hr_fn_no_ext, s, self.ext[1])
                lr_fp = os.path.join(self.dir_lr, "{0}".format("X2"), lr_fn)
                names_lr[si].append(lr_fp)
        print("len of names_lr={0}".format(len(names_lr)))
        print("len of names_lr[0]={0}".format(len(names_lr[0])))
        
        ### [y]
        print("[y] end yh_sr_exp1's _scan()")

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        ### [y]
        print("[y] now in yh_sr_exp1's _set_filesystem()")
        
        if self.train:
            super(yh_sr_exp1, self)._set_filesystem(dir_data)
            self.dir_hr = os.path.join(self.apath, 'yh_edsr_csh_axial_train_HR')
            self.dir_lr = os.path.join(self.apath, 'yh_edsr_csh_axial_train_LR_bicubic')
            if self.input_large: self.dir_lr += 'L'
        else:
            super(yh_sr_exp1, self)._set_filesystem(dir_data)
            self.dir_hr = os.path.join(self.apath, 'yh_edsr_csh_axial_val_HR')
            self.dir_lr = os.path.join(self.apath, 'yh_edsr_csh_axial_val_LR_bicubic')
            if self.input_large: self.dir_lr += 'L'
        
        ### [y]
        print("[y] self.apath={0}".format(self.apath))
        print("[y] self.dir_hr={0}".format(self.dir_hr))
        print("[y] self.dir_lr={0}".format(self.dir_lr))
        ### [y]
        print("[y] end yh_sr_exp1's _set_filesystem()")

