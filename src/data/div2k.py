import os
from data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        ### [y]
        print("[y] run DIV2K's __init__()")
        
        data_range = [r.split('-') for r in args.data_range.split('/')]
        ### [y]
        print("[y] ori data_range = {0}".format(data_range))
        
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        ### [y]
        print("[y] now data_range = {0}".format(data_range))

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        
        ###[y]
        print("[y] DIV2K, self.begin={0}".format(self.begin))
        print("[y] DIV2K, self.end={0}".format(self.end))
        
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        
        ### [y]
        print("[y] end DIV2K's __init__()")
        

    def _scan(self):
        ### [y]
        print("[y] start DIV2K's _scan()")
        
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        
        ### [y]
        print("[y] len of names_hr={0}".format(len(names_hr)))
        print("[y] len of names_lr={0}".format(len(names_lr)))
        
        ### [y]
        print("[y] end DIV2K's _scan()")

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        ### [y]
        print("[y] now in DIV2K's _set_filesystem()")
        
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        
        ### [y]
        print("[y] self.apath={0}".format(self.apath))
        print("[y] self.dir_hr={0}".format(self.dir_hr))
        print("[y] self.dir_lr={0}".format(self.dir_lr))
        ### [y]
        print("[y] end DIV2K's _set_filesystem()")

