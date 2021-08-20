import os
from data import srdata
from data import div2k

class DIV2KJPEG(div2k.DIV2K):
    def __init__(self, args, name='', train=True, benchmark=False):
        ### [y]
        print("[y] run DIV2KJPEG's __init__()")
        
        self.q_factor = int(name.replace('DIV2K-Q', ''))
        
        ### [y]
        print("[y] run DIV2KJPEG's super(DIV2KJPEG, self).--init---")
        
        super(DIV2KJPEG, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        
        ### [y]
        print("[y] end DIV2KJPEG's super(DIV2KJPEG, self).--init---")

    def _set_filesystem(self, dir_data):
        ### [y]
        print("[y] now in DIV2KJPEG's _set_filesystem()")
        
        self.apath = os.path.join(dir_data, 'DIV2K')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(
            self.apath, 'DIV2K_Q{}'.format(self.q_factor)
        )
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.jpg')
        
        ### [y]
        print("[y] self.apath={0}".format(self.apath))
        print("[y] self.dir_hr={0}".format(self.dir_hr))
        print("[y] self.dir_lr={0}".format(self.dir_lr))
        self.srdata_log.debug("[y] self.apath={0}".format(self.apath))
        self.srdata_log.debug("[y] self.dir_hr={0}".format(self.dir_hr))
        self.srdata_log.debug("[y] self.dir_lr={0}".format(self.dir_lr))
