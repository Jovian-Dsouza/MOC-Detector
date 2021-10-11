import os

class Opts:
    def __init__(self) -> None:
        self.split = 1
        self.K = 7
        self.ninput = 1
        self.resize_height = 288
        self.resize_width = 288

        self.root_dir  = os.path.join(os.path.dirname(__file__), '..')
        self.down_ratio = 4
        self.rgb_model = '../checkpoints/ucf_dla34_K7_rgb_coco.pth'
        self.flow_model = ''
        self.inference_dir = 'data'

        self.flip_test = None
        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]

        # system settings
        self.gpus = '0'
        self.num_workers = 0
        self.batch_size = 1
        self.master_batch_size = -1

        self.pin_memory = True

        self.num_classes = 24
        self.arch = 'dla_34'
        self.branch_info = {'hm': self.num_classes,
                            'mov': 2 * self.K,
                            'wh': 2 * self.K}
        self.head_conv = 256

        self.print_log = False
    
    def update_dataset(self, opt, dataset):
        opt.num_classes = dataset.num_classes
        opt.branch_info = {'hm': opt.num_classes,
                           'mov': 2 * opt.K,
                           'wh': 2 * opt.K}
        return opt