import os
import torch.multiprocessing as mp
class Config():
    
    def __init__(self, args):
        self.EVAL_EVERY_EPOCH = True
        # find num GPU
        self.N_GPU = len(args['GPUS'].split(','))
        self.GPUS = [int(gpu_id.strip()) for gpu_id in args['GPUS'].split(',')]

        # set os env var
        os.environ['CUDA_VISIBLE_DEVICES'] = args['GPUS']
        self.BATCH_SIZE = 256
        self.EVAL_BATCH_SIZE = 256
        self.RUN_MODE = args['RUN_MODE']
        self.NUM_WORKERS = 4
        self.MAX_EPOCH = 13
        self.CKPTS_PATH = 'ckpt'
        self.VERSION = args['VERSION']
        self.PROJ_DIM = 512
        self.HIDDEN_DIM = 256
        self.WORLD_SIZE = self.N_GPU
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        self.NR = 0
