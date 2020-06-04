from dataloader import DatasetLoader
from train import train_engine
import argparse, yaml
from config import Config
import torch.distributed as dist
import copy
import torch.multiprocessing as mp
# write a function to parse the arguments and conifg
def parse_args():
    
    parser = argparse.ArgumentParser(description='Visual Relationship Classifier')
    
    parser.add_argument('--GPUS', dest='GPUS',
                        help="GPUS eg. 0,1,2", type=str)
    parser.add_argument('--VERSION', dest='VERSION',
                        help="version to describe the model",
                        type=str)
    parser.add_argument('--RUN_MODE', dest='RUN_MODE',
                        help='train or val', type=str,
                        default='train')
    args = parser.parse_args()
    
    return args

def main():
    args = vars(parse_args())
    
    # create an object of class Config
    print('Loaded Configs')
    config = Config(args)

    # multiprocessing
    mp.spawn(train, nprocs = config.N_GPU, args = (config, ))

def train(gpu, config):

    ###############################################
    rank = config.NR * config.N_GPU + gpu
    dist.init_process_group(
    	backend='nccl',
   		init_method='env://',
    	world_size=config.WORLD_SIZE,
    	rank=rank
    )
    ###############################################
    # first loading the dataset
    print('Loading the training set')
    dataset = DatasetLoader(config)

    if config.EVAL_EVERY_EPOCH:
        print('Loading the eval set')
        config_val = copy.deepcopy(config)
        config_val.RUN_MODE = 'val'
        dataset_val = DatasetLoader(config_val)

    # train the model
    print('train engine called')
    train_engine(gpu,rank, config, dataset, dataset_val)


if __name__=='__main__':
    main()
