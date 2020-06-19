# all the imports go here
#-------------------------------
# Torch Related Imports 
#-------------------------------
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

#-----------------------------
# Dataloader and Test engine
#-----------------------------
from vis_rel.data.dataloader import DatasetLoader
from vis_rel.function.val import validate
from vis_rel.modules.frcnn_classifier import Net

#------------------------------
# Other utilities imported
#------------------------------
import os
import time
import wandb
import copy
import pprint


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    pass

# the train engine here
def train_engine(args, config):

    wandb.init(project="intern", name=config.VERSION, config=config)

    # print the args and the config
    pprint.pprint(args)
    pprint.pprint(config)

    # manually set random seed
    if config.RNG_SEED > -1:
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)
        torch.cuda.manual_seed_all(config.RNG_SEED)

    # cudnn
    torch.backends.cudnn.benchmark = True
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    if args.dist:
        model = Net(config)
        local_rank = int(os.environ.get('LOCAL_RANK') or 0)
        config.GPUS = str(local_rank)
        torch.cuda.set_device(local_rank)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'] or 23456)
        world_size = int(os.environ['WORLD_SIZE'] or 1)
        rank = int(os.environ['RANK'] or 0)

        if args.slurm:
            distributed.init_process_group(backend='nccl')
        else:
            distributed.init_process_group(
                backend='nccl',
                init_method='tcp://{}:{}'.format(master_address, master_port),
                world_size=world_size,
                rank=rank,
                group_name='mtorch')

        print(f'native distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')
        config.GPUS = str(local_rank)
        model = model.cuda()
        if not config.TRAIN.FP16:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        total_gpus = world_size

        # define a loss function and an optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    else:
        #os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
        model = eval(config.MODULE)(config)
    
        num_gpus = config.N_GPU
        assert num_gpus <= 1 or (not config.TRAIN.FP16), "Not support fp16 with torch.nn.DataParallel. " \
                                                         "Please use amp.parallel.DistributedDataParallel instead."
        total_gpus = num_gpus
        rank = None

        # model
        if num_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=config.GPUS).cuda()
        else:
            torch.cuda.set_device(int(config.GPUS))
            model.cuda()

        # define a loss function and an optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Log all the layers in the net using wandb
    wandb.watch(model, log='all')

    # apex: amp fp16 mixed-precision training
    if config.TRAIN.FP16:
        # model.apply(bn_fp16_half_eval)
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O2',
                                          keep_batchnorm_fp32=False,
                                          loss_scale=config.TRAIN.FP16_LOSS_SCALE,
                                          min_loss_scale=32.0)
        if args.dist:
            model = Apex_DDP(model, delay_allreduce=True)

    # create the dataloaders
    train_dataset = DatasetLoader(config)
    data_size = len(train_dataset)

    # optionally the eval after every epoch
    if config.EVAL_EVERY_EPOCH:
        config_val = copy.deepcopy(config)
        config_val.RUN_MODE = 'val'
        val_dataset = DatasetLoader(config)
        val_data_size = len(val_dataset)


    # if distributed training, create a train sampler
    if args.dist:
        train_sampler = Data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    train_dataloader = Data.DataLoader(
            train_dataset,
            config.BATCH_SIZE,
            shuffle = (train_sampler is None),
            num_workers = config.NUM_WORKERS,
            pin_memory = True,
            sampler = train_sampler
        ) 

    val_dataloader = Data.DataLoader(
            val_dataset,
            config.BATCH_SIZE,
            shuffle = False,
            num_workers = config.NUM_WORKERS,
            pin_memory = True,
            sampler = None
        ) 

    # Training the net 
    for epoch in range(config.MAX_EPOCH):

        time_start = time.time()

        # iterations
        for step, (
            inputs,
            labels
        ) in enumerate(train_dataloader):

            optimizer.zero_grad()

            # make the tensors for cuda
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            
            labels = labels.cuda()

            # obtain the predictions
            pred = model(inputs)

            # loss calculation
            loss = criterion(pred, labels)

            # back propagation
            loss.backward()

            # optimize
            optimizer.step()

            # print at every step
            print("\r[Epoch: %2d][Step: %d/%d][Loss: %.4f]" % (
                epoch + 1,
                step,
                int(data_size / config.BATCH_SIZE),
                loss
            ), end='    ')

            wandb.log({
                'Loss':loss
                })

        time_end = time.time()
        time_taken = time_end-time_start
        print('Finished in {}s'.format(int(time_taken)))

        # saving the checkpoints
        if args.dist and local_rank % config.N_GPU == 0:
            state = {
                'state_dict': model.state_dict(),
                #'optimizer': optimizer.state_dict(),
                'epoch': epoch+1
            }

            torch.save(
                state,
                config.OUTPUT_PATH + 
                '/ckpt_' + config.VERSION +
                '_epoch' + str(epoch+1) + 
                '.pkl'
            )
        elif not args.dist:
            state = {
                'state_dict': model.state_dict(),
                #'optimizer': optimizer.state_dict(),
                'epoch': epoch+1
            }

            torch.save(
                state,
                config.OUTPUT_PATH + 
                '/ckpt_' + config.VERSION +
                '_epoch' + str(epoch+1) + 
                '.pkl'
            )

        # Calling the val function
        if val_dataset is not None:
            validate(
                config,
                val_dataloader,
                val_data_size,
                model,
                local_rank
            )
