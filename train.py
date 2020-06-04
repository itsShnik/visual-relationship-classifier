# all the imports go here
import torch
import torch.nn as nn
import torch.utils.data as Data
from net import Net
import torch.optim as optim
import time

# the train engine here
def train_engine(config, dataset, dataset_eval=None):

    # here you will get all the things ffrom the dataset init
    data_size = dataset.data_size
    num_classes = dataset.num_rel_classes

    # Next you'll have to create a Net
    net = Net(config, num_classes)
    net.cuda()
    net.train()

    # parallelization
    if config.N_GPU > 1:
        net = nn.parallel.DataParallel(net, device_ids = config.GPUS)

    # Then the dataset enumerator and batching
    dataloader = Data.DataLoader(
            dataset,
            batch_size = config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )

    # define a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training the net 
    for epoch in range(config.MAX_EPOCH):

        time_start = time.time()

        # iterations
        for step, (
            inputs,
            labels
        ) in enumerate(dataloader):

            optimizer.zero_grad()

            # make the tensors for cuda
            for inp_tensor in inputs.values():
                inp_tensor = inp_tensor.cuda()
            
            labels = labels.cuda()

            # obtain the predictions
            pred = net(inputs)

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


        time_end = time.time()
        time_taken = time_end-time_start
        print('Finished in {}s'.format(int(time_taken)))

        # saving the checkpoints
        if config.N_GPU > 1:
            state = {
                'state_dict': net.module.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'epoch': epoch+1
            }
        else:
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'epoch': epoch+1
            }

            torch.save(
                state,
                config.CKPTS_PATH + 
                '/ckpt_' + config.VERSION +
                '/epoch' + str(epoch+1) + 
                '.pkl'
            )

    # Calling the val function
    if dataset_eval is not None:
        test_engine(
            config,
            dataset_eval,
            state_dict=net.state_dict(),
            validation=True
        )
