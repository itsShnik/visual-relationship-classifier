import torch
import torch.nn as nn
import torch.utils.data as Data
from vis_rel.modules import *
import wandb

@torch.no_grad()
def test_engine(config, dataset, state_dict, validation=True):

    # only for validation for now
    data_size = dataset.data_size
    num_classes = dataset.num_rel_classes

    # create a net object
    net = Net(config, num_classes)
    net.cuda()
    net.eval()

    # parallelize
    if config.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=config.GPUS)

    # load the state_dict
    net.load_state_dict(state_dict)

    # call the dataloader
    dataloader = Data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True, 
        drop_last=True
    )

    # make predictions
    total = 0
    correct = 0
    for step, (
            inputs,
            labels
        ) in enumerate(dataloader):

        print("\rEvaluation: [step %d/%d]" % (
                step,
                int(data_size / config.EVAL_BATCH_SIZE)), end='    ')

        # convert the inputs to cuda
        for k, v in inputs.items():
            inputs[k] = inputs[k].cuda()

        # convert labels to cuda
        labels = labels.cuda()
        
        # get the prediction
        pred = net(inputs)
        _, pred_indices = torch.max(pred, 1)
        
        total += labels.size(0)
        correct += (pred_indices == labels).sum().item()

    wandb.log({
        'Accuracy':100.0 * correct/total
        })
       
    print("Accuracy of the classifier is : %d / %d = %4f" % (correct, total, 100.0 * correct/total))
