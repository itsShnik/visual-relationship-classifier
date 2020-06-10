import torch
import wandb

@torch.no_grad()
def validate(config, dataloader, data_size, model, rank):

    # set model to evaluation mode
    model.eval()

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
            inputs[k] = inputs[k].cuda(rank, non_blocking=True)

        # convert labels to cuda
        labels = labels.cuda(rank, non_blocking=True)
        
        # get the prediction
        pred = model(inputs)
        _, pred_indices = torch.max(pred, 1)
        
        total += labels.size(0)
        correct += (pred_indices == labels).sum().item()

    wandb.log({
        'Accuracy':100.0 * correct/total
        })
       
    print("Accuracy of the classifier is : %d / %d = %4f" % (correct, total, 100.0 * correct/total))
