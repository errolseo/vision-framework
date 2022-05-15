
import time
import torch

import hydra
from omegaconf import DictConfig
from accelerate import Accelerator

from src import build_model, build_criterion, build_optimizer, build_data
from src.utils import set_seed, AverageMeter, ProgressMeter

@hydra.main(config_path="config")
def main(config: DictConfig) -> None:
    set_seed(config.random_seed)

    # build assets
    train_dl, valid_dl = build_data(config.data)
    model = build_model(config.model)
    criterion = build_criterion(config.criterion) if "criterion" in config else None
    optimizer = build_optimizer(config.optimizer, model.parameters())

    # set accelerator
    accelerator = Accelerator()
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)
    if valid_dl:
        valid_dl = accelerator.prepare(valid_dl)
    if criterion:
        criterion = accelerator.prepare(criterion)

    # set wandb
    if 'wandb' in config:
        if accelerator.is_local_main_process:
            import wandb
            wandb.init(config=dict(config), **config.wandb)
            wandb.watch(model)

    def accuracy(output, target, top_k=(1,)):
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def run_epoch(epoch, data_loader, mode='train'):
        # set stdout function
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')
        progress = ProgressMeter(len(data_loader), batch_time, data_time, losses, acc,
                                 prefix="Epoch: [{}]".format(epoch))

        if mode == "train":
            model.train()
        else:
            model.eval()

        end = time.time()
        for i, (source, targets) in enumerate(data_loader):
            data_time.update(time.time() - end)

            optimizer.zero_grad()
            if config.model.type == "transformers":
                logits, loss = model(source, labels=targets)
            elif config.model.type == "timm":
                logits = model(source)
                loss = criterion(logits, targets)
            else:
                raise Exception("Not implemented.")

            if mode == 'train':
                accelerator.backward(loss)
                optimizer.step()

            losses.update(loss.item(), source.size(0))
            acc1 = accuracy(accelerator.gather(logits), accelerator.gather(targets))
            acc.update(acc1[0].item(), source.size(0))
            batch_time.update(time.time() - end)

            if (i % config.print_freq == 0) or (i + 1 == len(data_loader)):
                progress.print(accelerator, i)
            end = time.time()

    for ep in range(config.epoch):
        run_epoch(ep, train_dl)
        if valid_dl:
            with torch.no_grad():
                run_epoch(ep, valid_dl, mode='valid')


if __name__ == '__main__':
    main()
