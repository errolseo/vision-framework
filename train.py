
import os
import time

import torch
import hydra
from omegaconf import DictConfig
from accelerate import Accelerator, DistributedDataParallelKwargs

from src import build_model, build_criterion, build_optimizer, build_data
from src.utils import set_seed, accuracy, AverageMeter, ProgressMeter

@hydra.main(config_path="config")
def main(config: DictConfig) -> None:
    # set
    set_seed(config.random_seed)
    os.makedirs(os.path.join(os.getcwd(), "model"), exist_ok=True)

    # build assets
    train_dl, valid_dl = build_data(config.data)
    model = build_model(config.model)
    criterion = build_criterion(config.criterion) if "criterion" in config else None
    optimizer = build_optimizer(config.optimizer, model.parameters())

    # set accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)

    # set wandb
    if 'wandb' in config:
        if accelerator.is_local_main_process:
            import wandb
            wandb.init(config=dict(config), **config.wandb)
            wandb.watch(model)

    def run_epoch(epoch, data_loader, mode='train'):
        model.train() if mode == "train" else model.eval()

        # set logger
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')
        progress = ProgressMeter(len(data_loader), batch_time, data_time, losses, acc,
                                 prefix=f"Epoch (T): [{epoch}]" if mode == "train" else f"Epoch (V): [{epoch}]")

        start_time = time.time()
        for i, (source, targets) in enumerate(data_loader):
            data_time.update(time.time() - start_time)

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
            batch_time.update(time.time() - start_time)

            if i % config.print_freq == 0:
                progress.print(accelerator, i)
            start_time = time.time()

        progress.print(accelerator, len(data_loader))
        if accelerator.is_local_main_process:
            wandb.log({
                f"{mode}_loss": losses.avg,
                f"{mode}_acc": acc.avg
            })

    # running train and valid
    for ep in range(config.epoch):
        run_epoch(ep, train_dl)
        if valid_dl and (ep + 1)  % config.valid_freq == 0:
            with torch.no_grad():
                run_epoch(ep, valid_dl, mode='valid')

            accelerator.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(os.getcwd(), "model", f"epoch_{ep + 1}.pth"))


if __name__ == '__main__':
    main()
