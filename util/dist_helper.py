import os
import subprocess

import torch
import torch.distributed as dist
from pathlib import Path


def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def save_model(args, epoch, model_without_ddp, optimizer):
    output_dir = Path(args.save_dir)
    epoch_name = str(epoch)

    checkpoint_path = os.path.join(output_dir, 'checkpoint-%s.pth' % epoch_name)

    to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
    }
    save_on_master(to_save, checkpoint_path)

def save_model_ibot(args, epoch, model_student, model_teacher, optimizer, ibot_loss):
    output_dir = Path(args.save_dir)
    epoch_name = str(epoch)

    checkpoint_path = os.path.join(output_dir, 'checkpoint-%s.pth' % epoch_name)

    to_save = {
            'model_student': model_student.state_dict(),
            'model_teacher': model_teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
    }
    save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("With optim & sched!")



