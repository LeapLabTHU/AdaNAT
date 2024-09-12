import pickle
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import os

from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from torch import distributed as dist
from loguru import logger

logging = logger


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    from libs.uvit import UViT
    return UViT(**kwargs)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name='customized', args=None, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from timm.scheduler.cosine_lr import CosineLRScheduler
        return CosineLRScheduler(
            optimizer,
            t_initial=args.train_steps,
            warmup_t=args.warmup_steps,
        )
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None, args=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema
        self.args = args

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key not in ['step', 'args'] and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path, ignored_keys=tuple()):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'), map_location='cpu')
        ignored_keys += ('step', 'args')
        for key, val in self.__dict__.items():
            if key not in ignored_keys and val is not None:
                ckpt = torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu')
                if key in ['nnet', 'nnet_ema']:
                    ckpt = {k: v for k, v in ckpt.items() if 'position_ids' not in k}
                val.load_state_dict(ckpt)
            else:
                logging.info(f'<<load state dict>>: ignore {key}')

    def resume(self, ckpt_root, step=None, ignored_keys=tuple()):
        if not os.path.exists(ckpt_root):
            logger.warning('ckpt_root does not exist when resuming.')
            return
        if ckpt_root.endswith('.ckpt'):
            ckpt_path = ckpt_root
        else:
            if step is None:
                ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
                if not ckpts:
                    return
                steps = map(lambda x: int(x.split(".")[0]), ckpts)
                step = max(steps)
            ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path, ignored_keys=ignored_keys)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device, tokenizer, args, txt_encoder=None):
    params = []

    nnet = get_nnet(**config.nnet, tokenizer=tokenizer, args=args, txt_encoder=txt_encoder)

    if args.pretrained_path and os.getenv('DEBUG', 'f') == 'f':
        logger.info(f'load pretrained model from {args.pretrained_path}')
        ckpt = torch.load(args.pretrained_path, map_location='cpu')
        if args.distill_to_lhs:
            teacher_ctx_emb_weights = {k.replace('context_embed.', ''): v for k, v in ckpt.items()
                                       if 'context_embed' in k}
            nnet.teacher_ctx_embed.load_state_dict(teacher_ctx_emb_weights)
        # remove keys that contains 'context_embed'
        if args.load_ctx_emb == 0:
            ckpt = {k: v for k, v in ckpt.items() if 'context_embed' not in k}
        msg = nnet.load_state_dict(ckpt, strict=False)
        logger.info(msg)

    if args.only_tune_adapt:
        assert args.pretrained_path
        nnet.requires_grad_(False)

        params += nnet.context_embed.parameters()
        if txt_encoder is not None:  # llm_query
            assert args.llm_query
            params.append(nnet.txt_encoder.query_emb)

        for p in params:
            p.requires_grad_(True)
    else:
        params += nnet.parameters()
    if args.use_ema:  # have problems for txt encoder!!!
        nnet_ema = get_nnet(**config.nnet, tokenizer=tokenizer, args=args, txt_encoder=txt_encoder)
        nnet_ema.eval()
    else:
        nnet_ema = None
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, args=args, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema, args=args)
    if args.use_ema:
        train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, dist=True):
    if path:
        os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes if dist else mini_batch_size

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process,
                            desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn())
        if dist:
            samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


from collections import defaultdict, deque


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if torch.__version__.startswith('2'):
        from torch import inf
    else:
        from torch._six import inf
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def adm_transform(pil_image, is_train=True):
    pil_image = pil_image.convert("RGB")
    arr = center_crop_arr(pil_image, 256)

    if is_train and random.random() < 0.5:
        arr = arr[:, ::-1]

    arr = arr.astype(np.float32) / 255.
    arr = np.transpose(arr, [2, 0, 1])
    return arr


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """
  Args:
    u: (N, T, D) tensor.
    v: (N, T, D) tensor.
  Returns:
    l1_loss: (N,) tensor of summed L1 loss.
  """
  assert u.shape == v.shape, (u.shape, v.shape)
  return ((u - v) ** 2).sum(dim=-1) ** 0.5

def all_gather(data, accelerator):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = accelerator.num_processes
    device = accelerator.device
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(device)
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(device))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def auto_load_model(args, model_without_ddp, optimizer, discriminator, d_optimizer, loss_scaler=None, lr_scheduler=None, counters=None):
    checkpoint = None
    if args.resume is not None:  # overriding auto-resuming
        assert Path(args.resume).exists(), 'Please specify a valid path for resuming'
        all_checkpoints = [args.resume]
    else:
        checkpoint = None
        import re
        all_checkpoints = sorted(
            Path(args.ckpt_dir).glob('ckpt_*.pth'),
            key=lambda x: int(re.search(r'ckpt_(\d+).pth', str(x)).group(1)),
            reverse=True
        )
        logger.debug('all checkpoints: ' + str(all_checkpoints))
    for ckpt_path in all_checkpoints:
        try:
            logger.debug(f'Try loading ckpt from {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            break
        except:
            logger.warning(f'{ckpt_path} cannot be loaded, try next')

    if checkpoint is not None:
        model_without_ddp.load_state_dict(checkpoint['model'])
        logger.info(f'Resume from {ckpt_path}')
        if counters is not None:
            for key in counters:
                if key in checkpoint:
                    counters[key] = checkpoint[key]
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        if 'discriminator' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if 'd_optimizer' in checkpoint:
            d_optimizer.load_state_dict(checkpoint['d_optimizer'])

def calc_fid(pred_tensor, m2, s2):
    m1 = torch.mean(pred_tensor, dim=0)
    pred_centered = pred_tensor - pred_tensor.mean(dim=0)
    s1 = torch.mm(pred_centered.T, pred_centered) / (pred_tensor.size(0) - 1)

    m1 = m1.double()
    s1 = s1.double()
    a = (m1 - m2).square().sum(dim=-1)
    b = s1.trace() + s2.trace()
    c = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum(dim=-1)

    _fid = (a + b - 2 * c).item()
    return _fid

def prepare_nnet(config_path, state_dict_path, accelerator, args):
    config = OmegaConf.load(config_path)
    nnet = get_nnet(**config.nnet, args=args)
    nnet = accelerator.prepare(nnet)
    nnet.module.load_state_dict(torch.load(args.state_dict_path + '/nnet_ema.pth', map_location='cpu'))
    nnet.eval()
    nnet.requires_grad_(False)
    return nnet


def isc_features_to_metric(feature, splits=10, shuffle=True, rng_seed=2020):
    assert torch.is_tensor(feature) and feature.dim() == 2
    N, C = feature.shape
    logger.info(f'isc feature shape is: {N},{C}')
    if shuffle:
        rng = np.random.RandomState(rng_seed)
        feature = feature[rng.permutation(N), :]
    feature = feature.double()

    p = feature.softmax(dim=1)
    log_p = feature.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * N // splits): ((i + 1) * N // splits), :]
        log_p_chunk = log_p[(i * N // splits): ((i + 1) * N // splits), :]
        q_chunk = p_chunk.mean(dim=0, keepdim=True)
        kl = p_chunk * (log_p_chunk - q_chunk.log())
        kl = kl.sum(dim=1).mean().exp().item()
        scores.append(kl)

    return float(np.mean(scores))


