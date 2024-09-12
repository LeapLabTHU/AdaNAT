import argparse
import math
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import accelerate
import einops
import numpy as np
import torch
from einops import rearrange
from loguru import logger
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

import taming.models.vqgan
import utils_uvit
from PPO import PPO
from libs.discriminator import ProjectedDiscriminator
from torch_fidelity import FeatureExtractorInceptionV3
from tsvdataset import CC3MTSV
from torchvision import datasets
from utils_uvit import auto_load_model, calc_fid


class BiasAct:
    def __init__(self):
        ratio = (torch.arange(args.gen_steps) + 1 - 1e-3) / args.gen_steps

        mask_ratio = torch.cos(ratio * math.pi * 0.5)
        self.manual_gen_ratios = torch.log(mask_ratio / (1 - mask_ratio)).to(device)

        def inverse_softplus(x):
            return x + torch.log(-torch.exp(-x) + 1)

        self.manual_temp = inverse_softplus(1 - ratio).to(device)
        self.manual_samp_temp = inverse_softplus(torch.ones((args.gen_steps,), dtype=torch.float32)).to(device)
        self.manual_cfg = inverse_softplus((torch.arange(args.gen_steps) + 1e-3) / args.gen_steps).to(device)

        self.activations = {'manual_gen_ratios': nn.Sigmoid(), 'manual_temp': nn.Softplus(),
                            'manual_samp_temp': nn.Softplus(), 'manual_cfg': nn.Softplus()}

    def __call__(self, actions, timestep=None):
        if timestep is not None:
            residuals = {k: getattr(self, k)[timestep] for k in args.upd_set}
        else:
            residuals = {k: einops.repeat(getattr(self, k), 'T -> B T', B=actions.shape[0]) for k in args.upd_set}
        if args.heu:
            actdict = {k: actions[:, idx] + residuals[k] for idx, k in enumerate(args.upd_set)}
        else:
            actdict = {k: actions[:, idx] for idx, k in enumerate(args.upd_set)}
        actdict = {k: self.activations[k](v) for k, v in actdict.items()}
        return actdict


def action2dict(action, timestep=None):
    B = action.shape[0]
    action = action.reshape(B, -1)
    assert action.shape[1] == len(args.upd_set), 'action shape: {}, upd_set: {}'.format(action.shape, args.upd_set)
    actdict = bias_act(action, timestep=timestep)
    return actdict


def get_args():
    parser = argparse.ArgumentParser()
    # Basics
    parser.add_argument('--has_continuous_action_space', type=bool, default=True)
    parser.add_argument('--max_training_timesteps', type=int, default=100000000)
    parser.add_argument('--save_model_freq', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--state_opt', type=str, nargs='+', default=['timestep', 'feat'])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--upd_set', type=str, nargs='+', default=['manual_gen_ratios', 'manual_temp',
                                                                   'manual_samp_temp', 'manual_cfg'])
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--dset', type=str, default='in256', choices=['in256', 'cc3m'])
    parser.add_argument('--eval_paths', type=str, nargs='+')
    parser.add_argument('--resume', type=str)
    # PPO hyperparameters
    parser.add_argument('--K_epochs', type=int, default=5)
    parser.add_argument('--D_epochs', type=int, default=5)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--trajectories_per_upd', type=int, default=4096)
    parser.add_argument('--action_std', type=float, nargs='+', default=[0.6])
    parser.add_argument('--decay_steps', type=int, nargs='+', default=[500])
    parser.add_argument('--decay_rate', type=float, default=0.3)
    parser.add_argument('--min_action_std', type=float, default=0.1)
    # NAT config
    parser.add_argument('--config', type=str)
    parser.add_argument('--state_dict_path', type=str, default=None)
    # NAT generation config
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--reference_image_path', type=str,
                        default='assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz')
    parser.add_argument('--n_samples', type=int, default=50000)
    parser.add_argument('--gen_steps', type=int, default=8)
    parser.add_argument('--heu', type=int, default=0,
                        help='Optionally use manual scheduling rules of existing works as residual for better performance')
    # Discriminator config
    parser.add_argument('--d_loss', type=str, default='bce')
    parser.add_argument('--data_transform', type=int, default=1)
    parser.add_argument('--c_dim', type=int, default=512)
    parser.add_argument('--output_dir', type=str)
    if os.getenv('DEBUG', 'f') == 't':
        args = parser.parse_known_args()[0]
    else:
        args = parser.parse_args()

    return args


class MuseGenerator:
    def __init__(self):
        self.seq_len = 256
        self.mask_ind = 1024
        self.nnet = self.prepare_nnet()
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor([0.0], device=device),
                                                             torch.tensor([1.0], device=device))
        self.context_generator = self.prepare_context_generator()

    def prepare_context_generator(self):
        if args.dset == 'in256':
            while True:
                yield torch.randint(0, 1000, (args.batch_size, 1), device=device)
        elif args.dset == 'cc3m':
            dataset = CC3MTSV(args.data_root, 'train', transform=partial(utils_uvit.adm_transform), txt_only=True)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
            dataloader = accelerator.prepare(dataloader)
            logger.info('train dataloader length: {}'.format(len(dataloader)))
            while True:
                for data in dataloader:
                    real_ids = tokenizer(data, max_length=77, padding='max_length', truncation=True,
                                         return_tensors='pt').input_ids
                    real_txt_features = txt_encoder(input_ids=real_ids.to(device))
                    yield real_txt_features
        else:
            raise NotImplementedError

    def reset(self, contexts=None):
        contexts = next(self.context_generator) if contexts is None else contexts
        if isinstance(contexts, CLIPTextModelOutput):
            B = contexts.text_embeds.shape[0]
        else:
            B = len(contexts)
        masked_ids = torch.full((B, self.seq_len), self.mask_ind, dtype=torch.long, device=device)
        self.state = self.nnet(masked_ids, context=contexts, return_dict=True)
        self.state['masked_ids'] = masked_ids
        self.state['timestep'] = torch.zeros((B,), dtype=torch.long, device=device)
        self.state['contexts'] = contexts
        return self.state

    def prepare_nnet(self):
        self.config = OmegaConf.load(args.config)
        self.nnet = utils_uvit.get_nnet(**self.config.nnet)
        self.nnet = accelerator.prepare(self.nnet)
        ckpt = torch.load(args.state_dict_path, map_location='cpu')
        ckpt = {k: v for k, v in ckpt.items() if 'position_ids' not in k}  # add handling for cc3m ckpt
        self.nnet.module.load_state_dict(ckpt)
        self.nnet.eval()
        self.nnet.requires_grad_(False)
        return self.nnet

    def add_gumbel_noise(self, t, temperature):
        gumbel_sample = self.gumbel_dist.sample(t.shape).squeeze()
        result = t + temperature.unsqueeze(1) * gumbel_sample
        return result

    def step(self, manual_gen_ratios, manual_temp, manual_samp_temp, manual_cfg):
        logits = self.state['logits']
        _empty_ctx = einops.repeat(empty_ctx, '... -> B ...', B=len(manual_gen_ratios))
        logits_wo_cfg = self.nnet(self.state['masked_ids'], context=_empty_ctx)
        logits = logits + manual_cfg.unsqueeze(1).unsqueeze(1) * (logits - logits_wo_cfg)
        # sampling & scoring
        is_mask = (self.state['masked_ids'] == self.mask_ind)
        sampled_ids = torch.distributions.Categorical(
            logits=logits / manual_samp_temp.unsqueeze(1).unsqueeze(2)).sample()
        logits = torch.log_softmax(logits, dim=-1)
        sampled_logits = torch.squeeze(
            torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
        sampled_ids = torch.where(is_mask, sampled_ids, self.state['masked_ids'])
        sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
        # masking
        mask_len = torch.floor(self.seq_len * manual_gen_ratios).clamp(max=self.seq_len-1)
        confidence = self.add_gumbel_noise(sampled_logits, manual_temp)
        sorted_confidence, _ = torch.sort(confidence, axis=-1)
        cut_off = torch.gather(sorted_confidence, 1, mask_len.long().unsqueeze(1))
        masking = (confidence < cut_off)
        masked_ids = torch.where(masking, self.mask_ind, sampled_ids)
        # update state
        self.state.update(self.nnet(masked_ids, context=self.state['contexts'], return_dict=True))
        self.state['masked_ids'] = masked_ids
        self.state['sampled_ids'] = sampled_ids
        self.state['timestep'] = self.state['timestep'] + 1
        return self.state


class FIDCalculator:
    def __init__(self, ref_stats_path, n_samples, gather=True):
        feature_extractor = FeatureExtractorInceptionV3(
            name='inception-v3',
            features_list=['2048', 'logits_unbiased'],
            feature_extractor_internal_dtype='float32',
            feature_extractor_weights_path='assets/pt_inception-2015-12-05-6726825d.pth'
        ).to(device)

        self.inception = feature_extractor
        self.n_samples = n_samples
        self.gather = gather

        with np.load(ref_stats_path) as f:
            self.ref_stats = (f['mu'][:], f['sigma'][:])
            self.ref_stats = [torch.from_numpy(x).to(device) for x in self.ref_stats]

        batch_size = args.batch_size * accelerator.num_processes if gather else args.batch_size
        self.total_iters = len(utils_uvit.amortize(self.n_samples, batch_size))
        self.has_init = False

    def init(self):
        self.pred_tensor = torch.empty((self.n_samples * 2, 2048), device=device)
        self.logits_tensor = torch.empty((self.n_samples * 2, 1008), device=device)
        self.idx = 0
        self.pbar = tqdm(total=self.total_iters, desc='FID', leave=True)
        self.has_init = True

    def get_metrics(self):
        pred_tensor = self.collate_tensor(self.pred_tensor)
        fid = calc_fid(pred_tensor, *self.ref_stats)
        logger.info('FID: {}'.format(fid))

        logits_tensor = self.collate_tensor(self.logits_tensor)
        isc = utils_uvit.isc_features_to_metric(logits_tensor)
        logger.info('ISC: {}'.format(isc))
        self.has_init = False
        return {f'fid_{self.n_samples}': fid, f'isc_{self.n_samples}': isc}

    def collate_tensor(self, tensor):
        tensor = tensor[:self.idx]
        if self.gather:
            tensor = accelerator.gather(tensor)
        assert self.n_samples <= tensor.shape[0]
        logger.info('Truncating tensor from {} to {} samples'.format(tensor.shape[0], self.n_samples))
        tensor = tensor[:self.n_samples]
        return tensor

    def add(self, samples):
        if not self.has_init:
            self.init()
        samples = samples.clamp_(0., 1.)
        features_2048, logits_unbiased = self.inception(samples.float())
        self.pred_tensor[self.idx:self.idx + features_2048.shape[0]] = features_2048
        self.logits_tensor[self.idx:self.idx + logits_unbiased.shape[0]] = logits_unbiased

        self.idx = self.idx + features_2048.shape[0]
        self.pbar.update(1)

        if self.pbar.n == self.total_iters:
            return self.get_metrics()


class DiscriminatorEnv:
    def __init__(self):
        super(DiscriminatorEnv, self).__init__()
        self.real_data_generator = self.real_imgs_generator()

        self.discriminator = self.prepare_discriminator()
        self.autoencoder = taming.models.vqgan.get_model().to(device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
        self.discriminator, self.optimizer = accelerator.prepare(self.discriminator, self.optimizer)

    def real_imgs_generator(self):
        if args.dset == 'in256':
            real_data = datasets.ImageFolder(args.data_root, transform=utils_uvit.adm_transform)
            dataloader = DataLoader(real_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                    pin_memory=True, persistent_workers=True)
            dataloader = accelerator.prepare(dataloader)
            while True:
                for data in dataloader:
                    real_samples, real_labels = data
                    real_labels = real_labels.unsqueeze(1).to(device)
                    real_samples = real_samples.to(device)
                    yield real_samples, real_labels
        elif args.dset == 'cc3m':
            dataset = CC3MTSV(args.data_root, 'train', transform=partial(utils_uvit.adm_transform), txt_only=False)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
            dataloader = accelerator.prepare(dataloader)
            while True:
                for data in dataloader:
                    real_samples, real_txts = data
                    real_ids = tokenizer(real_txts, max_length=77, padding='max_length', truncation=True,
                                         return_tensors='pt').input_ids
                    real_txt_features = txt_encoder(input_ids=real_ids.to(device))
                    real_samples = real_samples.to(device)
                    yield real_samples, real_txt_features
        else:
            raise NotImplementedError

    def prepare_discriminator(self):
        discriminator = ProjectedDiscriminator(c_dim=args.c_dim, data_transform=args.data_transform,
                                               cin=(args.dset == 'in256'))
        logger.info('Discriminator has {} parameters'.format(sum(p.numel() for p in discriminator.parameters())))
        return discriminator

    @torch.no_grad()
    def calc_reward(self, sampled_ids, contexts, done=False):
        if done:
            self.discriminator.eval()
            decoded_samples = self.decode(sampled_ids)
            reward = self.discriminator(decoded_samples, contexts)  # Bx1
            reward = reward.mean(dim=1, keepdim=True)
            reward = torch.sigmoid(reward)
        else:
            reward = torch.zeros((len(sampled_ids),), dtype=torch.float32, device=device)
            decoded_samples = None
        return reward, decoded_samples

    @torch.cuda.amp.autocast(enabled=True)
    def decode(self, sampled_ids):
        _z = rearrange(sampled_ids, 'b (i j) -> b i j', i=16, j=16)
        res = self.autoencoder.decode_code(_z)
        res = res.clamp_(0., 1.)
        return res

    def discriminator_forward(self, fake_samples, fake_labels):
        metrics = {}
        # prepare data
        real_samples, real_labels = next(self.real_data_generator)

        # forward
        with torch.set_grad_enabled(True):
            real_logits = self.discriminator(real_samples, real_labels)
            fake_logits = self.discriminator(fake_samples, fake_labels)
            if args.d_loss == 'bce':
                loss = torch.nn.functional.softplus(-real_logits).mean() + \
                       torch.nn.functional.softplus(fake_logits).mean()
            elif args.d_loss == 'hinge':
                loss = nn.ReLU()(1 - real_logits).mean() + \
                       nn.ReLU()(1 + fake_logits).mean()
            else:
                raise NotImplementedError
        # log
        real_logits = accelerator.gather(real_logits)
        fake_logits = accelerator.gather(fake_logits)
        metrics['d_loss'] = accelerator.gather(loss.detach()).mean().item()
        metrics['d_acc'] = ((real_logits > 0).float().mean() + (fake_logits < 0).float().mean()) / 2
        metrics['real_scores'] = real_logits.mean().item()
        metrics['fake_scores'] = fake_logits.mean().item()
        metrics['real_signs'] = real_logits.sign().mean().item()
        metrics['fake_signs'] = fake_logits.sign().mean().item()
        return loss, metrics

    def update_discriminator(self, fake_samples, fake_labels):
        self.discriminator.train()
        self.optimizer.zero_grad()
        loss, metrics = self.discriminator_forward(fake_samples, fake_labels)
        accelerator.backward(loss.mean())
        self.optimizer.step()
        return metrics


################################### Training ###################################
@logger.catch(reraise=(os.getenv('DEBUG', 'f') == 't'))
def train():
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info(f'Process {accelerator.process_index} using device: {device}')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.remove()
        logger.add(sys.stdout, level='INFO')
        logger.add(os.path.join(args.output_dir, 'output.log'), level='INFO')
    else:
        logger.remove()
        logger.add(sys.stdout, filter=lambda record: record["level"].name == "TRACE", level="TRACE")
    logger.info(f'Run on {accelerator.num_processes} devices')

    env = MuseGenerator()
    disc = DiscriminatorEnv()

    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    action_dim = len(args.upd_set)

    # prep action std
    action_std = torch.tensor(args.action_std, dtype=torch.float32, device=accelerator.device)
    if len(args.action_std) == 1:
        action_std = action_std.repeat(action_dim)
    else:
        assert len(args.action_std) == len(args.upd_set)
    assert len(action_std) == action_dim

    # initialize a PPO agent
    ppo_agent_wo_ddp = PPO(action_dim, args.lr, args.lr, args.gamma, args.K_epochs, args.eps_clip,
                           args.has_continuous_action_space, action_std_init=action_std,
                           device=accelerator.device,
                           state_opt=args.state_opt,
                           feat_dim=env.nnet.module.embed_dim, args=args)

    optimizer = torch.optim.Adam(ppo_agent_wo_ddp.parameters(), lr=args.lr)

    ppo_agent, optimizer = accelerator.prepare(ppo_agent_wo_ddp, optimizer)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    counters = {
        'global_step': 0,  # collect trajectories & update PPO agent
        'num_trajectories': 0,
        'num_agent_updates': 0,
        'num_discriminator_updates': 0,
    }
    if args.eval_paths:
        for eval_path in args.eval_paths:
            args.resume = eval_path
            auto_load_model(args, ppo_agent_wo_ddp, optimizer, disc.discriminator, disc.optimizer, accelerator.scaler,
                            counters=counters)
            evaluate(counters, env, disc, ppo_agent_wo_ddp)
        return
    else:
        auto_load_model(args, ppo_agent_wo_ddp, optimizer, disc.discriminator, disc.optimizer, accelerator.scaler,
                        counters=counters)

    best_fid = 1e9
    # training loop
    while counters['global_step'] < args.max_training_timesteps:
        # collect trajectories
        for _ in range(0, args.trajectories_per_upd, args.batch_size * accelerator.num_processes):
            env.reset()
            while True:
                action = ppo_agent(state=env.state, flag='select_action')
                env.step(**action2dict(action, timestep=env.state['timestep']))
                done = (env.state['timestep'][0].item() == args.gen_steps)
                reward, _ = disc.calc_reward(env.state['sampled_ids'], env.state['contexts'], done=done)
                ppo_agent(reward=reward, done=done, flag='store_transition')
                if done:
                    break
            counters['num_trajectories'] += args.batch_size * accelerator.num_processes
        # update PPO agent
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(ppo_agent.module.buffer.rewards),
                                       reversed(ppo_agent.module.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (ppo_agent.module.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        ppo_agent.module.buffer.rewards = rewards
        old_actions, old_states, old_logprobs, rewards, old_state_values = ppo_agent.module.buffer.to_tensor()
        running_rewards = accelerator.gather(rewards)
        rewards = (rewards - running_rewards.mean()) / (running_rewards.std() + 1e-7)
        advantages = rewards.detach() - old_state_values.detach()
        for _ in range(args.K_epochs):
            loss = ppo_agent(old_states, old_actions, old_logprobs, rewards, advantages, args)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
        counters['num_agent_updates'] += args.K_epochs
        ppo_agent.module.policy_old.load_state_dict(ppo_agent.module.policy.state_dict())
        ppo_agent.module.buffer.clear()
        # std decay
        def calculate_action_std(global_step, decay_rate, decay_steps, initial_action_std, min_action_std):
            assert isinstance(initial_action_std, list) and len(initial_action_std) == 1
            initial_action_std = initial_action_std[0]
            reductions = sum(global_step >= step for step in decay_steps)
            current_action_std = max(initial_action_std - reductions * decay_rate, min_action_std)
            return current_action_std
        decayed_action_std = calculate_action_std(counters['global_step'], args.decay_rate, args.decay_steps, args.action_std, args.min_action_std)
        decayed_action_std = torch.tensor([decayed_action_std], dtype=torch.float32, device=accelerator.device)
        decayed_action_std = decayed_action_std.repeat(action_dim)
        ppo_agent_wo_ddp.set_action_std(decayed_action_std)

        if accelerator.is_main_process:
            log_dict = dict(train_metrics={f'reward_mean': running_rewards.mean().item(),
                                           f'reward_std': running_rewards.std().item()},
                            **counters,
                            action_std_mean=ppo_agent_wo_ddp.action_std.mean().item(),
                            lr=optimizer.param_groups[0]['lr'],
                            )
            logger.info(log_dict)
        # update discriminator
        for i in range(args.D_epochs):
            env.reset()
            while True:
                action = ppo_agent(state=env.state, flag='select_action', update_buffer=False)
                env.step(**action2dict(action, timestep=env.state['timestep']))
                done = (env.state['timestep'][0].item() == args.gen_steps)
                if done:
                    break
            fake_labels = env.state['contexts']
            fake_samples = disc.decode(env.state['sampled_ids'])
            d_metrics = disc.update_discriminator(fake_samples, fake_labels)
            counters['num_discriminator_updates'] += 1
        if accelerator.is_main_process:
            log_dict = dict(train_metrics=d_metrics,
                            **counters,
                            )
            logger.info(log_dict)
        # validate
        if (counters['global_step'] + 1) % args.eval_freq == 0:
            fid = evaluate(counters, env, disc, ppo_agent_wo_ddp)
            if fid < best_fid:
                best_fid = fid
                logger.info('Current Best FID: {}'.format(best_fid))
        counters['global_step'] += 1
        # save model weights
        if (counters['global_step'] + 1) % args.save_model_freq == 0 and accelerator.is_main_process:
            state_dict = {
                'model': ppo_agent_wo_ddp.state_dict(),
                'discriminator': disc.discriminator.state_dict(),
                'd_optimizer': disc.optimizer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                **counters
            }
            if accelerator.scaler is not None:
                state_dict['scaler'] = accelerator.scaler.state_dict()
            torch.save(state_dict, os.path.join(args.ckpt_dir, f'ckpt_{counters["global_step"]}.pth'))

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


def evaluate(counters, env, disc, ppo_agent_wo_ddp):
    fid_calculator = FIDCalculator(ref_stats_path=args.reference_image_path,
                                   n_samples=args.n_samples, gather=True)
    test_rewards = []
    test_actions = []
    while True:
        if args.dset == 'cc3m':
            contexts, _ = next(test_txt_generator)
        else:
            contexts, _ = None, None
        env.reset(contexts=contexts)
        while True:  # collect trajectory
            test_action = ppo_agent_wo_ddp.policy_old.actor(env.state)
            env.step(**action2dict(test_action, timestep=env.state['timestep']))
            test_done = (env.state['timestep'][0].item() == args.gen_steps)
            test_reward, decoded_samples = disc.calc_reward(env.state['sampled_ids'],
                                                            env.state['contexts'], done=test_done)
            test_actions.append(test_action)
            if test_done:
                break
        res = fid_calculator.add(decoded_samples)
        test_rewards.append(test_reward)
        if res is not None:
            break
    # calc test actions
    test_rewards = torch.cat(test_rewards)
    test_rewards = accelerator.gather(test_rewards)
    test_reward_mean, test_reward_std = test_rewards.mean().item(), test_rewards.std().item()
    test_actions = torch.stack(test_actions).reshape(-1, args.gen_steps, args.batch_size, len(args.upd_set))
    test_actions = test_actions.permute(0, 2, 3, 1).contiguous()
    test_actions = accelerator.gather(test_actions).reshape(-1, len(args.upd_set), args.gen_steps)
    actdict = bias_act(test_actions)
    actdict_mean = {f'{k}_{i}': actdict[k][:, i].mean().item() for i in range(args.gen_steps) for k in args.upd_set}

    if accelerator.is_main_process:
        write_dict = dict(
            eval_metrics={**res,
                          f'reward_mean': test_reward_mean,
                          f'reward_std': test_reward_std},
            **{k: v.mean(dim=0).tolist() for k, v in actdict.items()}, **actdict_mean,
            **counters,
        )
        logger.info(write_dict)
    return res[f'fid_{args.n_samples}']


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=(os.getenv('find_unused', 't') == 't'), broadcast_buffers=False)

    args = get_args()
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision='fp16')
    logger.info(f'accelerator mixed precision: {accelerator.mixed_precision}')
    device = accelerator.device
    muse_gen_for_eval_imgs = MuseGenerator()

    if args.dset == 'cc3m':
        # prepare text encoder
        from transformers import CLIPTokenizer, CLIPTextModelWithProjection
        tokenizer = CLIPTokenizer.from_pretrained(f'assets/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k')
        tokenizer.pad_token_id = 0  # align with cc3m muse ckpt pretraining
        txt_encoder = CLIPTextModelWithProjection.from_pretrained('assets/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k', projection_dim=1280).to(device)
        txt_encoder.eval()
        txt_encoder.requires_grad_(False)
        # prepare cc3m validation set
        test_txt_dataset = CC3MTSV(args.data_root, split='val', txt_only=True)
        test_txt_dataloader = DataLoader(test_txt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_txt_dataloader = accelerator.prepare(test_txt_dataloader)
        logger.info('test dataloader length: {}'.format(len(test_txt_dataloader)))
        def get_test_txt_generator():
            while True:
                for caption in test_txt_dataloader:
                    real_ids = tokenizer(caption, max_length=77, padding='max_length', truncation=True,
                                         return_tensors='pt').input_ids
                    real_txt_features = txt_encoder(input_ids=real_ids.to(device))
                    yield real_txt_features, caption
        test_txt_generator = get_test_txt_generator()

    # prepare empty context for classifier-free guidance
    if args.dset == 'in256':
        empty_ctx = torch.full((1,), 1000, dtype=torch.long, device=device)
    elif args.dset == 'cc3m':
        empty_prompt = ['']
        empty_tok_res = tokenizer(empty_prompt, max_length=77, padding='max_length', truncation=True,
                                  return_tensors='pt')
        empty_ctx = txt_encoder(input_ids=empty_tok_res['input_ids'].to(device)).last_hidden_state.squeeze(0)
    else:
        raise NotImplementedError

    bias_act = BiasAct()
    train()