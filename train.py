import torch.nn as nn
import torch.optim as optim
import torch

import gym

from stable_baselines3 import A2C
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import safe_mean

#from register_policies import ImpalaPolicy
from utils import *
from env_wrapper import *

import numpy as np
import random
import argparse, pickle
import multiprocessing

import os, time, datetime, sys

class AnnotationBuffer(object):
    """Buffer of annotated pairs of clips

    Each entry is ([clip0, clip1], label)
    clip0, clip2 : lists of observations
    label : float in range {0, 0.5, 1} corresponding to which clip is preferred,
    where 0.5 means that clips are equal
    """

    def __init__(self, max_size=3000):
        self.max_size = max_size
        self.current_size = 0
        self.total_labels = 0

        #calculate max train and validation set sizes based on total max_size
        self.train_max_size = int(self.max_size * (1 - 1/np.exp(1)))
        self.val_max_size = self.max_size - self.train_max_size

        self.train_data_all = []
        self.val_data_all = []


    def add(self, data):
        '''
        1/e of data goes to the validatation set
        the rest goes to the training set
        '''
        # determine how much goes to train vs val set, such that
        # the total split is proportional to (e-1)/e
        new_train_size = int((self.current_size + len(data)) * (1 - 1/np.exp(1)))
        num_new_train_pairs = new_train_size - len(self.train_data_all)


        new_train_data = data[:num_new_train_pairs]
        new_val_data = data[num_new_train_pairs:]
        
        # Keeping all the samples
        self.val_data_all.extend(new_val_data)
        self.train_data_all.extend(new_train_data)
        self.current_size += len(data)
        self.total_labels += len(data)



        # Only recent samples are available for training
        # such that total training data size <= max_size   
        self.train_data = self.train_data_all[-self.train_max_size:]
        self.val_data = self.val_data_all[-self.val_max_size:]

        if self.current_size > self.max_size + 100:
            self.val_data_all = self.val_data.copy()
            self.train_data_all = self.train_data.copy()
            self.current_size = self.max_size

       
    def sample_batch(self, n):
        return random.sample(self.train_data, n)

    def val_iter(self):
        'iterator over validation set'
        return iter(self.val_data)


    @property
    def loss_lb(self):
        '''Train set loss lower bound'''
        even_pref_freq = np.mean([label == 0.5 for (c1, c2, label) in self.train_data])

        #taking into account that label noize is used
        return -((1 - even_pref_freq) * np.log(0.95) + even_pref_freq * np.log(0.5))

    @property
    def val_loss_lb(self):
        '''Validation set loss lower bound'''
        even_pref_freq = np.mean([label == 0.5 for (c1, c2, label) in self.val_data])

        #taking into account that label noize is used
        return -((1 - even_pref_freq) * np.log(0.95) + even_pref_freq * np.log(0.5))

    def get_all_pairs(self):
        '''
        Used to normalize the reward model
        '''
        return self.train_data_all + self.val_data_all

class RewardNet(nn.Module):
    """Here we set up a callable reward model
    Should have batch normalizatoin and dropout on conv layers
    
    """
    def __init__(self, l2 = 0.01, dropout = 0.2, env_type = 'procgen'):
        super().__init__()
        self.env_type = env_type
        if env_type == 'procgen':
            self.model = nn.Sequential(
                #conv1
                nn.Dropout2d(p=dropout),
                nn.Conv2d(3, 16, 3, stride=1),
                nn.MaxPool2d(4, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                #conv2
                nn.Dropout2d(p=dropout),
                nn.Conv2d(16, 16, 3, stride=1),
                nn.MaxPool2d(4, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                #conv3
                nn.Dropout2d(p=dropout),
                nn.Conv2d(16, 16, 3, stride=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                # 2 layer mlp
                nn.Flatten(),
                nn.Linear(11*11*16, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
            )
        elif env_type == 'atari':
            self.model = nn.Sequential(
                #conv1
                nn.Conv2d(4, 16, 7, stride=3),
                nn.BatchNorm2d(16, momentum = 0.01),
                nn.LeakyReLU(),
                nn.Dropout2d(p=dropout),
                #conv2
                nn.Conv2d(16, 16, 5, stride=2),
                nn.BatchNorm2d(16, momentum = 0.01),
                nn.LeakyReLU(),
                nn.Dropout2d(p=dropout),
                #conv3
                nn.Conv2d(16, 16, 3, stride=1),
                nn.BatchNorm2d(16, momentum = 0.01),
                nn.LeakyReLU(),
                nn.Dropout2d(p=dropout),
                #conv4
                nn.Conv2d(16, 16, 3, stride=1),
                nn.BatchNorm2d(16, momentum = 0.01),
                nn.LeakyReLU(),
                nn.Dropout2d(p=dropout),
                # 2 layer mlp
                nn.Flatten(),
                nn.Linear(7*7*16, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
            )

        self.mean = 0
        self.std = 0.05
        self.l2 = l2

    # def forward(self, clip):
    #     '''
    #     predicts the (!) unnormalized sum of rewards for a given clip
    #     used only for assigning preferences, so normalization is unnecesary
    #     '''
    #     # if self.env_type == 'procgen':
    #     clip = clip.permute(0, 3, 1, 2)

    #     # normalizing observations to be in [0,1] and adding noize
    #     clip = clip / 255.0 #+ clip.new(clip.size()).normal_(0,0.1)

    #     return torch.sum(self.model(clip))

    def forward(self, x):
        # if self.env_type == 'procgen':
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float().to(torch.device('cuda:0'))
            
        x = x.permute(0,3,1,2).float()

        # we don't add noize during evaluation
        x = x / 255.0

        # normalizing output to be 0 mean, 0.05 std over the annotation buffer
        rewards = self.model(x)
        rewards = (0.05 / self.std) * (rewards - self.mean) 

        if self.training:
            out = torch.sum(rewards)
        else:
            out = torch.squeeze(rewards).detach().cpu().numpy()

        return out

    def rew_fn(self, x):
        return self.forward(x)

    def save(self, path):
        torch.save(self.model, path)

    def set_mean_std(self, pairs, device = 'cuda:0'):
        '''
        computes the mean and std over provided pairs data,
        and sets the mean and std properties
        '''
        self.eval()
        rewards = []
        for clip0, clip1 , label in pairs:
            rewards.extend(self.forward(clip0))
            rewards.extend(self.forward(clip1))

        unnorm_rewards = (self.std / 0.05) * np.array(rewards) + self.mean
        self.mean, self.std = np.mean(unnorm_rewards), np.std(unnorm_rewards)


class RewardEnsemble:
    """Ensemble of reward predictors as described in the paper (Section 2.2.3).

    Each predictor is trained on |D| samples drawn with replacement from D.
    The estimate r_hat is defined by independently normalizing each predictor
    and then averaging the results.
    """

    def __init__(self, n_members=3, l2=0.01, dropout=0.2, env_type='procgen'):
        self.n_members = n_members
        self.members = [RewardNet(l2=l2, dropout=dropout, env_type=env_type) for _ in range(n_members)]

    def __call__(self, x):
        """Average predictions across independently-normalized ensemble members."""
        preds = [member(x) for member in self.members]
        return np.mean(preds, axis=0)

    def to(self, device):
        for member in self.members:
            member.to(device)
        return self

    def train(self):
        for member in self.members:
            member.train()

    def eval(self):
        for member in self.members:
            member.eval()

    def parameters(self):
        """Yield all parameters from all members (for saving/loading)."""
        for member in self.members:
            yield from member.parameters()

    def save(self, path):
        torch.save([m.model for m in self.members], path)

    def set_mean_std(self, pairs, device='cuda:0'):
        for member in self.members:
            member.set_mean_std(pairs, device)

    @property
    def l2(self):
        return self.members[0].l2

    @l2.setter
    def l2(self, value):
        for member in self.members:
            member.l2 = value

    def predict_returns_per_member(self, clip):
        """Return per-member predicted returns for a clip (used for disagreement-based query selection)."""
        return [member(clip) for member in self.members]


def rm_loss_func(ret0, ret1, label, device = 'cuda:0'):
    '''custom loss function, to allow for float labels
    unlike in nn.CrossEntropyLoss'''

    #compute log(p1), log(p2) where p_i = exp(ret_i) / (exp(ret_1) + exp(ret_2))
    sm = nn.Softmax(dim = 0)
    preds = sm(torch.stack((ret0, ret1)))
    #getting log of predictions after adding label noize
    log_preds = torch.log(preds * 0.9 + 0.05)

    #compute cross entropy given the label
    target = torch.tensor([1-label, label]).to(device)
    loss = - torch.sum(log_preds * target)

    return loss

@timeitt
def calc_val_loss(reward_model, data_buffer, device):
    '''
    computes average loss over the validation set
    '''

    loss = 0
    num_pairs = 0
    for clip0, clip1 , label in data_buffer.val_iter():

        ret0 = reward_model(clip0)
        ret1 = reward_model(clip1)
        loss += rm_loss_func(ret0, ret1, label, device).item()
        num_pairs += 1

    av_loss = loss / num_pairs

    return av_loss


@timeitt
def train_reward_single(reward_model, optimizer, adaptive, data_buffer, num_samples, batch_size, device = 'cuda:0'):
    '''
    Trains a single reward model for num_batches from data_buffer using bootstrap sampling
    (sampling with replacement from the training data).
    '''
    num_batches = int(num_samples / batch_size)
    reward_model.to(device)
    reward_model.train()
    weight_decay = reward_model.l2
    av_loss = 0
    val_loss = calc_val_loss(reward_model, data_buffer, device)
    losses = []

    print(f'Vall loss: {val_loss:6.4f}, Vall loss LB : {data_buffer.val_loss_lb:6.4f},')
    for batch_i in range(1, num_batches + 1):
        # Bootstrap sampling: sample with replacement from the training data
        annotations = [random.choice(data_buffer.train_data) for _ in range(batch_size)]
        loss = 0
        optimizer.zero_grad()

        for clip0, clip1 , label in annotations:

            ret0 = reward_model(clip0)
            ret1 = reward_model(clip1)
            loss += rm_loss_func(ret0, ret1, label, device)

        loss = loss / batch_size
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if batch_i % 100 == 0:

            av_loss = np.mean(losses[-100:])
            if adaptive:

                if val_loss > 1.5 * (av_loss):
                    for g in optimizer.param_groups:
                        g['weight_decay'] = g['weight_decay'] * 1.1
                        weight_decay = g['weight_decay']
                elif val_loss < av_loss * 1.1:
                     for g in optimizer.param_groups:
                        g['weight_decay'] = g['weight_decay'] / 1.1
                        weight_decay = g['weight_decay']
                val_loss = calc_val_loss(reward_model, data_buffer, device)
            else:
                pass

            print(f'batch : {batch_i}, loss : {av_loss:6.4f}, val loss: {val_loss:6.4f},  L2 : {weight_decay:8.6f}')

    reward_model.l2 = weight_decay
    return reward_model, optimizer, (av_loss, val_loss, weight_decay)


@timeitt
def train_reward(ensemble, optimizers, adaptive, data_buffer, num_samples, batch_size, device = 'cuda:0'):
    '''
    Trains the reward model ensemble. Each member is trained on |D| samples
    drawn with replacement from D (bootstrap sampling), as described in the paper
    (Section 2.2.3).
    '''
    all_stats = []
    for i, (member, opt) in enumerate(zip(ensemble.members, optimizers)):
        print(f'\n--- Training ensemble member {i+1}/{ensemble.n_members} ---')
        member, opt, stats = train_reward_single(member, opt, adaptive, data_buffer, num_samples, batch_size, device)
        all_stats.append(stats)

    # Average stats across ensemble members for logging
    avg_train_loss = np.mean([s[0] for s in all_stats])
    avg_val_loss = np.mean([s[1] for s in all_stats])
    avg_l2 = np.mean([s[2] for s in all_stats])

    # Independently normalize each member then set ensemble-level l2
    ensemble.set_mean_std(data_buffer.get_all_pairs())
    ensemble.l2 = avg_l2

    return ensemble, optimizers, (avg_train_loss, avg_val_loss, avg_l2)


    

@timeitt
def train_policy(policy, num_steps, rl_steps, log_name, callback):
    '''
    Trains policy for num_steps
    Returns retrained policy
    '''
    
    # Implementation of the learning rate decay
    policy.learning_rate = 0.0007*(1 - rl_steps/8e7)
    policy._setup_lr_schedule()

    # reset_num timesteps allows having single TB_log when calling .learn() multiple times
    policy.learn(num_steps, reset_num_timesteps=False, tb_log_name=log_name, callback=callback)

    return policy
   

@timeitt
def collect_annotations(venv, policy, num_pairs, clip_size, ensemble, to_cuda = True, device = 'cuda:0'):
    '''
    Collects episodes using the provided policy, slices them to snippets of given length,
    selects pairs using ensemble disagreement (Section 2.2.4) and adds a label based on
    which snippet had larger reward.

    Per the paper: we draw 10x more clip pair candidates than needed, use each reward
    predictor to predict which segment is preferred, then select pairs with highest
    variance across ensemble members.
    '''

    n_envs = venv.num_envs

    clip_pool = []
    obs_stack = []
    # we take a noop step in the environment, instead of doing reset(), because AtariWrapper
    # raises error if you happen to call reset one step before dying

    obs_b, *_ = venv.step(n_envs*[0])

    # Collect 10x as many clips as needed for disagreement-based selection
    while len(clip_pool) < 10 * num_pairs * 2:
        clip_returns = n_envs * [0]
        for _ in range(clip_size):
            # _states are only useful when using LSTM policies
            action_b , _states = policy.predict(obs_b)
            obs_stack.append(obs_b)

            obs_b, r_b, dones, infos = venv.step(action_b)
            clip_returns += r_b

        obs_stack = np.array(obs_stack)
        clip_pool.extend([dict(observations = obs_stack[:, i, :], sum_rews = clip_returns[i]) for i in range(n_envs)])

        obs_stack = []

    # Generate 10x candidate pairs
    n_candidates = min(10 * num_pairs, len(clip_pool) // 2)
    candidate_pairs = np.random.choice(clip_pool, (n_candidates, 2), replace = False)

    # Select pairs with highest ensemble disagreement (Section 2.2.4):
    # For each pair, each ensemble member predicts which segment is preferred,
    # then select pairs with highest variance across members.
    ensemble.eval()
    pair_variances = []
    for clip0, clip1 in candidate_pairs:
        obs0 = torch.tensor(clip0['observations'], device=device, dtype=torch.uint8)
        obs1 = torch.tensor(clip1['observations'], device=device, dtype=torch.uint8)

        # Get per-member predicted returns
        member_prefs = []
        for member in ensemble.members:
            ret0 = member(obs0)
            ret1 = member(obs1)
            # Preference as probability that clip0 is preferred
            member_prefs.append((ret0 - ret1))

        pair_variances.append(np.var(member_prefs))

    # Select top num_pairs by variance
    top_indices = np.argsort(pair_variances)[-num_pairs:]
    selected_pairs = candidate_pairs[top_indices]

    data = []
    for clip0, clip1 in selected_pairs:

        if clip0['sum_rews'] > clip1['sum_rews']:
            label = 0.0
        elif clip0['sum_rews'] < clip1['sum_rews']:
            label = 1.0
        elif clip0['sum_rews'] == clip1['sum_rews']:
            label = 0.5

        if to_cuda:
            clip0 = torch.tensor(clip0['observations'], device = torch.device('cuda:0'), dtype = torch.uint8)
            clip1 = torch.tensor(clip1['observations'], device = torch.device('cuda:0'), dtype = torch.uint8)
            label = torch.tensor(label)
        else:
            clip0 = np.array(clip0['observations'], dtype = np.uint8)
            clip1 = np.array(clip1['observations'], dtype = np.uint8)
            label = np.array(label)

        data.append((clip0, clip1, label))

    return data

def main():
    #check for uncommited changes
    commit_check()

    ##setup args
    parser = argparse.ArgumentParser(description='Reward learning from preferences')

    parser.add_argument('--env_type', type=str, default='atari')
    parser.add_argument('--env_name', type=str, default='BeamRider')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=1)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='LOGS')
    parser.add_argument('--log_prefix', type=str, default='')
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--cpu_buffer', dest = 'on_cuda', action='store_false', help = 'whether to store buffet on cpu or GPU \
                                                                                        by default requires up to 8GB memory on GPU')

    parser.add_argument('--resume_training', action='store_true')

    parser.add_argument('--init_buffer_size', type=int, default=500)
    parser.add_argument('--init_train_size', type=int, default=10**5, help='number of labels to process during initial training of the reward model')
    parser.add_argument('--clip_size', type=int, default=25, help='number of frames in each clip generated for comparison')
    parser.add_argument('--total_timesteps', type=int, default=5*10**7, help='total number of RL timesteps to be taken')
    parser.add_argument('--n_labels', type=int, default=6800, help="total number of labels to collect throughout the training")
    parser.add_argument('--steps_per_iter', type=int, default=5*10**4, help="number of RL steps taken on each iteration")
    parser.add_argument('--pairs_per_iter', type=int, default=5*10**3, help='number of labels the reward model is trained on each iteration')
    parser.add_argument('--pairs_in_batch', type=int, default=16, help='batch size for reward model training')
    parser.add_argument('--l2', type=float, default=0.0001, help='initial l2 regularization for a reward model')
    parser.add_argument('--adaptive', dest='adaptive', action='store_true')
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false')
    parser.set_defaults(adaptive=True)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()

    args.ppo_kwargs = dict(verbose=1, n_steps=256, noptepochs=3, nminibatches = 8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\n Using {device} for training')

    run_dir, monitor_dir, video_dir = setup_logging(args)
    global LOG_TIME
    LOG_TIME = os.path.join(run_dir, "TIME_LOG.txt")


    ### Initializing objects ###
    
    # If resuming some earlier training run - load stored objects
    if args.resume_training:
        args = load_args(args) 
        reward_model, policy, data_buffer, i_num = load_state(run_dir)
           
    
    atari_name = args.env_name + "NoFrameskip-v4"
    venv_fn = lambda: make_atari_continuous(atari_name, n_envs=16)
    annotation_env = make_atari_continuous(atari_name, n_envs=16)  
    annotation_env.reset()
    iter_time = 0

    # In case this is a fresh experiment - initialize fresh objects
    if not args.resume_training:
        store_args(args, run_dir)
        policy = A2C('CnnPolicy', venv_fn(), verbose=1, tensorboard_log="TB_LOGS", ent_coef=0.01, learning_rate = 0.0007,
            policy_kwargs={"optimizer_class" : torch.optim.Adam, "optimizer_kwargs" : {"eps" : 1e-5, "betas" : [.99,.999]}})
        reward_model = RewardEnsemble(n_members=3, l2=args.l2, dropout=args.dropout, env_type=args.env_type)
        data_buffer = AnnotationBuffer()

    # initializing per-member RM optimizers
    rm_optimizers = [optim.Adam(member.parameters(), lr=0.0003, weight_decay=reward_model.l2) for member in reward_model.members]
    
    #creating the environment with reward replaced by the ensemble-averaged prediction
    reward_model.to(device)
    proxy_reward_function = lambda x: reward_model(x)
    proxy_reward_venv = Vec_reward_wrapper(venv_fn(), proxy_reward_function)

    # resetting the environment to avoid raising error from reset_num_timesteps
    proxy_reward_venv.reset()
    policy.set_env(proxy_reward_venv)


    # eval_env_fn = lambda: make_atari_default(atari_name, n_envs=16, seed = 0, vec_env_cls = SubprocVecEnv)
    # video_env_fn= lambda: make_atari_default(atari_name, vec_env_cls = DummyVecEnv)

    # in case this is a fresh run, collect init_buffer_size samples to AnnotationBuffer
    # and train the reward model on init_train_size number of samples with replacement
    if not args.resume_training:
       
        t_start = time.time()
        print(f'================== Initial iter ====================')

        annotations = collect_annotations(annotation_env, policy, args.init_buffer_size, args.clip_size, reward_model, args.on_cuda, device)
        data_buffer.add(annotations)

        print(f'Buffer size = {data_buffer.current_size}')

        reward_model, rm_optimizers, rm_train_stats = train_reward(reward_model, rm_optimizers, args.adaptive, data_buffer, args.init_train_size, args.pairs_in_batch)
        # this callback adds values to TensorBoard logs for easier plotting
        reward_model.eval()
        callback = TensorboardCallback((data_buffer.total_labels, data_buffer.loss_lb, iter_time, rm_train_stats))
        policy = train_policy(policy, args.steps_per_iter, 0,  args.log_name, callback)

        save_state(run_dir, 0, reward_model, policy, data_buffer)

        true_performance = safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer])

        t_finish = time.time()
        iter_time = t_finish - t_start
        log_iter(run_dir, args.steps_per_iter, data_buffer, true_performance, 0, rm_train_stats, iter_time)
        
        print(f'Iteration took {time.gmtime(t_finish - t_start).tm_min} min {time.gmtime(t_finish - t_start).tm_sec} sec')
        
        # i_num is the number of training iterations taken      
        i_num = 1 


    num_iters = int(args.total_timesteps / args.steps_per_iter)
    # calculating the initial number of pairs to collect 
    num_pairs = init_num_pairs = round((args.n_labels - args.init_buffer_size) / 0.239 / num_iters) 

    print('init_num_pairs = {}'.format(init_num_pairs))
    for i in range(i_num, num_iters):
        t_start = time.time()
        print(f'================== iter : {i} ====================')

        rl_steps = i * args.steps_per_iter
        # decaying the number of pairs to collect
        num_pairs = round(init_num_pairs / (rl_steps/(args.total_timesteps/10) + 1))

        annotations = collect_annotations(annotation_env, policy, num_pairs, args.clip_size, reward_model, args.on_cuda, device)
        data_buffer.add(annotations)

        print(f'Buffer size = {data_buffer.current_size}')

        reward_model, rm_optimizers, rm_train_stats = train_reward(reward_model, rm_optimizers, args.adaptive, data_buffer, args.pairs_per_iter, args.pairs_in_batch)

        #TODO : pretify passing data to callback
        callback = TensorboardCallback((data_buffer.total_labels, data_buffer.loss_lb, iter_time, rm_train_stats))
        policy = train_policy(policy, args.steps_per_iter, rl_steps, args.log_name, callback)

        # storing the state every 1M steps
        # this assumes that steps_per_iter devides 10**6
        if rl_steps % (10**6) == 0:
            save_state(run_dir, i, reward_model, policy, data_buffer)

        # record_video(policy, video_env_fn(), video_dir, 4000, f"{i}_ITER00_{args.env_name}")
        # true_performance = eval_policy(venv_fn(), policy, n_eval_episodes=50)
        # proxy_performance = eval_policy(test_env, policy, n_eval_episodes=50)

        true_performance = safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer])

        # print(f'True policy preformance = {true_performance}') 
        # print(f'Proxy policy preformance = {proxy_performance}') 

        t_finish = time.time()
        iter_time = t_finish - t_start
        log_iter(run_dir, rl_steps, data_buffer, true_performance, 0, rm_train_stats, iter_time) 
        
        if LOG_TIME:
            with open(LOG_TIME, 'a') as f:
                f.write(f'Iteration took {time.gmtime(iter_time).tm_min} min {time.gmtime(iter_time).tm_sec} sec\n')
                f.write(f'================== iter : {i+1} ====================\n')
        else:
            print(f'Iteration took {time.gmtime(iter_time).tm_min} min {time.gmtime(iter_time).tm_sec} sec')
     

if __name__ == '__main__':
    main()

