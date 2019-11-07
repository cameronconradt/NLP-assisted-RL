import argparse

from torch import optim

from env import Env
import time
from pytorch_transformers import RobertaModel
from yolo import Yolo
import pickle
from dqn import DQN
import numpy as np
from itertools import count
from memory import ReplayMemory
import torch
import torch.nn.functional as F
from tqdm import tqdm
from bz2 import BZ2File

parser = argparse.ArgumentParser()
# Arguments used for Yolo
parser.add_argument("--model_def", type=str, default="data/yolo.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="data/yolo.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0, help="iou threshold for non-maximum suppression")

# Arguments used for NLP
parser.add_argument('--nlp_map_obj', type=str, default='data/output_condensed.obj', help='map used for nlp augmentation')

parser.add_argument("--games_def_path", type=str, default='data/small_set.txt', help='path to txt with list of games to run')
parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes to train on per game')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--batch-size', type=int, default=50, help='size of batch to pull from memory')

args = parser.parse_args()
print(args)


def optimize_model(memory, batch_size, Transition, policy, target, optimizer):
    GAMMA = 0.999
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states_nlp = torch.stack([s[0] for s in batch.next_state
                                       if s is not None])
    non_final_next_states_img = torch.stack([s[1] for s in batch.next_state
                                       if s is not None])
    state_batch_nlp = torch.cat([s[0] for s in batch.state
                                       if s is not None])
    state_batch_img = torch.cat([s[1] for s in batch.state
                                       if s is not None])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy([state_batch_nlp, state_batch_img]).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states_nlp are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target([non_final_next_states_nlp, non_final_next_states_img]).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


start = time.time()
roberta_model = RobertaModel.from_pretrained('roberta-base').cuda()
nlp_dict = pickle.load(open(args.nlp_map_obj, 'rb'))
yolo_model = Yolo(args)
games_file = open(args.games_def_path, 'rt')
policy_net = None
envs = []
val_mem = ReplayMemory()
for line in games_file:
    envs.append(Env(args, line.strip(), roberta_model, nlp_dict, yolo_model))
done = True
for i in tqdm(range(args.evaluation_size), desc='Eval loop'):
    env = envs[np.random.randint(0, len(envs))]
    if done:
        state, done = env.reset(), False
    action = np.random.randint(0, env.action_space.n)
    next_state, reward, done = env.step(action)
    next_state = [next_state[0].cpu(), next_state[1].cpu()]
    val_mem.push(state, action, next_state, reward)

for episode in tqdm(range(args.num_episodes), desc='Episode Loop'):
    env = envs[np.random.randint(0, len(envs))]
    state = env.reset()
    if policy_net is None:
        policy_net = DQN(state[0], state[1], env.action_space.n).cuda()
        target_net = DQN(state[0], state[1], env.action_space.n).cuda()
        optimizer = optim.Adam(policy_net.parameters())
    else:
        policy_net.update_output(state[0], state[1], env.action_space.n)
        target_net.update_output(state[0], state[1], env.action_space.n)

    done = False
    for i in count():
        if done:
            break
        action = policy_net.forward(state)
        next_state, reward, done = env.step(action)
        # optimize_model(val_mem, args.batch_size, val_mem.Transition, policy_net, target_net, optimizer)

        state = next_state
        policy_net.update_output(state[0], state[1], env.action_space.n)
        target_net.update_output(state[0], state[1], env.action_space.n)

end = time.time()
print('create each iter', end - start)
