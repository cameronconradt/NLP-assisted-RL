import argparse

from torch import optim

from env import Env
import time
from yolo import Yolo
import pickle
from dqn import DQN
import numpy as np
from memory import ReplayMemory
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
# Arguments used for Yolo
parser.add_argument("--model_def", type=str, default="data/yolo.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="data/yolo.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0, help="iou threshold for non-maximum suppression")

# Arguments used for NLP
parser.add_argument('--nlp_map_obj', type=str, default='data/nlp_map_roberta_output.obj', help='map used for nlp augmentation')

parser.add_argument("--games_def_path", type=str, default='data/small_set.txt', help='path to txt with list of games to run')
parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to train on per game')
parser.add_argument('--evaluation-size', type=int, default=103, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--batch-size', type=int, default=50, help='size of batch to pull from memory')
parser.add_argument('--episode-length', type=int, default=100000,  help='max number of steps per episode')

args = parser.parse_args()
print(args)


def optimize_model(memory, batch_size, Transition, policy, target, optimizer):
    GAMMA = 0.999
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if memory.len(env.game_name) < batch_size:
        return
    transitions = memory.sample(batch_size, env.game_name)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    state_nlp = torch.cat([x[0] for x in batch.state]).cuda()
    state_img = torch.cat([x[1] for x in batch.state]).cuda()
    action = torch.tensor([x for x in batch.action], dtype=torch.long).cuda()
    next_state_nlp = torch.cat([x[0] for x in batch.next_state]).cuda()
    next_state_img = torch.cat([x[1] for x in batch.next_state]).cuda()
    reward = torch.tensor([x for x in batch.reward], dtype=torch.float).cuda()
    done = torch.tensor([x for x in batch.done], dtype=torch.float).cuda()
    optimizer.zero_grad()
    values = policy_net((state_nlp, state_img)).squeeze().gather(1, action.unsqueeze(1))
    target_vals = reward + GAMMA * (1-done) * torch.max(target_net((next_state_nlp, next_state_img)).squeeze(), 1)[0]
    loss = torch.mean((target_vals - values) ** 2)
    del target, state_nlp, state_img, action, next_state_img, next_state_nlp, reward, done
    loss.backward()
    optimizer.step()
    if global_steps % 10000 == 0:
        target_net.load_state_dict(policy_net.state_dict())


def plot(total_rewards, final=False):
    for key, data_list in total_rewards.items():
        if len(data_list) > 0:
            plt.plot(data_list, label=key)
    plt.legend()
    if not final:
        plt.savefig('rewards_temp.png')
        pickle.dump(total_rewards, open('rewards_temp.obj', 'wb'))
    else:
        plt.savefig('final_rewards.png')
        pickle.dump(tot_rewards, open('final_rewards.obj', 'wb'))
    plt.show()


start = time.time()
nlp_dict = pickle.load(open(args.nlp_map_obj, 'rb'))
print("nlp dict loaded")
yolo_model = Yolo(args)
games_file = open(args.games_def_path, 'rt')
policy_net = None
envs = []
val_mem = ReplayMemory()
objective = torch.nn.MSELoss()
tot_rewards = {}
for line in games_file:
    envs.append(Env(args, line.strip(), nlp_dict, yolo_model))
    val_mem.add_env(line.strip())
    tot_rewards.update({line.strip(): []})
done = True
for env in envs:
    gc.collect()
    for i in tqdm(range(args.evaluation_size), desc='Eval loop'):
        if done:
            state, done = env.reset(), False
        action = np.random.randint(0, env.action_space.n)
        next_state, reward, done = env.step(action)
        val_mem.push([state[0].cpu(), state[1].cpu()], torch.Tensor([action]), [next_state[0].cpu(), next_state[1].cpu()], torch.Tensor([reward]), torch.Tensor([done]), env.game_name)
        state = next_state
        del next_state, reward, action

global_steps = 0
env_pos = 0
for env in envs:
    state = env.reset()
    if policy_net is None:
        policy_net = DQN(state[0], state[1], env.action_space.n).cuda()
        target_net = DQN(state[0], state[1], env.action_space.n).cuda()
        optimizer = optim.Adam(policy_net.parameters())
    else:
        policy_net.update_output(env.action_space.n)
        target_net.update_output(env.action_space.n)
    for episode in tqdm(range(args.num_episodes // len(envs)), desc='Episodes'):
        gc.collect()
        state = env.reset()
        done = False
        rewards = 0
        for step in range(args.episode_length):
            if done:
                done = False
                break
            chosen = policy_net.forward(state)
            chosen = chosen.cpu()
            action = chosen.detach().numpy().argmax()
            next_state, reward, done = env.step(action)
            optimize_model(val_mem, args.batch_size, val_mem.Transition, policy_net, target_net, optimizer)
            val_mem.push([state[0].cpu(), state[1].cpu()], torch.Tensor([action]), [next_state[0].cpu(), next_state[1].cpu()], torch.Tensor([reward]), torch.Tensor([done]), env.game_name)
            state = next_state
            global_steps += 1
            rewards += reward
        tot_rewards[env.game_name].append(rewards)
        plot(tot_rewards)
        env_pos = (env_pos + 1) % len(envs)


plot(tot_rewards, True)
end = time.time()
print('create each iter', end - start)
