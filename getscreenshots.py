from gym import envs
import random
import imageio
import multiprocessing
import os


def get_screenshots(list):
    for i in list:
        if 'atari' in i.entry_point:
            env = envs.make(i.id)
            env.reset()
            print(i.id)
            for j in range(100000):
                n_actions = env.action_space.n
                action = random.randrange(n_actions)
                _, _, done, _ = env.step(action)
                if done:
                    env.reset()
                if j % 1000 == 0:
                    imageio.imwrite('screenshots/' + i.id + '_' + str(j) + '.jpg', env.render(mode='rgb_array'))


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


games_to_get = ['Asteroids', 'Bowling', 'Breakout', 'Freeway', 'MsPacman']
os.makedirs("screenshots", exist_ok=True)
games = list(envs.registry.all())
gamename_list = []
for i in games:
    if 'atari' in i.entry_point and i._env_name not in gamename_list and i._env_name in games_to_get and 'v4' in i.id:
        gamename_list.append(i)
chunked = divide_chunks(gamename_list, multiprocessing.cpu_count())
threads = []

for i in gamename_list:
    thread = multiprocessing.Process(target=get_screenshots, args=[[i]])
    threads.append(thread)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()


