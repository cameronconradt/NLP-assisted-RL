from gym import envs
import random
import imageio
import multiprocessing

def get_screenshots(list):
    for i in list:
        if 'atari' in i.entry_point:
            env = envs.make(i.id)
            env.reset()
            print(i.id)
            for j in range(10000):
                n_actions = env.action_space.n
                action = random.randrange(n_actions)
                _, _, done, _ = env.step(action)
                if done:
                    env.reset()
                if j % 100 == 0:
                    imageio.imwrite('screenshots/' + i.id + '_' + str(j) + '.jpg', env.render(mode='rgb_array'))

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


games = list(envs.registry.all())
chunked = divide_chunks(games, multiprocessing.cpu_count() // 2)
threads = []

for i in chunked:
    thread = multiprocessing.Process(target=get_screenshots, args=[i])
    threads.append(thread)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()


