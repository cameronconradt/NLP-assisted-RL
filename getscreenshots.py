from gym import envs
import random
import imageio

for i in envs.registry.all():
    if 'atari' in i.entry_point:
        env = envs.make(i.id)
        env.reset()
        for j in range(1000):
            n_actions = env.action_space.n
            action = random.randrange(n_actions)
            env.step(action)
            if j % 100 == 0:
                image = env.render(mode='rgb_array')
                imageio.imwrite('screenshots/' + i.id + '_' + str(j) + '.jpg', image)
