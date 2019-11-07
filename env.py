import torch
from gym.envs.atari.atari_env import AtariEnv
import torchvision.transforms

class Env(AtariEnv):
    def __init__(self, args, game, model, nlp_dict, yolo_model):
        self.__dict__.update(locals())
        super(Env, self).__init__(game, obs_type='image')

    def step(self, action):
        _, reward, done, _ = super(Env, self).step(action)
        return self._get_image(), reward, done

    def _get_image(self):
        state = super(Env, self)._get_image().transpose((2, 0, 1))
        state = torch.from_numpy(state).float().cuda() / 255
        predictions = self.yolo_model.get_obj(state)
        invalid = self.nlp_dict[list(self.nlp_dict.keys())[0]].cuda()
        invalid = self.model(invalid)
        invalid = torch.zeros_like(invalid[0]).float()
        toReturn_nlp = []
        toReturn_img = []
        for key in predictions.keys():
            if key in self.nlp_dict.keys():
                toReturn_nlp.append(self.model(self.nlp_dict[key].cuda())[0].float())
                toReturn_img.append(predictions[key].cuda().float())
            else:
                for i in range(list(predictions[key].size())[0]):
                    toReturn_nlp.append(invalid)
                toReturn_img.append(predictions[key].cuda().float())

        return [torch.cat(toReturn_nlp), torch.cat(toReturn_img).unsqueeze(0)]

