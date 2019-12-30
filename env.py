import torch
from transformers import RobertaTokenizer
from gym.envs.atari.atari_env import AtariEnv
import torchvision.transforms

class Env(AtariEnv):
    def __init__(self, args, game, nlp_dict, yolo_model):
        self.__dict__.update(locals())
        self.game_name = game
        super(Env, self).__init__(game, obs_type='image')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def step(self, action):
        _, reward, done, _ = super(Env, self).step(action)
        return self._get_image(), reward, done

    def _get_image(self):
        state = super(Env, self)._get_image().transpose((2, 0, 1))
        state = torch.from_numpy(state).float().cuda() / 255
        predictions = self.yolo_model.get_obj(state)
        invalid = torch.zeros_like(self.nlp_dict[list(self.nlp_dict.keys())[0]]).float().cuda()
        toReturn_nlp = []
        toReturn_img = []
        max_dim = 16
        keys = list(predictions.keys())
        for i in range(max_dim):
            if len(keys) <= i:
                toReturn_img.append(torch.zeros((1, 210, 160)).cuda().float())
                toReturn_nlp.append(invalid)
            else:
                key = keys[i]
                if key in self.nlp_dict.keys():
                    with torch.no_grad():
                        toReturn_nlp.append(self.nlp_dict[key])
                    toReturn_img.append(predictions[key].cuda().float())
                else:
                    for i in range(list(predictions[key].size())[0]):
                        toReturn_nlp.append(invalid)
                    toReturn_img.append(predictions[key].cuda().float())

        return [torch.cat(toReturn_nlp).unsqueeze(0), torch.cat(toReturn_img).unsqueeze(0)]

