import gym
import gym_pcgrl

import numpy as np
import math
import os

import pdb

# render obs array as a string
render = lambda obs:print('\n'.join(["".join([str(i) for i in obs[j,:,0]]) for j in range(obs.shape[0])]))
# clean the input action
get_action = lambda a: a.item() if hasattr(a, "item") else a
# unwrap all the environments and get the PcgrlEnv
get_pcgrl_env = lambda env: env if "PcgrlEnv" in str(type(env)) else get_pcgrl_env(env.env)
# for the guassian attention
pdf = lambda x,mean,sigma: math.exp(-1/2 * math.pow((x-mean)/sigma,2))/math.exp(0)

"""
Return a Box instead of dictionary by stacking different similar objects

Can be stacked as Last Layer
"""
class ToImage(gym.Wrapper):
    def __init__(self, game, names, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
        self.shape = None
        depth=0
        max_value = 0
        for n in names:
            assert n in self.env.observation_space.spaces.keys(), 'This wrapper only works if your observation_space is spaces.Dict with the input names.'
            if self.shape == None:
                self.shape = self.env.observation_space[n].shape
            new_shape = self.env.observation_space[n].shape
            depth += 1 if len(new_shape) <= 2 else new_shape[2]
            assert self.shape[0] == new_shape[0] and self.shape[1] == new_shape[1], 'This wrapper only works when all objects have same width and height'
            if self.env.observation_space[n].high.max() > max_value:
                max_value = self.env.observation_space[n].high.max()
        self.names = names

        self.observation_space = gym.spaces.Box(low=0, high=max_value,shape=(self.shape[0], self.shape[1], depth))

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        final = np.empty([])
        for n in self.names:
            if len(final.shape) == 0:
                final = obs[n].reshape(self.shape[0], self.shape[1], -1)
            else:
                final = np.append(final, obs[n].reshape(self.shape[0], self.shape[1], -1), axis=2)
        return final

"""
Return a single array with all in it

can be stacked as Last Layer
"""
class ToFlat(gym.Wrapper):
    def __init__(self, game, names, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
        length=0
        max_value=0
        for n in names:
            assert n in self.env.observation_space.spaces.keys(), 'This wrapper only works if your observation_space is spaces.Dict with the input names.'
            new_shape = self.env.observation_space[n].shape
            length += np.prod(new_shape)
            if self.env.observation_space[n].high.max() > max_value:
                max_value = self.env.observation_space[n].high.max()
        self.names = names
        self.observation_space = gym.spaces.Box(low=0, high=max_value, shape=(length,))

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        concatenations = []
        for n in self.names:
            concatenations.append(obs[n].flatten())
        return np.concatenate(concatenations)

"""
Transform any object in the dictionary to one hot encoding

can be stacked
"""
class OneHotEncoding(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a {} key'.format(name)
        self.name = name

        self.observation_space = gym.spaces.Dict({})
        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        new_shape = []
        shape = self.env.observation_space[self.name].shape
        self.dim = self.observation_space[self.name].high.max() - self.observation_space[self.name].low.min() + 1
        for v in shape:
            new_shape.append(v)
        new_shape.append(self.dim)
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.uint8)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        old = obs[self.name]
        obs[self.name] = np.eye(self.dim)[old]
        return obs

"""
Adding sum of the heatmap to the observation grid

can be stacked
"""
class AddChanges(gym.Wrapper):
    def __init__(self, game, is_map=True, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'heatmap' in self.env.observation_space.spaces.keys(), 'Heatmap does not exists in the environment space.'
        heatmap_obs = self.observation_space.spaces['heatmap']
        self.is_map = is_map

        if not self.is_map:
            self.observation_space.spaces['changes'] = gym.spaces.Box(low=np.array([heatmap_obs.low.min()]), high=np.array([heatmap_obs.high.max()]), dtype=heatmap_obs.dtype)
        else:
            self.observation_space.spaces['changes'] = gym.spaces.Box(low=heatmap_obs.low.min(), high=heatmap_obs.high.max(), shape=heatmap_obs.shape, dtype=heatmap_obs.dtype)

    def reset(self):
        obs = self.env.reset()
        return self.transform(obs)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def transform(self, obs):
        obs['changes'] = obs['heatmap'].sum()
        if self.is_map:
            obs['changes'] = np.full(obs['heatmap'].shape, obs['changes'])
        return obs

"""
Returns reward at the end of the episode

Can be stacked
"""
class LateReward(gym.Wrapper):
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
        self.acum_reward = 0

    def reset(self):
        self.acum_reward = 0
        return self.env.reset()

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        self.acum_reward += reward
        reward=[0,self.acum_reward][done]
        return obs, reward, done, info

"""
Transform the input space to a 3D map of values where the argmax value will be applied

can be stacked
"""
class ActionMap(gym.Wrapper):
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'map' in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a map key'
        self.old_obs = None
        self.one_hot = len(self.env.observation_space['map'].shape) > 2
        w, h, dim = 0, 0, 0
        if self.one_hot:
            h, w, dim = self.env.observation_space['map'].shape
        else:
            h, w = self.env.observation_space['map'].shape
            dim = self.env.observation_space['map'].high.max()
        self.h = h
        self.w = w
        self.dim = self.env.get_num_tiles()
       #self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(h,w,dim))
        self.action_space = gym.spaces.Discrete(h*w*self.dim)

    def reset(self):
        self.old_obs = self.env.reset()
        return self.old_obs

    def step(self, action):
       #y, x, v = np.unravel_index(np.argmax(action), action.shape)
        y, x, v = np.unravel_index(action, (self.h, self.w, self.dim))
        if 'pos' in self.old_obs:
            o_x, o_y = self.old_obs['pos']
            if o_x == x and o_y == y:
                obs, reward, done, info = self.env.step(v)
            else:
                o_v = self.old_obs['map'][o_y][o_x]
                if self.one_hot:
                    o_v = o_v.argmax()
                obs, reward, done, info = self.env.step(o_v)
        else:
            obs, reward, done, info = self.env.step([x, y, v])
        self.old_obs = obs
        return obs, reward, done, info

"""
Add visited map to the observation which help keep track of all the tiles that
have been visited but not changed. It was designed to battle the need for having
a memory for the turtle representation.

can be stacked
"""
class VisitedMap(gym.Wrapper):
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'map' in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a map key'
        shape = self.env.observation_space.spaces["map"].shape
        if len(shape) > 2:
            (self._h, self._w, _) = shape
        else:
            (self._h, self._w) = shape
        self._has_pos = 'pos' in self.env.observation_space.spaces.keys()
        max_iterations = get_pcgrl_env(self.env)._max_iterations

        self.observation_space.spaces['visits'] = gym.spaces.Box(low=0, high=max_iterations+1, shape=(self._h, self._w), dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        self._visits = np.zeros((self._h, self._w))
        if self._has_pos:
            (x, y) = obs["pos"]
            self._visits[y][x] += 1
        return self.transform(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not done:
            if self._has_pos:
                (x, y) = obs["pos"]
            else:
                (x, y) = (action[0], action[1])
            self._visits[y][x] += 1
        obs = self.transform(obs)
        return obs, reward, done, info

    def transform(self, obs):
        obs['visits'] = self._visits.copy()
        return obs

"""
Normalize a certain attribute by the max and min values of its observation_space

can be stacked
"""
class Normalize(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a {} key'.format(name)
        self.name = name
        self.low = self.env.observation_space[self.name].low
        self.high = self.env.observation_space[self.name].high

        self.observation_space = gym.spaces.Dict({})
        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        shape = self.observation_space[self.name].shape
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        test = obs[self.name]
        obs[self.name] = (test - self.low).astype(np.float32) / (self.high - self.low).astype(np.float32)
        obs[self.name] = np.mod(obs[self.name], 1)
        return obs

"""
Inverse the values a certain attribute of its observation_space

can be stacked
"""
class Inverse(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a {} key'.format(name)
        self.name = name
        self.low = self.env.observation_space[self.name].low
        self.high = self.env.observation_space[self.name].high

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        test = obs[self.name]
        obs[self.name] = self.high - test + self.low
        return obs

"""
Crops and centers the view around the agent and replace the map with cropped version
The crop size can be larger than the actual view, it just pads the outside
This wrapper only works on games with a position coordinate

can be stacked
"""
class Cropped(gym.Wrapper):
    def __init__(self, game, crop_size, pad_value, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'pos' in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a position'
        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a {} key'.format(name)
        assert len(self.env.observation_space.spaces[name].shape) == 2, "This wrapper only works on 2D arrays."
        self.name = name
        self.size = crop_size
        self.pad = crop_size//2
        self.pad_value = pad_value

        self.observation_space = gym.spaces.Dict({})
        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        high_value = self.observation_space[self.name].high.max()
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=high_value, shape=(crop_size, crop_size), dtype=np.uint8)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        map = obs[self.name]
        x, y = obs['pos']

        #View Centering
        padded = np.pad(map, self.pad, constant_values=self.pad_value)
        cropped = padded[y:y+self.size, x:x+self.size]
        obs[self.name] = cropped

        return obs

"""
Add a 2D map with a window of 1s as pos key instead of normal pos
This wrapper only works on games with a position coordinate

can be stacked
"""
class PosImage(gym.Wrapper):
    def __init__(self, game, pos_size=1, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'pos' in self.env.observation_space.spaces.keys(), 'This wrapper only works for views that have a position'
        assert 'map' in self.env.observation_space.spaces.keys(), 'This wrapper only works for views that have a map'
        shape = self.env.observation_space.spaces["map"].shape
        if len(shape) > 2:
            x, y, _ = shape
        else:
            x, y = shape
        self.size = pos_size
        self.pad = pos_size//2
        self.observation_space = gym.spaces.Dict({})
        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        self.observation_space.spaces['pos'] = gym.spaces.Box(low=0, high=1, shape=(x, y), dtype=np.float32)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        map = obs['map']
        x, y = obs['pos']

        pos = np.zeros_like(map)
        low_y,high_y=np.clip(y-self.pad,0,map.shape[0]),np.clip(y+(self.size-self.pad),0,map.shape[0])
        low_x,high_x=np.clip(x-self.pad,0,map.shape[1]),np.clip(x+(self.size-self.pad),0,map.shape[1])
        pos[low_y:high_y,low_x:high_x] = 1

        obs['pos'] = pos
        return obs

"""
Add the ability to load nice levels that the system

can be stacked
"""
class BootStrapping(gym.Wrapper):
    def __init__(self, game, folder_loc, max_files=100, tries_to_age=10, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        self.pcgrl_env = get_pcgrl_env(self.env);
        self.pcgrl_env.adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        self.old_map = None
        self.total_reward = 0
        self.new_run = True
        self.random_run = False
        self.folder_loc = folder_loc
        self.max_files = max_files
        self.tries_to_age = tries_to_age
        if not os.path.exists(self.folder_loc):
            os.makedirs(self.folder_loc)
        self.current_index = 0
        self.file_age = [0]*self.max_files
        self.file_tries = [0]*self.max_files

    def reset(self):
        self.new_run = True
        self.random_run = False
        self.total_reward = 0
        files = [f for f in os.listdir(self.folder_loc) if "map" in f]
        if len(files) >= self.max_files:
            if self.pcgrl_env._rep._random.random() < 0.7:
                self.random_run = True
            else:
                self.current_index = self.pcgrl_env._rep._random.randint(self.max_files)
                good_map = np.load(os.path.join(self.folder_loc, "map_{}.npy".format(self.current_index)))
                self.pcgrl_env._rep._old_map = good_map
                self.pcgrl_env._rep._random_start = False
        obs = self.env.reset()
        self.old_map = self.pcgrl_env._rep._map
        self.pcgrl_env._rep._random_start = True
        if len(files) < self.max_files:
            if os.path.exists(os.path.join(self.folder_loc, "map_{}.npy".format(self.current_index))):
                self.current_index += 1
                if self.current_index > self.max_files:
                    self.current_index -= self.max_files
            np.save(os.path.join(self.folder_loc, "map_{}".format(self.current_index)), self.old_map)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_reward += reward
        if done and self.new_run and not self.random_run:
            self.new_run = False
            if self.total_reward > max(self.tries_to_age - self.file_age[self.current_index], 0):
                self.file_age[self.current_index] += 1
                np.save(os.path.join(self.folder_loc, "map_{}".format(self.current_index)), self.old_map)
            else:
                self.file_tries[self.current_index] += 1
                if self.file_tries[self.current_index] - self.file_age[self.current_index] > self.tries_to_age:
                    self.file_tries[self.current_index] = 0
                    self.file_age[self.current_index] = 0
                    os.remove(os.path.join(self.folder_loc, "map_{}.npy".format(self.current_index)))
        self.old_map = self.pcgrl_env._rep._map
        return obs, reward, done, info

"""
Similar to the Image Wrapper but the values in the image
are sampled from gaussian distribution

Can be stacked
"""
class PosGaussianImage(PosImage):
    def __init__(self, game, pos_size=5, guassian_std=1, **kwargs):
        Image.__init__(self, game, pos_size, **kwargs)
        assert guassian_std > 0, 'gaussian distribution need positive standard deviation'
        self.guassian = guassian_std

    def transform(self, obs):
        shape = obs['map'].shape
        pos_x, pos_y = obs['pos']
        obs = Image.transform(self, obs)
        for y in range(min(self.pad + 1,shape[0]//2+1)):
            for x in range(min(self.pad + 1,shape[1]//2+1)):
                value = pdf(np.linalg.norm(np.array([x, y])), 0, self.guassian)
                obs_y, obs_x = pos_y+y,pos_x+x
                if obs_y >= 0 and obs_x >= 0 and obs_y < shape[1] and obs_x < shape[0]:
                    obs['pos'][obs_y][obs_x][1] *= value
                obs_y, obs_x = pos_y-y,pos_x+x
                if obs_y >= 0 and obs_x >= 0 and obs_y < shape[1] and obs_x < shape[0]:
                    obs['pos'][obs_y][obs_x][1] *= value
                obs_y, obs_x = pos_y+y,pos_x-x
                if obs_y >= 0 and obs_x >= 0 and obs_y < shape[1] and obs_x < shape[0]:
                    obs['pos'][obs_y][obs_x][1] *= value
                obs_y, obs_x = pos_y-y,pos_x-x
                if obs_y >= 0 and obs_x >= 0 and obs_y < shape[1] and obs_x < shape[0]:
                    obs['pos'][obs_y][obs_x][1] *= value
        return obs

################################################################################
#   Final used wrappers for the experiments
################################################################################

"""
The wrappers we use for narrow and turtle experiments
"""
class CroppedImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, crop_size, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Cropping the map to the correct crop_size
        env = Cropped(self.pcgrl_env, crop_size, self.pcgrl_env.get_border_tile(), 'map')
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Indices for flatting
        flat_indeces = ['map']
        # Cropping the heatmap similar to the map
        if kwargs.get('add_heatmap', True):
            env = Cropped(env, crop_size, 0, 'heatmap')
            env = Normalize(env, 'heatmap')
            flat_indeces.append('heatmap')
        # Adding changes to the channels
        if kwargs.get('add_changes', True):
            env = AddChanges(env, True)
            env = Cropped(env, crop_size, 0, 'changes')
            env = Normalize(env, 'changes')
            flat_indeces.append('changes')
        # Adding Visited Map
        if kwargs.get('add_visits', True):
            env = VisitedMap(env)
            env = Cropped(env, crop_size, 0, 'visits')
            env = Normalize(env, 'visits')
            flat_indeces.append('visits')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indeces)
        gym.Wrapper.__init__(self, self.env)

"""
This wrapper ignore location data, pretty useful with wide representation
"""
class ImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Indices for flatting
        flat_indeces = ['map']
        env = self.pcgrl_env
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Normalize the heatmap
        if kwargs.get('add_heatmap', True):
            env = Normalize(env, 'heatmap')
            flat_indeces.append('heatmap')
        # Adding changes to the channels
        if kwargs.get('add_changes', True):
            env = AddChanges(env, True)
            env = Normalize(env, 'changes')
            flat_indeces.append('changes')
        # Adding visited map
        if kwargs.get('add_visits', True):
            env = VisitedMap(env)
            env = Normalize(env, 'visits')
            flat_indeces.append('visits')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indeces)
        gym.Wrapper.__init__(self, self.env)

"""
Similar to the previous wrapper but the input now is the index in a 3D map (height, width, num_tiles) of the highest value
Used for wide experiments
"""
class ActionMapImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Indices for flatting
        flat_indeces = ['map']
        env = self.pcgrl_env
        # Add the action map wrapper
        env = ActionMap(env)
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Normalize the heatmap
        if kwargs.get('add_heatmap', True):
            env = Normalize(env, 'heatmap')
            flat_indeces.append('heatmap')
        # Adding changes to the channels
        if kwargs.get('add_changes', True):
            env = AddChanges(env, True)
            env = Normalize(env, 'changes')
            flat_indeces.append('changes')
        # Adding visited map
        if kwargs.get('add_visits', True):
            env = VisitedMap(env)
            env = Normalize(env, 'visits')
            flat_indeces.append('visits')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indeces)
        gym.Wrapper.__init__(self, self.env)

"""
Instead of cropping we are appending 1s in position layer
"""
class PositionImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, pos_size, guassian_std=0, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Indeces for flatting
        flat_indeces = ['map']
        env = self.pcgrl_env
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Normalize the heatmap
        if kwargs.get('add_heatmap', True):
            env = Normalize(env, 'heatmap')
            flat_indeces.append('heatmap')
        # Adding changes to the channels
        if kwargs.get('add_changes', True):
            env = AddChanges(env, True)
            env = Normalize(env, 'changes')
            flat_indeces.append('changes')
        # Adding visited map
        if kwargs.get('add_visits', True):
            env = VisitedMap(env)
            env = Normalize(env, 'visits')
            flat_indeces.append('visits')
        # Transform the pos to image
        if kwargs.get('add_pos', True):
            if guassian_std > 0:
                env = PosImage(env, pos_size, guassian_std)
            else:
                env = PosImage(env, pos_size)
            flat_indeces.append('pos')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indeces)
        gym.Wrapper.__init__(self, self.env)

"""
Flat input for fully connected layers
"""
class FlatPCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Indeces for flatting
        flat_indeces = ['map']
        env = self.pcgrl_env
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Adding the normalized heatmap
        if kwargs.get('add_heatmap', True):
            env = Normalize(env, 'heatmap')
            flat_indeces.append('heatmap')
        # Adding changes to the channels
        if kwargs.get('add_changes', True):
            env = AddChanges(env, False)
            env = Normalize(env, 'changes')
            flat_indeces.append('changes')
        # Adding visited map
        if kwargs.get('add_visits', True):
            env = VisitedMap(env)
            env = Normalize(env, 'visits')
            flat_indeces.append('visits')
        # Adding the normalized position
        if kwargs.get('add_pos', True):
            if 'pos' in self.pcgrl_env.observation_space.spaces.keys():
                env = Normalize(env, 'pos')
                flat_indeces.append('pos')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToFlat(env, flat_indeces)
        gym.Wrapper.__init__(self, self.env)
