import gym
import gym_pcgrl
import tensorflow as tf

supported_environments = [env.id for env in gym.envs.registry.all() if "gym_pcgrl" in env.entry_point]
print(f"Supported environments: ", end="")
print(*supported_environments)


env = gym.make('coding_game-narrow-v0')

for i_episode in range(20):
  obs = env.reset()
  for t in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward)
    env.render('human')
    if done:
      print("Episode finished after {} timesteps".format(t+1))
      break
env.close()