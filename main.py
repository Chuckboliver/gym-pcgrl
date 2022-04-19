import gym
import gym_pcgrl

supported_environments = [env.id for env in gym.envs.registry.all() if "gym_pcgrl" in env.entry_point]
print(f"Supported environments: ", end="")
print(*supported_environments)


env = gym.make('sokoban-narrow-v0')
obs = env.reset()
for t in range(1000):
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  env.render('human')

  if done:
    print("Episode finished after {} timesteps".format(t+1))
    break