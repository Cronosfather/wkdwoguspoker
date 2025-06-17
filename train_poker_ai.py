from stable_baselines3 import PPO
from poker_env import PokerEnv

env = PokerEnv()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)  # 만 번~십만 번까지 늘릴수록 더 잘 배움
model.save("final_poker_model")  # 학습된 모델 저장