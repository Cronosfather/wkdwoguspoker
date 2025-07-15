import numpy as np
from typing import Dict, Tuple
import os
import json

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    from gymnasium import spaces
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False

from .memory import LearningMemory
from .opponent_model import PlayerProfiling
from .strategy import HandStrengthEvaluator

class AdaptivePokerAI(gym.Env):
    def __init__(self, model_path: str = None, learning_rate: float = 0.0003):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(60,), dtype=np.float32
        )
        
        self.learning_rate = learning_rate
        self.memory = LearningMemory()
        self.model = None
        self.win_rate = 0.5
        self.games_played = 0
        self.wins = 0
        self.model_path = model_path or "adaptive_poker_ai_v2"
        self.current_observation = None
        self.player_profiling = PlayerProfiling()
        self.hand_evaluator = HandStrengthEvaluator()
        self.ai_chips = 1000

        self.learning_stats = {
            'games_played': 0, 'wins': 0, 'losses': 0,
            'win_rate_history': [], 'learning_episodes': 0
        }
        
        if PPO_AVAILABLE:
            self._initialize_model()
        else:
            print("⚠️ PPO를 사용할 수 없습니다. 단순 적응 AI를 사용합니다.")

    def _initialize_model(self):
        try:
            if os.path.exists(f"{self.model_path}.zip"):
                self.model = PPO.load(self.model_path, env=DummyVecEnv([lambda: self]))
                print(f"✅ 기존 모델을 로드했습니다: {self.model_path}")
                if os.path.exists(f"{self.model_path}_stats.json"):
                    with open(f"{self.model_path}_stats.json", 'r') as f:
                        self.learning_stats = json.load(f)
            else:
                env = DummyVecEnv([lambda: self])
                self.model = PPO('MlpPolicy', env,
                            learning_rate=self.learning_rate, n_steps=4096,
                            batch_size=64, n_epochs=10, verbose=1)
                print("🆕 새로운 PPO 모델(v2)을 생성했습니다.")
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
            self.model = None

    def decide_action(self, observation: np.ndarray, legal_actions: Dict[int, int]) -> Tuple[int, int]:
        if self.model is None:
            if 1 in legal_actions:
                return 1, legal_actions[1]
            return 0, 0

        action, _ = self.model.predict(observation, deterministic=False)
        action = int(action)

        if action in legal_actions:
            return action, legal_actions[action]
        else:
            if 4 in legal_actions: return 4, legal_actions[4]
            if 3 in legal_actions: return 3, legal_actions[3]
            if 2 in legal_actions: return 2, legal_actions[2]
            if 1 in legal_actions: return 1, legal_actions[1]
            return 0, 0

    def learn_from_hand(self, final_reward: float):
        self.memory.finish_hand(final_reward)
        if self.model is not None and len(self.memory.experiences) >= 128:
            try:
                self.model.learn(total_timesteps=10000, reset_num_timesteps=False)
                self.learning_stats['learning_episodes'] += 1
            except Exception as e:
                print(f"⚠️ 학습 오류: {e}")

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_observation = np.zeros(60, dtype=np.float32)
        return self.current_observation, {}

    def step(self, action):
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return self.current_observation, reward, done, truncated, info

    def update_stats(self, won: bool):
        self.games_played += 1
        if won:
            self.wins += 1
        
        if self.games_played > 0:
            self.win_rate = self.wins / self.games_played
        
        self.learning_stats['games_played'] = self.games_played
        self.learning_stats['wins'] = self.wins
        self.learning_stats['losses'] = self.games_played - self.wins
        self.learning_stats['win_rate_history'].append(self.win_rate)
        
        if len(self.learning_stats['win_rate_history']) > 100:
            self.learning_stats['win_rate_history'] = self.learning_stats['win_rate_history'][-100:]

    def save_model(self):
        if self.model is not None:
            try:
                self.model.save(self.model_path)
                with open(f"{self.model_path}_stats.json", 'w') as f:
                    json.dump(self.learning_stats, f, indent=2)
                print(f"💾 모델이 저장되었습니다: {self.model_path}")
            except Exception as e:
                print(f"❌ 모델 저장 실패: {e}")

    def get_learning_progress(self) -> str:
        if self.games_played == 0:
            return "🆕 아직 게임을 시작하지 않았습니다."
        
        recent_win_rate = np.mean(self.learning_stats['win_rate_history'][-10:]) if len(self.learning_stats['win_rate_history']) >= 10 else self.win_rate
        
        progress = f"""
🧠 AI 학습 현황 (v2):
📊 총 게임: {self.games_played}게임
🏆 승률: {self.win_rate:.1%} ({self.games_played-self.wins}패)
📈 최근 10게임 승률: {recent_win_rate:.1%}
🎓 학습 루프: {self.learning_stats['learning_episodes']}회
💾 경험 데이터: {len(self.memory.experiences)}개
        """
        return progress.strip()
