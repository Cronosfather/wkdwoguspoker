from collections import deque
from typing import Dict, Optional
import numpy as np
import random

class LearningMemory:
    def __init__(self, max_size: int = 10000):
        self.experiences = deque(maxlen=max_size)
        self.current_hand_data = []

    def start_hand(self):
        self.current_hand_data = []

    def add_experience(self, observation: np.ndarray, action: int, reward: float, 
                     next_observation: Optional[np.ndarray] = None, done: bool = False):
        experience = {
            'observation': observation.copy(),
            'action': action,
            'reward': reward,
            'next_observation': next_observation.copy() if next_observation is not None else None,
            'done': done
        }
        self.current_hand_data.append(experience)

    def finish_hand(self, final_reward: float):
        for i, exp in enumerate(self.current_hand_data):
            discount_factor = 0.95 ** (len(self.current_hand_data) - i - 1)
            exp['final_reward'] = final_reward * discount_factor
            self.experiences.append(exp)
        
        self.current_hand_data = []

    def get_batch(self, batch_size: int = 32) -> Dict:
        if len(self.experiences) < batch_size:
            batch_size = len(self.experiences)
        
        if batch_size == 0:
            return None
            
        batch = random.sample(list(self.experiences), batch_size)
        
        observations = np.array([exp['observation'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp.get('final_reward', exp['reward']) for exp in batch])
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards
        }
