from collections import defaultdict, deque
from typing import Dict
import numpy as np

class OpponentRangeModel:
    def get_range_probabilities(self, player_type: str) -> Dict[str, float]:
        base_ranges = {
            "premium": 0.10,
            "strong_pairs": 0.10,
            "strong_aces": 0.15,
            "suited_connectors": 0.15,
            "speculative": 0.25,
            "weak": 0.25,
        }

        if player_type == "tight_aggressive":
            adjustments = {"premium": 0.2, "strong_pairs": 0.1, "strong_aces": 0.1, "speculative": -0.1, "weak": -0.3}
        elif player_type == "tight_passive":
            adjustments = {"premium": 0.1, "strong_pairs": 0.1, "strong_aces": 0.05, "speculative": -0.1, "weak": -0.15}
        elif player_type == "loose_aggressive":
            adjustments = {"premium": -0.05, "suited_connectors": 0.1, "speculative": 0.1, "weak": 0.1}
        elif player_type == "loose_passive":
            adjustments = {"premium": -0.08, "suited_connectors": 0.05, "speculative": 0.1, "weak": 0.15}
        else: # unknown or normal
            adjustments = {}

        final_ranges = base_ranges.copy()
        for hand_type, adj in adjustments.items():
            final_ranges[hand_type] += adj

        total = sum(final_ranges.values())
        if total > 0:
            for hand_type in final_ranges:
                final_ranges[hand_type] = max(0, final_ranges[hand_type] / total)
        
        return final_ranges

class PlayerProfiling:
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.range_model = OpponentRangeModel()
        
        self.fold_frequency = deque(maxlen=history_size)
        self.raise_frequency = deque(maxlen=history_size)
        self.call_frequency = deque(maxlen=history_size)
        
        self.preflop_aggression = deque(maxlen=history_size)
        self.postflop_aggression = deque(maxlen=history_size)
        
        self.betting_sizes = defaultdict(list)
        
        self.showdown_results = deque(maxlen=50)
        
        self.player_type = "unknown"
        self.confidence = 0.0

    def record_action(self, action: str, amount: int, pot_size: int, 
                     game_stage: str, position: str = ""):
        if action == "fold":
            self.fold_frequency.append(1)
        else:
            self.fold_frequency.append(0)
            
        if action == "raise":
            self.raise_frequency.append(1)
            bet_ratio = amount / max(pot_size, 1)
            self.betting_sizes[game_stage].append(bet_ratio)
        else:
            self.raise_frequency.append(0)
            
        if action == "call":
            self.call_frequency.append(1)
        else:
            self.call_frequency.append(0)
        
        aggression = 1 if action == "raise" else 0
        if game_stage == "preflop":
            self.preflop_aggression.append(aggression)
        else:
            self.postflop_aggression.append(aggression)

    def record_showdown(self, had_strong_hand: bool, won: bool):
        self.showdown_results.append((had_strong_hand, won))

    def get_player_stats(self) -> Dict:
        if not self.fold_frequency:
            player_type = "unknown"
            base_stats = {"fold_rate": 0.5, "raise_rate": 0.2, "call_rate": 0.3, "player_type": player_type}
        else:
            player_type = self._classify_player_type()
            base_stats = {
                "fold_rate": np.mean(self.fold_frequency),
                "raise_rate": np.mean(self.raise_frequency),
                "call_rate": np.mean(self.call_frequency),
                "preflop_aggression": np.mean(self.preflop_aggression) if self.preflop_aggression else 0.2,
                "postflop_aggression": np.mean(self.postflop_aggression) if self.postflop_aggression else 0.2,
                "player_type": player_type,
                "bluff_success_rate": self._estimate_bluff_success_rate()
            }
        
        estimated_range = self.range_model.get_range_probabilities(player_type)
        base_stats["estimated_range"] = estimated_range
        
        return base_stats

    def _classify_player_type(self) -> str:
        if len(self.fold_frequency) < 10:
            return "unknown"
        
        fold_rate = np.mean(self.fold_frequency)
        raise_rate = np.mean(self.raise_frequency)
        
        vpip = 1 - fold_rate
        
        pfr = np.mean(self.preflop_aggression) if self.preflop_aggression else 0
        
        if vpip < 0.2:
            if pfr > 0.15:
                return "tight_aggressive"
            else:
                return "tight_passive"
        elif vpip > 0.4:
            if pfr > 0.25:
                return "loose_aggressive"
            else:
                return "loose_passive"
        else:
            if pfr > 0.2:
                return "normal_aggressive"
            else:
                return "normal_passive"

    def _estimate_bluff_success_rate(self) -> float:
        if len(self.showdown_results) < 5:
            return 0.3
        
        bluff_wins = sum(1 for strong, won in self.showdown_results 
                        if not strong and won)
        total_weak_showdowns = sum(1 for strong, won in self.showdown_results 
                                 if not strong)
        
        if total_weak_showdowns == 0:
            return 0.3
        
        return bluff_wins / total_weak_showdowns
