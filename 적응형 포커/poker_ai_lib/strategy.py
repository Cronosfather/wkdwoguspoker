from collections import deque
from typing import Dict, Tuple, List
import numpy as np
import random
from .core import PokerHand, Card

class BluffingStrategy:
    def __init__(self):
        self.recent_bluffs = deque(maxlen=10)
        self.image_factor = 0.5
        self.meta_game_state = "balanced"

    def should_bluff(self, hand_strength: float, pot_odds: float, 
                    opponent_stats: Dict, game_context: Dict) -> Tuple[bool, float]:
        base_bluff_prob = self._calculate_base_bluff_probability(
            hand_strength, pot_odds, game_context
        )
        
        opponent_adjustment = self._adjust_for_opponent(
            base_bluff_prob, opponent_stats
        )
        
        meta_adjustment = self._apply_meta_game_adjustment(opponent_adjustment)
        
        final_bluff_prob = np.clip(meta_adjustment, 0.0, 0.7)
        
        bluff_intensity = self._calculate_bluff_intensity(
            final_bluff_prob, opponent_stats, game_context
        )
        
        should_bluff = random.random() < final_bluff_prob
        
        if should_bluff:
            self.recent_bluffs.append(1)
        else:
            self.recent_bluffs.append(0)
            
        return should_bluff, bluff_intensity

    def _calculate_base_bluff_probability(self, hand_strength: float, 
                                        pot_odds: float, game_context: Dict) -> float:
        hand_factor = max(0, 0.35 - hand_strength)
        
        pot_factor = min(0.25, pot_odds * 0.1)
        
        stage = game_context.get("stage", "preflop")
        stage_factor = {
            "preflop": 0.10,
            "flop": 0.20, 
            "turn": 0.25,
            "river": 0.30
        }.get(stage, 0.2)
        
        position_factor = game_context.get("position_advantage", 0.1)
        
        return hand_factor + pot_factor + stage_factor + position_factor

    def _adjust_for_opponent(self, base_prob: float, opponent_stats: Dict) -> float:
        player_type = opponent_stats.get("player_type", "unknown")
        fold_rate = opponent_stats.get("fold_rate", 0.3)
        estimated_range = opponent_stats.get("estimated_range", {})

        if "tight" in player_type:
            type_adjustment = 0.15
        elif "loose" in player_type:
            type_adjustment = -0.15
        else:
            type_adjustment = 0.0

        fold_adjustment = (fold_rate - 0.3) * 0.5

        strong_range_prob = estimated_range.get("premium", 0) + estimated_range.get("strong_pairs", 0)
        weak_range_prob = estimated_range.get("weak", 0.25)
        
        range_adjustment = (weak_range_prob - strong_range_prob) * 0.4

        return base_prob + type_adjustment + fold_adjustment + range_adjustment

    def _apply_meta_game_adjustment(self, base_prob: float) -> float:
        recent_bluff_rate = np.mean(self.recent_bluffs) if self.recent_bluffs else 0.2
        
        if recent_bluff_rate > 0.4:
            meta_adjustment = -0.2
            self.image_factor = min(1.0, self.image_factor + 0.1)
        elif recent_bluff_rate < 0.1:
            meta_adjustment = 0.15
            self.image_factor = max(0.0, self.image_factor - 0.1)
        else:
            meta_adjustment = 0.0
        
        return base_prob + meta_adjustment

    def _calculate_bluff_intensity(self, bluff_prob: float, opponent_stats: Dict, 
                                 game_context: Dict) -> float:
        base_intensity = bluff_prob * 1.5
        
        call_rate = opponent_stats.get("call_rate", 0.3)
        if call_rate > 0.4:
            intensity_boost = 0.3
        else:
            intensity_boost = 0.0
        
        stage = game_context.get("stage", "preflop")
        if stage == "river":
            stage_boost = 0.2
        else:
            stage_boost = 0.0
        
        return np.clip(base_intensity + intensity_boost + stage_boost, 0.3, 1.2)

class HandStrengthEvaluator:
    @staticmethod
    def evaluate_preflop_strength(hole_cards: List) -> float:
        if len(hole_cards) != 2:
            return 0.5
        
        card1, card2 = hole_cards
        rank1, rank2 = card1.rank.value, card2.rank.value
        
        if rank1 == rank2:
            if rank1 >= 10:
                return 0.9
            elif rank1 >= 7:
                return 0.7
            else:
                return 0.5
        
        suited = card1.suit == card2.suit
        
        high_cards = sorted([rank1, rank2], reverse=True)
        
        if high_cards[0] == 14:
            if high_cards[1] >= 10:
                return 0.85 if suited else 0.8
            elif high_cards[1] >= 7:
                return 0.6 if suited else 0.5
        
        if high_cards[0] >= 10 and high_cards[1] >= 10:
            return 0.7 if suited else 0.6
        
        if suited and abs(rank1 - rank2) <= 4:
            return 0.55
        
        if abs(rank1 - rank2) <= 2:
            return 0.5
        
        return 0.3

    @staticmethod
    def evaluate_postflop_strength(best_hand, community_cards: List) -> float:
        if best_hand is None:
            return 0.1
        
        rank_scores = {
            1: 0.1,
            2: 0.3,
            3: 0.5,
            4: 0.65,
            5: 0.75,
            6: 0.8,
            7: 0.9,
            8: 0.95,
            9: 0.98,
            10: 1.0
        }
        
        base_score = rank_scores.get(best_hand.rank.value, 0.1)
        
        board_danger = HandStrengthEvaluator._assess_board_danger(community_cards)
        adjusted_score = base_score * (1 - board_danger * 0.3)
        
        return np.clip(adjusted_score, 0.05, 1.0)

    @staticmethod
    def _assess_board_danger(community_cards: List) -> float:
        if len(community_cards) < 3:
            return 0.0
        
        danger = 0.0
        ranks = [card.rank.value for card in community_cards]
        suits = [card.suit for card in community_cards]
        
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}
        max_suit_count = max(suit_counts.values())
        if max_suit_count >= 3:
            danger += 0.3
        
        sorted_ranks = sorted(set(ranks))
        for i in range(len(sorted_ranks) - 2):
            if sorted_ranks[i+2] - sorted_ranks[i] <= 4:
                danger += 0.2
                break
        
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        if any(count >= 2 for count in rank_counts.values()):
            danger += 0.2
        
        if any(rank >= 10 for rank in ranks):
            danger += 0.1
        
        return min(danger, 1.0)
