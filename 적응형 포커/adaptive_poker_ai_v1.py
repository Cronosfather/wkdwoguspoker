# 필요한 라이브러리들을 임포트하는 부분이다.
import numpy as np  # 다차원 배열 및 수학적 연산을 위해 사용되는 라이브러리이다.
import random  # 무작위 선택 및 셔플을 위해 사용되는 라이브러리이다.
from typing import List, Tuple, Dict, Optional, Any  # 타입 힌팅을 위해 사용되는 모듈이다.
from enum import Enum  # 상수 집합을 정의하기 위해 사용되는 클래스이다.
from itertools import combinations  # 조합을 계산하기 위해 사용되는 함수이다.
import pickle  # 파이썬 객체를 저장하고 불러오기 위해 사용된다 (현재 코드에서는 사용되지 않음).
import os  # 운영체제와 상호작용하기 위해 사용된다 (예: 파일 경로 확인).
from collections import defaultdict, deque  # 효율적인 데이터 구조를 제공하는 모듈이다.
import json  # JSON 형식의 데이터를 다루기 위해 사용된다.

# 강화학습 라이브러리인 stable_baselines3를 임포트하는 부분이다.
# 라이브러리가 없을 경우를 대비한 예외 처리를 포함한다.
try:
    from stable_baselines3 import PPO # PPO 알고리즘을 사용하기 위한 클래스이다.
    from stable_baselines3.common.env_checker import check_env # Gym 환경이 유효한지 검사하는 함수이다.
    from stable_baselines3.common.vec_env import DummyVecEnv # 벡터화된 환경을 생성하기 위한 클래스이다.
    import gymnasium as gym  # 강화학습 환경을 구축하기 위한 표준 라이브러리이다.
    from gymnasium import spaces  # 행동 및 관측 공간을 정의하기 위한 클래스이다.
    PPO_AVAILABLE = True # 라이브러리가 성공적으로 임포트되었음을 나타내는 플래그이다.
except ImportError:
    print("Warning: stable_baselines3가 설치되지 않았습니다. pip install stable-baselines3 gymnasium로 설치해주세요.")
    PPO_AVAILABLE = False # 라이브러리가 없음을 나타내는 플래그이다.

# 카드의 무늬(Suit)를 정의하는 Enum 클래스이다.
class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"  
    SPADES = "♠"

# 카드의 숫자(Rank)를 정의하는 Enum 클래스이다. Ace를 가장 높은 값으로 처리한다.
class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

# 한 장의 카드를 나타내는 클래스이다.
class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit  # 카드의 무늬이다.
        self.rank = rank  # 카드의 랭크(숫자)이다.
    
    # 카드를 사람이 읽기 쉬운 문자열로 표현하는 메소드이다.
    def __str__(self):
        rank_str = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 
                   9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
        return f"{rank_str[self.rank.value]}{self.suit.value}"
    
    # 객체의 공식적인 문자열 표현을 정의하는 메소드이다.
    def __repr__(self):
        return str(self)

# 포커 핸드(족보)의 서열을 정의하는 Enum 클래스이다.
class HandRank(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10

# 5장의 카드로 구성된 포커 핸드를 평가하고 비교하는 클래스이다.
class PokerHand:
    def __init__(self, cards: List[Card]):
        # 카드를 랭크가 높은 순으로 정렬하여 저장한다.
        self.cards = sorted(cards, key=lambda x: x.rank.value, reverse=True)
        # 핸드를 평가하여 족보(rank)와 그 가치(value)를 결정한다.
        self.rank, self.value = self._evaluate_hand()
    
    # 5장의 카드를 분석하여 족보를 결정하는 핵심 메소드이다.
    def _evaluate_hand(self) -> Tuple[HandRank, List[int]]:
        ranks = [card.rank.value for card in self.cards]
        suits = [card.suit for card in self.cards]
        rank_counts = {}
        
        # 각 랭크의 개수를 센다.
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        is_flush = len(set(suits)) == 1  # 모든 카드의 무늬가 같은지 확인하여 플러시 여부를 결정한다.
        is_straight = self._is_straight(ranks)  # 카드의 랭크가 연속적인지 확인하여 스트레이트 여부를 결정한다.
        
        counts = sorted(rank_counts.values(), reverse=True) # 랭크 개수를 내림차순으로 정렬한다. (예: 풀하우스 -> [3, 2])
        unique_ranks = sorted(rank_counts.keys(), key=lambda x: (rank_counts[x], x), reverse=True) # 족보 계산에 사용될 랭크 순서이다.
        
        # 높은 족보부터 순서대로 확인한다.
        if is_straight and is_flush:
            if min(ranks) == 10: # 로열 스트레이트 플러시인지 확인한다.
                return HandRank.ROYAL_FLUSH, [14]
            return HandRank.STRAIGHT_FLUSH, [max(ranks)]
        
        if counts == [4, 1]:
            return HandRank.FOUR_KIND, unique_ranks
        
        if counts == [3, 2]:
            return HandRank.FULL_HOUSE, unique_ranks
        
        if is_flush:
            return HandRank.FLUSH, sorted(ranks, reverse=True)
        
        if is_straight:
            return HandRank.STRAIGHT, [max(ranks)]
        
        if counts == [3, 1, 1]:
            return HandRank.THREE_KIND, unique_ranks
        
        if counts == [2, 2, 1]:
            return HandRank.TWO_PAIR, unique_ranks
        
        if counts == [2, 1, 1, 1]:
            return HandRank.PAIR, unique_ranks
        
        return HandRank.HIGH_CARD, sorted(ranks, reverse=True)
    
    # 스트레이트 여부를 판별하는 내부 메소드이다.
    def _is_straight(self, ranks: List[int]) -> bool:
        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) != 5:
            return False
        
        # 일반적인 스트레이트를 확인한다.
        if unique_ranks[-1] - unique_ranks[0] == 4:
            return True
        
        # A-2-3-4-5 스트레이트(마운틴)를 예외적으로 확인한다.
        if unique_ranks == [2, 3, 4, 5, 14]:
            return True
        
        return False
    
    # 두 포커 핸드의 우열을 비교하는 연산자(>)를 정의한다.
    def __gt__(self, other):
        if self.rank.value != other.rank.value:
            return self.rank.value > other.rank.value
        return self.value > other.value

# 52장의 카드 덱을 생성하고 관리하는 클래스이다.
class Deck:
    def __init__(self):
        self.cards = []
        self.reset()
    
    # 덱을 52장의 카드로 새로 만들고 무작위로 섞는다.
    def reset(self):
        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]
        random.shuffle(self.cards)
    
    # 덱의 맨 위에서 카드 한 장을 뽑아 반환한다.
    def deal_card(self) -> Card:
        return self.cards.pop()
    
# 상대방의 핸드 범위를 플레이어 타입에 따라 추정하는 클래스이다.
class OpponentRangeModel:
    """
    상대방의 핸드 레인지를 플레이어 타입에 따라 추정하는 클래스이다.
    핸드 카테고리:
    - premium: AA, KK, QQ, AKs
    - strong_pairs: JJ, TT, 99
    - strong_aces: AQs, AJs, ATs, AKo
    - suited_connectors: KQs, QJs, JTs, T9s
    - speculative: 88, 77, A9s-A2s, KJo, QTo
    - weak: 나머지 모든 핸드
    """
    # 플레이어 타입에 따라 핸드 레인지 확률을 반환하는 메소드이다.
    def get_range_probabilities(self, player_type: str) -> Dict[str, float]:
        # 기본 핸드 레인지 확률 분포이다.
        base_ranges = {
            "premium": 0.10,
            "strong_pairs": 0.10,
            "strong_aces": 0.15,
            "suited_connectors": 0.15,
            "speculative": 0.25,
            "weak": 0.25,
        }

        # 플레이어 타입에 따라 확률을 조정하는 값이다.
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

        # 기본 확률에 조정값을 더해 최종 확률을 계산한다.
        final_ranges = base_ranges.copy()
        for hand_type, adj in adjustments.items():
            final_ranges[hand_type] += adj

        # 확률의 총합이 1이 되도록 정규화한다.
        total = sum(final_ranges.values())
        if total > 0:
            for hand_type in final_ranges:
                final_ranges[hand_type] = max(0, final_ranges[hand_type] / total)
        
        return final_ranges

# 상대 플레이어의 행동 패턴을 분석하여 프로파일링하는 클래스이다.
class PlayerProfiling:
    """상대 플레이어의 행동 패턴과 핸드 레인지를 분석하는 클래스이다"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size # 분석할 최근 행동의 개수이다.
        self.range_model = OpponentRangeModel() # 핸드 레인지 추정 모델을 인스턴스화한다.
        
        # 플레이어의 행동 빈도를 추적하기 위한 deque이다.
        self.fold_frequency = deque(maxlen=history_size)
        self.raise_frequency = deque(maxlen=history_size)
        self.call_frequency = deque(maxlen=history_size)
        
        # 게임 단계별 공격성을 추적하기 위한 deque이다.
        self.preflop_aggression = deque(maxlen=history_size)
        self.postflop_aggression = deque(maxlen=history_size)
        
        # 베팅 크기 패턴을 저장하기 위한 딕셔너리이다.
        self.betting_sizes = defaultdict(list)
        
        # 쇼다운 결과를 통해 블러핑 성공률을 추적하기 위한 deque이다.
        self.showdown_results = deque(maxlen=50)
        
        # 추정된 플레이어 타입과 그 신뢰도이다.
        self.player_type = "unknown"
        self.confidence = 0.0
    
    # 플레이어의 행동을 기록하는 메소드이다.
    def record_action(self, action: str, amount: int, pot_size: int, 
                     game_stage: str, position: str = ""):
        """플레이어 행동 기록"""
        # 기본 행동 빈도를 기록한다.
        if action == "fold":
            self.fold_frequency.append(1)
        else:
            self.fold_frequency.append(0)
            
        if action == "raise":
            self.raise_frequency.append(1)
            # 팟 대비 베팅 크기 비율을 저장한다.
            bet_ratio = amount / max(pot_size, 1)
            self.betting_sizes[game_stage].append(bet_ratio)
        else:
            self.raise_frequency.append(0)
            
        if action == "call":
            self.call_frequency.append(1)
        else:
            self.call_frequency.append(0)
        
        # 상황별 공격성을 기록한다.
        aggression = 1 if action == "raise" else 0
        if game_stage == "preflop":
            self.preflop_aggression.append(aggression)
        else:
            self.postflop_aggression.append(aggression)
    
    # 쇼다운 결과를 기록하여 블러핑 패턴을 분석하는 데 사용한다.
    def record_showdown(self, had_strong_hand: bool, won: bool):
        """쇼다운 결과 기록 (블러핑 패턴 분석용)"""
        self.showdown_results.append((had_strong_hand, won))
    
    # 분석된 플레이어 통계와 추정 핸드 레인지를 반환하는 메소드이다.
    def get_player_stats(self) -> Dict:
        """플레이어 통계 및 추정 핸드 레인지 반환"""
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
        
        # 추정된 핸드 레인지를 통계에 추가한다.
        estimated_range = self.range_model.get_range_probabilities(player_type)
        base_stats["estimated_range"] = estimated_range
        
        return base_stats
    
    # 기록된 데이터를 바탕으로 플레이어의 스타일을 분류하는 메소드이다.
    def _classify_player_type(self) -> str:
        """플레이어 타입 분류"""
        if len(self.fold_frequency) < 10:
            return "unknown"
        
        fold_rate = np.mean(self.fold_frequency)
        raise_rate = np.mean(self.raise_frequency)
        
        # VPIP (Voluntarily Put In Pot): 자발적으로 팟에 돈을 넣는 비율이다.
        vpip = 1 - fold_rate
        
        # PFR (Pre-Flop Raise): 프리플롭에서 레이즈하는 비율이다.
        pfr = np.mean(self.preflop_aggression) if self.preflop_aggression else 0
        
        if vpip < 0.2:  # 매우 신중한 플레이어 (타이트)
            if pfr > 0.15:
                return "tight_aggressive"
            else:
                return "tight_passive"
        elif vpip > 0.4:  # 많은 핸드로 참여하는 플레이어 (루즈)
            if pfr > 0.25:
                return "loose_aggressive"
            else:
                return "loose_passive"
        else:  # 일반적인 플레이어
            if pfr > 0.2:
                return "normal_aggressive"
            else:
                return "normal_passive"
    
    # 쇼다운 결과를 바탕으로 상대의 블러핑 성공률을 추정하는 메소드이다.
    def _estimate_bluff_success_rate(self) -> float:
        """상대의 블러핑 성공률 추정"""
        if len(self.showdown_results) < 5:
            return 0.3  # 데이터가 부족하면 기본값을 반환한다.
        
        # 약한 핸드로 이긴 경우를 블러핑 성공으로 간주한다.
        bluff_wins = sum(1 for strong, won in self.showdown_results 
                        if not strong and won)
        total_weak_showdowns = sum(1 for strong, won in self.showdown_results 
                                 if not strong)
        
        if total_weak_showdowns == 0:
            return 0.3
        
        return bluff_wins / total_weak_showdowns

# 블러핑 전략을 결정하는 로직을 담당하는 클래스이다.
class BluffingStrategy:
    """블러핑 전략을 담당하는 클래스이다"""
    
    def __init__(self):
        self.recent_bluffs = deque(maxlen=10)  # 최근 블러핑 시도 기록이다.
        self.image_factor = 0.5  # AI 자신의 현재 이미지 (0: 타이트, 1: 루즈)이다.
        self.meta_game_state = "balanced"  # 메타게임 상태 (균형, 착취, 조정)이다.
        
    # 다양한 요소를 고려하여 블러핑 여부와 강도를 결정하는 메소드이다.
    def should_bluff(self, hand_strength: float, pot_odds: float, 
                    opponent_stats: Dict, game_context: Dict) -> Tuple[bool, float]:
        """블러핑 여부와 강도 결정"""
        
        # 기본 블러핑 확률을 계산한다.
        base_bluff_prob = self._calculate_base_bluff_probability(
            hand_strength, pot_odds, game_context
        )
        
        # 상대방의 통계에 따라 확률을 조정한다.
        opponent_adjustment = self._adjust_for_opponent(
            base_bluff_prob, opponent_stats
        )
        
        # 메타게임(게임의 흐름)을 고려하여 확률을 조정한다.
        meta_adjustment = self._apply_meta_game_adjustment(opponent_adjustment)
        
        # 최종 블러핑 확률을 결정한다.
        final_bluff_prob = np.clip(meta_adjustment, 0.0, 0.7)
        
        # 블러핑 강도(베팅 크기)를 계산한다.
        bluff_intensity = self._calculate_bluff_intensity(
            final_bluff_prob, opponent_stats, game_context
        )
        
        should_bluff = random.random() < final_bluff_prob
        
        if should_bluff:
            self.recent_bluffs.append(1)
        else:
            self.recent_bluffs.append(0)
            
        return should_bluff, bluff_intensity
    
    # 핸드 강도, 팟 오즈, 게임 상황을 바탕으로 기본 블러핑 확률을 계산하는 메소드이다.
    def _calculate_base_bluff_probability(self, hand_strength: float, 
                                        pot_odds: float, game_context: Dict) -> float:
        """기본 블러핑 확률 계산"""
        # 핸드가 약할수록 블러핑 확률이 증가한다.
        hand_factor = max(0, 0.35 - hand_strength)
        
        # 팟 오즈가 좋을수록 블러핑 확률이 증가한다.
        pot_factor = min(0.25, pot_odds * 0.1)
        
        # 게임 단계(리버로 갈수록)에 따라 블러핑 확률을 조정한다.
        stage = game_context.get("stage", "preflop")
        stage_factor = {
            "preflop": 0.10,
            "flop": 0.20, 
            "turn": 0.25,
            "river": 0.30
        }.get(stage, 0.2)
        
        # 유리한 포지션일수록 블러핑 확률이 증가한다.
        position_factor = game_context.get("position_advantage", 0.1)
        
        return hand_factor + pot_factor + stage_factor + position_factor
    
    # 상대방의 타입과 추정 핸드 레인지에 따라 블러핑 확률을 조정하는 메소드이다.
    def _adjust_for_opponent(self, base_prob: float, opponent_stats: Dict) -> float:
        """상대 타입과 추정 레인지에 따른 블러핑 확률 조정"""
        player_type = opponent_stats.get("player_type", "unknown")
        fold_rate = opponent_stats.get("fold_rate", 0.3)
        estimated_range = opponent_stats.get("estimated_range", {})

        # 플레이어 타입에 따라 조정한다. (타이트한 상대에게는 더 많이 블러핑)
        if "tight" in player_type:
            type_adjustment = 0.15
        elif "loose" in player_type:
            type_adjustment = -0.15
        else:
            type_adjustment = 0.0

        # 상대의 폴드율에 따라 조정한다. (폴드를 많이 할수록 더 많이 블러핑)
        fold_adjustment = (fold_rate - 0.3) * 0.5

        # 상대의 추정 핸드 레인지에 따라 조정한다. (상대가 약한 핸드를 가졌을 것 같으면 더 많이 블러핑)
        strong_range_prob = estimated_range.get("premium", 0) + estimated_range.get("strong_pairs", 0)
        weak_range_prob = estimated_range.get("weak", 0.25)
        
        range_adjustment = (weak_range_prob - strong_range_prob) * 0.4

        return base_prob + type_adjustment + fold_adjustment + range_adjustment
    
    # 메타게임을 고려하여 블러핑 확률을 조정하는 메소드이다.
    def _apply_meta_game_adjustment(self, base_prob: float) -> float:
        """메타게임을 고려한 조정"""
        # 최근에 블러핑을 너무 많이 했다면, 빈도를 줄여 상대의 의심을 피한다.
        recent_bluff_rate = np.mean(self.recent_bluffs) if self.recent_bluffs else 0.2
        
        if recent_bluff_rate > 0.4:
            meta_adjustment = -0.2
            self.image_factor = min(1.0, self.image_factor + 0.1)  # 자신의 이미지를 루즈하게 조정한다.
        elif recent_bluff_rate < 0.1:
            meta_adjustment = 0.15
            self.image_factor = max(0.0, self.image_factor - 0.1)  # 자신의 이미지를 타이트하게 조정한다.
        else:
            meta_adjustment = 0.0
        
        return base_prob + meta_adjustment
    
    # 블러핑의 강도(베팅 크기)를 계산하는 메소드이다.
    def _calculate_bluff_intensity(self, bluff_prob: float, opponent_stats: Dict, 
                                 game_context: Dict) -> float:
        """블러핑 강도 계산 (베팅 크기 결정용)"""
        # 블러핑 성공 확률이 높을수록 더 강하게 베팅한다.
        base_intensity = bluff_prob * 1.5
        
        # 상대가 콜을 많이 하는 타입이면 더 크게 베팅하여 폴드를 유도한다.
        call_rate = opponent_stats.get("call_rate", 0.3)
        if call_rate > 0.4:
            intensity_boost = 0.3
        else:
            intensity_boost = 0.0
        
        # 리버에서는 블러핑을 더 크게 하여 승리를 가져오려 시도한다.
        stage = game_context.get("stage", "preflop")
        if stage == "river":
            stage_boost = 0.2
        else:
            stage_boost = 0.0
        
        return np.clip(base_intensity + intensity_boost + stage_boost, 0.3, 1.2)
    
# 핸드의 강도를 0에서 1 사이의 값으로 정량적으로 평가하는 클래스이다.
class HandStrengthEvaluator:
    """핸드 강도를 0-1 스케일로 평가"""
    
    @staticmethod
    # 프리플롭 상태에서 2장의 핸드 카드만으로 강도를 평가하는 정적 메소드이다.
    def evaluate_preflop_strength(hole_cards: List) -> float:
        """프리플롭 핸드 강도 평가"""
        if len(hole_cards) != 2:
            return 0.5
        
        card1, card2 = hole_cards
        rank1, rank2 = card1.rank.value, card2.rank.value
        
        # 페어일 경우, 랭크가 높을수록 강하다.
        if rank1 == rank2:
            if rank1 >= 10:  # TT 이상
                return 0.9
            elif rank1 >= 7:  # 77-99
                return 0.7
            else:  # 22-66
                return 0.5
        
        suited = card1.suit == card2.suit # 카드의 무늬가 같은지 여부이다.
        
        high_cards = sorted([rank1, rank2], reverse=True)
        
        # AK, AQ 등 높은 랭크의 카드 조합일수록 강하다.
        if high_cards[0] == 14:  # 에이스
            if high_cards[1] >= 10:
                return 0.85 if suited else 0.8
            elif high_cards[1] >= 7:
                return 0.6 if suited else 0.5
        
        # 기타 브로드웨이(T, J, Q, K, A) 카드 조합이다.
        if high_cards[0] >= 10 and high_cards[1] >= 10:
            return 0.7 if suited else 0.6
        
        # 무늬가 같고 숫자가 연결된 카드(수딧 커넥터)이다.
        if suited and abs(rank1 - rank2) <= 4:
            return 0.55
        
        # 숫자가 연결된 카드(커넥터)이다.
        if abs(rank1 - rank2) <= 2:
            return 0.5
        
        return 0.3  # 그 외의 약한 핸드이다.
    
    @staticmethod
    # 포스트플롭 상태에서 최종 족보와 커뮤니티 카드를 고려하여 강도를 평가하는 정적 메소드이다.
    def evaluate_postflop_strength(best_hand, community_cards: List) -> float:
        """포스트플롭 핸드 강도 평가"""
        if best_hand is None:
            return 0.1
        
        # 족보의 서열에 따라 기본 점수를 매긴다.
        rank_scores = {
            1: 0.1,   # 하이카드
            2: 0.3,   # 원페어
            3: 0.5,   # 투페어
            4: 0.65,  # 트리플
            5: 0.75,  # 스트레이트
            6: 0.8,   # 플러시
            7: 0.9,   # 풀하우스
            8: 0.95,  # 포카드
            9: 0.98,  # 스트레이트 플러시
            10: 1.0   # 로열 플러시
        }
        
        base_score = rank_scores.get(best_hand.rank.value, 0.1)
        
        # 보드의 위험도(상대에게 더 좋은 패가 나올 가능성)에 따라 점수를 조정한다.
        board_danger = HandStrengthEvaluator._assess_board_danger(community_cards)
        adjusted_score = base_score * (1 - board_danger * 0.3)
        
        return np.clip(adjusted_score, 0.05, 1.0)
    
    @staticmethod
    # 커뮤니티 카드의 위험도를 평가하는 내부 정적 메소드이다.
    def _assess_board_danger(community_cards: List) -> float:
        """보드의 위험도 평가 (0-1)"""
        if len(community_cards) < 3:
            return 0.0
        
        danger = 0.0
        ranks = [card.rank.value for card in community_cards]
        suits = [card.suit for card in community_cards]
        
        # 플러시 드로우 가능성이 높을수록 위험도가 증가한다.
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}
        max_suit_count = max(suit_counts.values())
        if max_suit_count >= 3:
            danger += 0.3
        
        # 스트레이트 드로우 가능성이 높을수록 위험도가 증가한다.
        sorted_ranks = sorted(set(ranks))
        for i in range(len(sorted_ranks) - 2):
            if sorted_ranks[i+2] - sorted_ranks[i] <= 4:
                danger += 0.2
                break
        
        # 보드에 페어가 깔리면 풀하우스 등의 위험이 있어 위험도가 증가한다.
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        if any(count >= 2 for count in rank_counts.values()):
            danger += 0.2
        
        # 보드에 높은 카드가 깔리면 상대의 족보가 높을 가능성이 있어 위험도가 증가한다.
        if any(rank >= 10 for rank in ranks):
            danger += 0.1
        
        return min(danger, 1.0)
    
# AI의 학습 경험을 저장하고 관리하는 클래스이다. (리플레이 버퍼)
class LearningMemory:
    """AI의 학습 메모리를 관리하는 클래스이다"""
    def __init__(self, max_size: int = 10000):
        self.experiences = deque(maxlen=max_size) # 최대 크기가 정해진 큐(deque)에 경험을 저장한다.
        self.current_hand_data = [] # 현재 핸드에서 발생한 경험을 임시로 저장하는 리스트이다.
        
    # 새 핸드가 시작될 때 호출되는 메소드이다.
    def start_hand(self):
        """새 핸드 시작"""
        self.current_hand_data = []
    
    # 하나의 경험(상태, 행동, 보상 등)을 임시 리스트에 추가하는 메소드이다.
    def add_experience(self, observation: np.ndarray, action: int, reward: float, 
                     next_observation: Optional[np.ndarray] = None, done: bool = False):
        """경험 추가"""
        experience = {
            'observation': observation.copy(),
            'action': action,
            'reward': reward,
            'next_observation': next_observation.copy() if next_observation is not None else None,
            'done': done
        }
        self.current_hand_data.append(experience)
    
    # 핸드가 종료될 때 호출되어, 최종 보상을 바탕으로 경험을 가공하여 주 메모리에 저장한다.
    def finish_hand(self, final_reward: float):
        """핸드 종료 시 경험을 메모리에 저장"""
        # 핸드에서 발생한 모든 행동에 대해 최종 보상을 역전파한다.
        for i, exp in enumerate(self.current_hand_data):
            # 나중에 한 행동일수록 최종 보상에 더 큰 영향을 미쳤다고 가정하여 할인율을 적용한다.
            discount_factor = 0.95 ** (len(self.current_hand_data) - i - 1)
            exp['final_reward'] = final_reward * discount_factor
            self.experiences.append(exp)
        
        self.current_hand_data = []
    
    # 학습에 사용할 배치 데이터를 무작위로 샘플링하여 반환하는 메소드이다.
    def get_batch(self, batch_size: int = 32) -> Dict:
        """학습용 배치 데이터 반환"""
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

# 강화학습 에이전트 역할을 하는 핵심 AI 클래스이다. gym.Env를 상속받는다.
class AdaptivePokerAI(gym.Env):
    def __init__(self, model_path: str = None, learning_rate: float = 0.0003):
        super().__init__()
        # 행동 공간을 정의한다. (0:폴드, 1:콜, 2:레이즈50%, 3:레이즈100%, 4:올인)
        self.action_space = spaces.Discrete(5)
        
        # 관측 공간을 정의한다. 60개의 float 값으로 게임 상태를 표현한다.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(60,), dtype=np.float32
        )
        
        self.learning_rate = learning_rate # 학습률이다.
        self.memory = LearningMemory() # 학습 메모리 인스턴스이다.
        self.model = None # PPO 모델이 저장될 변수이다.
        self.win_rate = 0.5 # AI의 승률이다.
        self.games_played = 0 # 총 플레이한 게임 수이다.
        self.wins = 0 # 승리한 게임 수이다.
        self.model_path = model_path or "adaptive_poker_ai_v2" # 모델 저장 경로이다.
        self.current_observation = None # 현재 관측 상태이다.
        self.player_profiling = PlayerProfiling() # 상대 분석기 인스턴스이다.
        self.hand_evaluator = HandStrengthEvaluator() # 핸드 평가기 인스턴스이다.
        self.ai_chips = 1000 # AI의 칩 개수이다.

        # 학습 관련 통계를 저장하는 딕셔너리이다.
        self.learning_stats = {
            'games_played': 0, 'wins': 0, 'losses': 0,
            'win_rate_history': [], 'learning_episodes': 0
        }
        
        if PPO_AVAILABLE:
            self._initialize_model() # PPO 모델을 초기화한다.
        else:
            print("⚠️ PPO를 사용할 수 없습니다. 단순 적응 AI를 사용합니다.")

    # PPO 모델을 초기화하는 메소드이다. 저장된 파일이 있으면 로드하고, 없으면 새로 생성한다.
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

    # 현재 관측 상태를 바탕으로 PPO 모델을 사용하여 행동을 결정하는 메소드이다.
    def decide_action(self, observation: np.ndarray, legal_actions: Dict[int, int]) -> Tuple[int, int]:
        if self.model is None:
            # PPO 모델이 없으면, 규칙 기반으로 가장 안전한 행동(콜 또는 폴드)을 선택한다.
            if 1 in legal_actions: # Call이 가능하면 Call한다.
                return 1, legal_actions[1]
            return 0, 0 # 아니면 Fold한다.

        # PPO 모델을 사용하여 행동을 예측한다. deterministic=False는 탐험을 위해 무작위성을 추가한다.
        action, _ = self.model.predict(observation, deterministic=False)
        action = int(action)

        # 예측된 행동이 현재 가능한 행동인지 확인하고, 불가능하면 대안을 선택한다.
        if action in legal_actions:
            return action, legal_actions[action]
        else:
            # 가능한 가장 공격적인 행동부터 순서대로 선택한다.
            if 4 in legal_actions: return 4, legal_actions[4]
            if 3 in legal_actions: return 3, legal_actions[3]
            if 2 in legal_actions: return 2, legal_actions[2]
            if 1 in legal_actions: return 1, legal_actions[1]
            return 0, 0

    # 한 핸드가 끝난 후, 최종 보상을 바탕으로 모델을 학습시키는 메소드이다.
    def learn_from_hand(self, final_reward: float):
        self.memory.finish_hand(final_reward)
        if self.model is not None and len(self.memory.experiences) >= 128:
            try:
                # 경험 데이터가 충분히 쌓이면 모델의 learn 메소드를 호출하여 학습을 진행한다.
                self.model.learn(total_timesteps=10000, reset_num_timesteps=False)
                self.learning_stats['learning_episodes'] += 1
            except Exception as e:
                print(f"⚠️ 학습 오류: {e}")

    # Gym 환경의 필수 메소드들이다.
    # 환경을 초기 상태로 리셋한다.
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_observation = np.zeros(60, dtype=np.float32)
        return self.current_observation, {}
    
    # 에이전트의 행동을 받아 환경을 다음 상태로 전환한다. (이 코드에서는 게임 로직이 대부분 처리)
    def step(self, action):
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return self.current_observation, reward, done, truncated, info

    # 게임이 끝난 후 승패 여부에 따라 통계를 업데이트하는 메소드이다.
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
        
        # 최근 100개의 승률 기록만 유지한다.
        if len(self.learning_stats['win_rate_history']) > 100:
            self.learning_stats['win_rate_history'] = self.learning_stats['win_rate_history'][-100:]

    # 학습된 모델과 통계 데이터를 파일에 저장하는 메소드이다.
    def save_model(self):
        if self.model is not None:
            try:
                self.model.save(self.model_path)
                with open(f"{self.model_path}_stats.json", 'w') as f:
                    json.dump(self.learning_stats, f, indent=2)
                print(f"💾 모델이 저장되었습니다: {self.model_path}")
            except Exception as e:
                print(f"❌ 모델 저장 실패: {e}")

    # 현재 AI의 학습 진행 상황을 문자열로 반환하는 메소드이다.
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

# 실제 포커 게임의 진행을 관리하는 메인 클래스이다.
class LearningPokerGame:
    def __init__(self, ai_model_path: str = None):
        self.deck = Deck()
        self.player_chips = 1000
        self.ai_chips = 1000
        self.pot = 0
        self.community_cards = []
        self.player_hand = []
        self.ai_hand = []
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        self.ai = AdaptivePokerAI(ai_model_path) # AI 에이전트를 생성한다.
        self.current_observation = None
        print("🤖 실시간 학습 포커 AI(v2)가 준비되었습니다!")
        print(self.ai.get_learning_progress())

    # 현재 상황에서 AI가 할 수 있는 합법적인 행동 목록을 반환하는 메소드이다.
    def _get_legal_actions(self) -> Dict[int, int]:
        legal_actions = {0: 0} # Fold는 항상 가능하다.
        need_to_call = self.current_bet - self.ai_bet

        if self.ai_chips >= need_to_call:
            legal_actions[1] = need_to_call # Call

            # Raise 옵션들을 계산한다.
            # 팟의 50% 레이즈
            raise_50_amount = int(self.pot * 0.5)
            if self.ai_chips >= need_to_call + raise_50_amount:
                legal_actions[2] = need_to_call + raise_50_amount

            # 팟의 100% 레이즈
            raise_100_amount = self.pot
            if self.ai_chips >= need_to_call + raise_100_amount:
                legal_actions[3] = need_to_call + raise_100_amount

            # 올인
            legal_actions[4] = self.ai_chips

        return legal_actions

    # AI의 행동을 결정하고 실행하는 과정을 관리하는 메소드이다.
    def ai_action(self) -> Tuple[str, int]:
        observation = self.get_observation() # 현재 게임 상태를 관측 벡터로 변환한다。
        self.current_observation = observation
        self.ai.ai_chips = self.ai_chips # AI 객체에 현재 칩 정보를 업데이트한다。

        legal_actions = self._get_legal_actions() # 가능한 행동 목록을 가져온다。
        
        action_code, amount = self.ai.decide_action(observation, legal_actions) # AI에게 행동 결정을 요청한다。

        # AI의 경험을 메모리에 저장한다. (보상은 핸드 종료 시 결정된다)
        self.ai.memory.add_experience(observation, action_code, 0.0)
        
        action_map = {0: 'fold', 1: 'call', 2: 'raise', 3: 'raise', 4: 'raise'}
        return action_map[action_code], amount

    # ... (start_new_hand, get_observation, deal_*, get_best_hand, determine_winner 등은 이전과 거의 동일) ...
    # 새 핸드를 시작하기 위해 모든 변수를 초기화하는 메소드이다.
    def start_new_hand(self):
        self.deck.reset()
        self.community_cards = []
        self.player_hand = []
        self.ai_hand = []
        self.pot = 0
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        self.ai.memory.start_hand()
        small_blind = 10
        big_blind = 20
        self.player_chips -= small_blind
        self.ai_chips -= big_blind
        self.pot = small_blind + big_blind
        self.player_bet = small_blind
        self.ai_bet = big_blind
        self.current_bet = big_blind
        for _ in range(2):
            self.player_hand.append(self.deck.deal_card())
            self.ai_hand.append(self.deck.deal_card())

    # 현재 게임 상태를 AI가 이해할 수 있는 60개의 숫자 배열(관측 벡터)로 변환하는 메소드이다.
    def get_observation(self) -> np.ndarray:
        obs = np.zeros(60, dtype=np.float32)
        # AI의 핸드 카드 정보를 인코딩한다。
        for i, card in enumerate(self.ai_hand):
            if i < 2:
                suit_idx = list(Suit).index(card.suit)
                obs[i*4 + suit_idx] = 1.0 # 무늬는 원-핫 인코딩
                obs[8 + i] = card.rank.value / 14.0 # 랭크는 0~1 사이로 정규화
        # 커뮤니티 카드 정보를 인코딩한다。
        for i, card in enumerate(self.community_cards):
            if i < 5:
                suit_idx = list(Suit).index(card.suit)
                obs[10 + i*4 + suit_idx] = 1.0
                obs[30 + i] = card.rank.value / 14.0
        
        total_chips = self.ai_chips + self.player_chips + self.pot
        total_chips = max(total_chips, 1)
        # 팟 크기, 칩 스택 등 게임의 주요 상태 정보를 정규화하여 인코딩한다。
        obs[35] = np.clip(self.pot / total_chips, 0.0, 1.0)
        obs[36] = np.clip(self.ai_chips / total_chips, 0.0, 1.0)
        obs[37] = np.clip(self.player_chips / total_chips, 0.0, 1.0)
        obs[38] = np.clip(self.current_bet / max(self.pot, 1), 0.0, 1.0)
        obs[39] = np.clip(self.ai_bet / total_chips, 0.0, 1.0)
        obs[40] = np.clip(self.player_bet / total_chips, 0.0, 1.0)
        # 게임 단계를 인코딩한다。
        obs[41] = len(self.community_cards) / 5.0
        obs[42] = 1.0 if len(self.community_cards) == 0 else 0.0 # 프리플롭
        obs[43] = 1.0 if len(self.community_cards) == 3 else 0.0 # 플롭
        obs[44] = 1.0 if len(self.community_cards) == 4 else 0.0 # 턴
        obs[45] = 1.0 if len(self.community_cards) == 5 else 0.0 # 리버
        need_to_call = max(0, self.current_bet - self.ai_bet)
        obs[46] = np.clip(need_to_call / max(self.pot, 1), 0.0, 1.0)
        obs[47] = 1.0 if self.current_bet > 0 else 0.0
        # AI의 과거 학습 데이터를 인코딩한다。
        obs[48] = self.ai.win_rate
        obs[49] = len(self.ai.memory.experiences) / 10000.0
        return obs

    # 플롭(커뮤니티 카드 3장)을 딜링하는 메소드이다.
    def deal_flop(self):
        for _ in range(3):
            self.community_cards.append(self.deck.deal_card())
    # 턴(커뮤니티 카드 1장)을 딜링하는 메소드이다.
    def deal_turn(self):
        self.community_cards.append(self.deck.deal_card())
    # 리버(커뮤니티 카드 1장)을 딜링하는 메소드이다.
    def deal_river(self):
        self.community_cards.append(self.deck.deal_card())

    # 주어진 핸드 카드와 커뮤니티 카드를 조합하여 만들 수 있는 최상의 5장 족보를 반환하는 메소드이다.
    def get_best_hand(self, hole_cards: List[Card]) -> PokerHand:
        all_cards = hole_cards + self.community_cards
        if len(all_cards) < 5:
            return None
        best_hand = None
        for combo in combinations(all_cards, 5):
            hand = PokerHand(list(combo))
            if best_hand is None or hand > best_hand:
                best_hand = hand
        return best_hand

    # 쇼다운에서 플레이어와 AI 중 최종 승자를 결정하는 메소드이다.
    def determine_winner(self) -> str:
        player_best = self.get_best_hand(self.player_hand)
        ai_best = self.get_best_hand(self.ai_hand)
        if player_best is None or ai_best is None: return "tie"
        if player_best > ai_best: return "player"
        elif ai_best > player_best: return "ai"
        else: return "tie"

    # 현재 게임 상태를 콘솔에 출력하는 메소드이다.
    def display_game_state(self, hide_ai_cards=True):
        print("\n" + "="*50)
        print(f"💰 팟: {self.pot}칩")
        print(f"🎯 현재 베팅: {self.current_bet}칩")
        print(f"👤 플레이어 칩: {self.player_chips}칩 (베팅: {self.player_bet}칩)")
        print(f"🤖 AI 칩: {self.ai_chips}칩 (베팅: {self.ai_bet}칩)")
        print(f"🧠 AI 승률: {self.ai.win_rate:.1%} ({self.ai.games_played}게임)")
        print("-"*50)
        if hide_ai_cards:
            print(f"🤖 AI 핸드: [🂠, 🂠]")
        else:
            print(f"🤖 AI 핸드: {self.ai_hand}")
        print(f"👤 플레이어 핸드: {self.player_hand}")
        if self.community_cards:
            print(f"🃏 커뮤니티 카드: {self.community_cards}")
        print("="*50)

    # 한 라운드의 베팅을 처리하는 메소드이다. 플레이어 입력과 AI 행동을 번갈아 처리한다.
    def betting_round(self) -> bool:
        action_taken = False
        while True:
            self.display_game_state()
            need_to_call = self.current_bet - self.player_bet
            if need_to_call > 0:
                print(f"\n💡 콜하려면 {need_to_call}칩이 필요합니다.")
            print("\n🎮 행동을 선택하세요:")
            print("1. 폴드 (포기)")
            print(f"\n2. {'콜 (' + str(need_to_call) + '칩)' if need_to_call > 0 else '체크'}")
            print("\n3. 레이즈")
            print("\n4. AI 학습 현황 보기")
            try:
                choice = int(input("선택 (1-4): "))
                if choice == 4:
                    print(self.ai.get_learning_progress())
                    continue
                elif choice == 1: # 플레이어가 폴드한 경우
                    self.ai_chips += self.pot
                    self.ai.learn_from_hand(1.0) # AI 승리, 보상 +1
                    self.ai.update_stats(True)
                    return False
                elif choice == 2: # 플레이어가 콜 또는 체크한 경우
                    if need_to_call > 0:
                        call_amount = min(need_to_call, self.player_chips)
                        self.player_chips -= call_amount
                        self.player_bet += call_amount
                        self.pot += call_amount
                    action_taken = True
                    break
                elif choice == 3: # 플레이어가 레이즈한 경우
                    min_raise = max(20, need_to_call + 20)
                    max_raise = self.player_chips
                    if min_raise > max_raise:
                        print("❌ 레이즈할 수 없습니다. 올인하거나 콜하세요.")
                        continue
                    raise_amount = int(input(f"레이즈 금액 입력 ({min_raise}~{max_raise}): "))
                    if raise_amount < min_raise or raise_amount > max_raise:
                        print("❌ 잘못된 금액입니다.")
                        continue
                    self.player_chips -= raise_amount
                    self.player_bet += raise_amount
                    self.pot += raise_amount
                    self.current_bet = self.player_bet
                    action_taken = True
                    break
                else:
                    print("❌ 잘못된 선택입니다.")
            except ValueError:
                print("❌ 숫자를 입력하세요.")

        need_to_call = self.current_bet - self.ai_bet
        if need_to_call == 0 and action_taken:
            print("🤖 AI가 체크했습니다.")
            return True

        ai_action, ai_amount = self.ai_action() # AI의 행동을 가져온다。
        action_map_str = {0: "폴드", 1: "콜", 2: "팟의 50% 레이즈", 3: "팟의 100% 레이즈", 4: "올인"}
        print(f"🤖 AI의 선택: {action_map_str.get(ai_action, '알 수 없음')} ({ai_amount}칩)")

        if ai_action == 0: # AI가 폴드한 경우
            self.player_chips += self.pot
            self.ai.learn_from_hand(-1.0) # AI 패배, 보상 -1
            self.ai.update_stats(False)
            return False
        elif ai_action == 1: # AI가 콜한 경우
            if ai_amount > 0:
                self.ai_chips -= ai_amount
                self.ai_bet += ai_amount
                self.pot += ai_amount
            return True
        else: # AI가 레이즈한 경우, 베팅 라운드를 다시 진행한다.
            self.ai_chips -= ai_amount
            self.ai_bet += ai_amount
            self.pot += ai_amount
            self.current_bet = self.ai_bet
            return self.betting_round()
        
    def get_best_hand(self, hole_cards: List[Card]) -> PokerHand:
        """7장 카드에서 최고의 5장 조합 찾기"""
        all_cards = hole_cards + self.community_cards
        if len(all_cards) < 5:
            return None
        
        best_hand = None
        for combo in combinations(all_cards, 5):
            hand = PokerHand(list(combo))
            if best_hand is None or hand > best_hand:
                best_hand = hand
        return best_hand

    # 한 핸드의 게임을 시작부터 쇼다운까지 진행하는 메소드이다.
    def play_hand(self):
        print("\n🎴 새로운 핸드를 시작합니다!")
        self.start_new_hand()
        if not self.betting_round(): return
        print("\n🎲 플롭 카드를 공개합니다!")
        self.deal_flop()
        self.current_bet = 0; self.player_bet = 0; self.ai_bet = 0
        if not self.betting_round(): return
        print("\n🎲 턴 카드를 공개합니다!")
        self.deal_turn()
        self.current_bet = 0; self.player_bet = 0; self.ai_bet = 0
        if not self.betting_round(): return
        print("\n🎲 리버 카드를 공개합니다!")
        self.deal_river()
        self.current_bet = 0; self.player_bet = 0; self.ai_bet = 0
        if not self.betting_round(): return
        
        player_best = self.get_best_hand(self.player_hand)
        ai_best = self.get_best_hand(self.ai_hand)
        winner = self.determine_winner()
        
        print("\n" + "="*20 + " 🎭 쇼다운 " + "="*20)
        self.display_game_state(hide_ai_cards=False)
        winner = self.determine_winner()
        if winner == "player":
            print("\n🎉 플레이어 승리!")
            self.player_chips += self.pot
            self.ai.learn_from_hand(-1.0) # AI 패배, 보상 -1
            self.ai.update_stats(False)
        elif winner == "ai":
            print("\n🤖 AI 승리!")
            self.ai_chips += self.pot
            self.ai.learn_from_hand(1.0) # AI 승리, 보상 +1
            self.ai.update_stats(True)
        else:
            print("\n🤝 무승부!")
            self.player_chips += self.pot // 2
            self.ai_chips += self.pot // 2
            self.ai.learn_from_hand(0.1) # 무승부는 작은 긍정적 보상
            
        
        if player_best:
            print(f"\n👤 플레이어 베스트 핸드: {player_best.cards} ({player_best.rank.name})")
        else:
            print("\n👤 플레이어 베스트 핸드: (핸드 없음)")

        if ai_best:
            print(f"🤖 AI 베스트 핸드: {ai_best.cards} ({ai_best.rank.name})")
        else:
            print("🤖 AI 베스트 핸드: (핸드 없음)")

    # 플레이어가 종료를 원할 때까지 게임을 계속 진행하는 메인 루프이다.
    def play_game(self):
        print("🃏 실시간 학습 포커 AI(v2) 대전에 오신 것을 환영합니다! 🃏")
        hand_count = 0
        save_interval = 10 # 10 핸드마다 모델을 자동 저장한다。
        while self.player_chips > 0 and self.ai_chips > 0:
            print(f"\n💰 현재 칩 상황 - 👤 플레이어: {self.player_chips}, 🤖 AI: {self.ai_chips}")
            print("\n🎮 메뉴: 1.AI 학습 현황, 2.새 핸드 시작, 3.모델 저장, 4.게임 종료")
            try:
                choice = int(input("선택 (1-4): "))
                if choice == 2:
                    self.play_hand()
                    hand_count += 1
                    if hand_count % save_interval == 0:
                        print(f"\n💾 {save_interval}핸드 완료! 자동 저장 중...")
                        self.ai.save_model()
                elif choice == 1:
                    print(self.ai.get_learning_progress())
                elif choice == 3:
                    self.ai.save_model()
                elif choice == 4:
                    break
            except ValueError:
                print("❌ 숫자를 입력하세요.")
        print("\n🏁 게임 종료!")
        self.ai.save_model() # 게임 종료 시 최종 학습 내용을 저장한다。
        print(self.ai.get_learning_progress())


# 이 스크립트가 직접 실행될 때 main 로직을 수행한다.
if __name__ == "__main__":
    # 학습된 모델이 저장될 파일 경로이다.
    model_path = "learning_poker_ai"
    
    print("🚀 실시간 학습 포커 AI를 시작합니다!")
    print("=" * 50)
    
    # 게임 객체를 생성하고 게임을 시작한다.
    game = LearningPokerGame(model_path)
    game.play_game()
    
    print("\n🎯 게임을 종료합니다. 다음에 다시 시작하면 AI가 학습한 내용을 기억합니다!")
