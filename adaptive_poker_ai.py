import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from enum import Enum
from itertools import combinations
import pickle
import os
from collections import deque
import json

# stable_baselines3가 설치되어 있어야 합니다
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    from gymnasium import spaces
    PPO_AVAILABLE = True
except ImportError:
    print("Warning: stable_baselines3가 설치되지 않았습니다. pip install stable-baselines3 gymnasium로 설치해주세요.")
    PPO_AVAILABLE = False

class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"  
    SPADES = "♠"

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

class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
    
    def __str__(self):
        rank_str = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 
                   9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
        return f"{rank_str[self.rank.value]}{self.suit.value}"
    
    def __repr__(self):
        return str(self)

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

class PokerHand:
    def __init__(self, cards: List[Card]):
        self.cards = sorted(cards, key=lambda x: x.rank.value, reverse=True)
        self.rank, self.value = self._evaluate_hand()
    
    def _evaluate_hand(self) -> Tuple[HandRank, List[int]]:
        ranks = [card.rank.value for card in self.cards]
        suits = [card.suit for card in self.cards]
        rank_counts = {}
        
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        is_flush = len(set(suits)) == 1
        is_straight = self._is_straight(ranks)
        
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(rank_counts.keys(), key=lambda x: (rank_counts[x], x), reverse=True)
        
        if is_straight and is_flush:
            if min(ranks) == 10:
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
    
    def _is_straight(self, ranks: List[int]) -> bool:
        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) != 5:
            return False
        
        # 일반적인 스트레이트
        if unique_ranks[-1] - unique_ranks[0] == 4:
            return True
        
        # A-2-3-4-5 스트레이트 (로우 스트레이트)
        if unique_ranks == [2, 3, 4, 5, 14]:
            return True
        
        return False
    
    def __gt__(self, other):
        if self.rank.value != other.rank.value:
            return self.rank.value > other.rank.value
        return self.value > other.value

class Deck:
    def __init__(self):
        self.cards = []
        self.reset()
    
    def reset(self):
        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]
        random.shuffle(self.cards)
    
    def deal_card(self) -> Card:
        return self.cards.pop()

class LearningMemory:
    """AI의 학습 메모리를 관리하는 클래스"""
    def __init__(self, max_size: int = 10000):
        self.experiences = deque(maxlen=max_size)
        self.current_hand_data = []
        
    def start_hand(self):
        """새 핸드 시작"""
        self.current_hand_data = []
    
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
    
    def finish_hand(self, final_reward: float):
        """핸드 종료 시 경험을 메모리에 저장"""
        # 보상을 역전파하여 각 행동에 할당
        for i, exp in enumerate(self.current_hand_data):
            # 최종 보상을 거리에 따라 할인하여 적용
            discount_factor = 0.95 ** (len(self.current_hand_data) - i - 1)
            exp['final_reward'] = final_reward * discount_factor
            self.experiences.append(exp)
        
        self.current_hand_data = []
    
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

class AdaptivePokerAI:
    """실시간으로 학습하는 포커 AI"""
    def __init__(self, model_path: str = None, learning_rate: float = 0.0003):
        self.learning_rate = learning_rate
        self.memory = LearningMemory()
        self.model = None
        self.win_rate = 0.5
        self.games_played = 0
        self.wins = 0
        self.model_path = model_path or "adaptive_poker_ai"
        
        # 학습 통계
        self.learning_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'win_rate_history': [],
            'learning_episodes': 0
        }
        
        if PPO_AVAILABLE:
            self._initialize_model()
        else:
            print("⚠️ PPO를 사용할 수 없습니다. 단순 적응 AI를 사용합니다.")
    
    def _initialize_model(self):
        """PPO 모델 초기화"""
        try:
            # 기존 모델이 있으면 로드
            if os.path.exists(f"{self.model_path}.zip"):
                self.model = PPO.load(self.model_path)
                print(f"✅ 기존 모델을 로드했습니다: {self.model_path}")
                
                # 통계 로드
                if os.path.exists(f"{self.model_path}_stats.json"):
                    with open(f"{self.model_path}_stats.json", 'r') as f:
                        self.learning_stats = json.load(f)
                        self.games_played = self.learning_stats['games_played']
                        self.wins = self.learning_stats['wins']
                        if self.games_played > 0:
                            self.win_rate = self.wins / self.games_played
            else:
                # 새 모델 생성
                observation_space = spaces.Box(low=0, high=1, shape=(60,), dtype=np.float32)
                action_space = spaces.Discrete(3)  # 0: fold, 1: call, 2: raise
                
                self.model = PPO('MlpPolicy', 
                               DummyVecEnv([lambda: self]),
                               learning_rate=self.learning_rate,
                               n_steps=2048,
                               batch_size=64,
                               n_epochs=10,
                               verbose=1)
                print("🆕 새로운 PPO 모델을 생성했습니다.")
                
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
            self.model = None
    
    def decide_action(self, observation: np.ndarray, legal_actions: List[str], 
                     amounts: Dict[str, int]) -> Tuple[str, int]:
        """AI의 행동 결정"""
        if self.model is None:
            # 단순 적응 AI
            return self._simple_adaptive_action(legal_actions, amounts)
        
        try:
            action_probs, _ = self.model.predict(observation, deterministic=False)
            action = int(action_probs[0]) if hasattr(action_probs, '__len__') else int(action_probs)
            
            # 액션을 실제 게임 행동으로 변환
            if action == 0 and 'fold' in legal_actions:
                return 'fold', 0
            elif action == 1 and 'call' in legal_actions:
                return 'call', amounts.get('call', 0)
            elif action == 2 and 'raise' in legal_actions:
                return 'raise', amounts.get('raise', amounts.get('call', 0) + 20)
            else:
                # 유효하지 않은 액션의 경우 콜 또는 폴드
                if 'call' in legal_actions:
                    return 'call', amounts.get('call', 0)
                else:
                    return 'fold', 0
                    
        except Exception as e:
            print(f"⚠️ AI 예측 오류: {e}")
            return self._simple_adaptive_action(legal_actions, amounts)
    
    def _simple_adaptive_action(self, legal_actions: List[str], amounts: Dict[str, int]) -> Tuple[str, int]:
        """단순 적응형 AI (PPO 없이도 동작)"""
        # 승률에 따라 공격성 조절
        aggression = min(0.8, max(0.2, self.win_rate))
        
        if 'fold' in legal_actions and random.random() < (1 - aggression) * 0.3:
            return 'fold', 0
        elif 'raise' in legal_actions and random.random() < aggression * 0.4:
            return 'raise', amounts.get('raise', amounts.get('call', 0) + 20)
        elif 'call' in legal_actions:
            return 'call', amounts.get('call', 0)
        else:
            return 'fold', 0
    
    def learn_from_hand(self, final_reward: float):
        """핸드 결과로부터 학습"""
        self.memory.finish_hand(final_reward)
        
        if self.model is not None and len(self.memory.experiences) >= 64:
            try:
                # 배치 학습 수행
                batch = self.memory.get_batch(64)
                if batch is not None:
                    self.model.learn(total_timesteps=64, reset_num_timesteps=False)
                    self.learning_stats['learning_episodes'] += 1
                    
            except Exception as e:
                print(f"⚠️ 학습 오류: {e}")
    
    def update_stats(self, won: bool):
        """게임 통계 업데이트"""
        self.games_played += 1
        if won:
            self.wins += 1
        
        self.win_rate = self.wins / self.games_played
        
        # 통계 업데이트
        self.learning_stats['games_played'] = self.games_played
        self.learning_stats['wins'] = self.wins
        self.learning_stats['losses'] = self.games_played - self.wins
        self.learning_stats['win_rate_history'].append(self.win_rate)
        
        # 최근 10게임만 유지
        if len(self.learning_stats['win_rate_history']) > 10:
            self.learning_stats['win_rate_history'] = self.learning_stats['win_rate_history'][-10:]
    
    def save_model(self):
        """모델과 통계 저장"""
        if self.model is not None:
            try:
                self.model.save(self.model_path)
                
                # 통계 저장
                with open(f"{self.model_path}_stats.json", 'w') as f:
                    json.dump(self.learning_stats, f, indent=2)
                    
                print(f"💾 모델이 저장되었습니다: {self.model_path}")
            except Exception as e:
                print(f"❌ 모델 저장 실패: {e}")
    
    def get_learning_progress(self) -> str:
        """학습 진행 상황 반환"""
        if self.games_played == 0:
            return "🆕 아직 게임을 시작하지 않았습니다."
        
        recent_win_rate = np.mean(self.learning_stats['win_rate_history'][-5:]) if len(self.learning_stats['win_rate_history']) >= 5 else self.win_rate
        
        progress = f"""
🧠 AI 학습 현황:
📊 총 게임: {self.games_played}게임
🏆 승률: {self.win_rate:.1%} ({self.wins}승 {self.games_played-self.wins}패)
📈 최근 승률: {recent_win_rate:.1%}
🎓 학습 에피소드: {self.learning_stats['learning_episodes']}회
💾 경험 데이터: {len(self.memory.experiences)}개
        """
        
        return progress.strip()

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
        
        # 적응형 AI 초기화
        self.ai = AdaptivePokerAI(ai_model_path)
        self.current_observation = None
        
        print("🤖 실시간 학습 포커 AI가 준비되었습니다!")
        print(self.ai.get_learning_progress())
    
    def start_new_hand(self):
        """새 핸드 시작"""
        self.deck.reset()
        self.community_cards = []
        self.player_hand = []
        self.ai_hand = []
        self.pot = 0
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        
        # AI 메모리 초기화
        self.ai.memory.start_hand()
        
        # 블라인드 베팅
        small_blind = 10
        big_blind = 20
        
        self.player_chips -= small_blind
        self.ai_chips -= big_blind
        self.pot = small_blind + big_blind
        self.player_bet = small_blind
        self.ai_bet = big_blind
        self.current_bet = big_blind
        
        # 카드 배분
        for _ in range(2):
            self.player_hand.append(self.deck.deal_card())
            self.ai_hand.append(self.deck.deal_card())
    
    def get_observation(self) -> np.ndarray:
        """PPO 모델을 위한 관찰 벡터 생성"""
        obs = np.zeros(60, dtype=np.float32)
        
        # AI 핸드 카드 인코딩 (8개 값: 각 카드당 슈트 4개)
        for i, card in enumerate(self.ai_hand):
            if i < 2:  # 홀 카드는 2장
                suit_idx = list(Suit).index(card.suit)
                obs[i*4 + suit_idx] = 1.0
                obs[8 + i] = card.rank.value / 14.0  # 정규화된 랭크
        
        # 커뮤니티 카드 인코딩 (20개 슈트 + 5개 랭크)
        for i, card in enumerate(self.community_cards):
            if i < 5:  # 최대 5장
                suit_idx = list(Suit).index(card.suit)
                obs[10 + i*4 + suit_idx] = 1.0
                obs[30 + i] = card.rank.value / 14.0
        
        # 게임 상태 정보 (25개 값)
        obs[35] = self.pot / 2000.0  # 정규화된 팟 크기
        obs[36] = self.ai_chips / 2000.0  # AI 칩
        obs[37] = self.player_chips / 2000.0  # 플레이어 칩
        obs[38] = self.current_bet / 200.0  # 현재 베팅
        obs[39] = self.ai_bet / 200.0  # AI 베팅
        obs[40] = self.player_bet / 200.0  # 플레이어 베팅
        obs[41] = len(self.community_cards) / 5.0  # 게임 단계
        
        # 베팅 라운드 정보
        obs[42] = 1.0 if len(self.community_cards) == 0 else 0.0  # 프리플롭
        obs[43] = 1.0 if len(self.community_cards) == 3 else 0.0  # 플롭
        obs[44] = 1.0 if len(self.community_cards) == 4 else 0.0  # 턴
        obs[45] = 1.0 if len(self.community_cards) == 5 else 0.0  # 리버
        
        # 베팅 상황
        need_to_call = max(0, self.current_bet - self.ai_bet)
        obs[46] = need_to_call / 200.0  # 콜해야 할 금액
        obs[47] = 1.0 if need_to_call > 0 else 0.0  # 콜 필요 여부
        
        # 추가 특성들
        obs[48] = self.ai.win_rate  # AI의 현재 승률
        obs[49] = len(self.ai.memory.experiences) / 10000.0  # 경험 데이터 양
        
        return obs

    def ai_action(self) -> Tuple[str, int]:
        """AI의 행동 결정"""
        observation = self.get_observation()
        self.current_observation = observation
        
        need_to_call = max(0, self.current_bet - self.ai_bet)
        
        # 가능한 행동과 금액 정의
        legal_actions = ['fold']
        amounts = {}
        
        if need_to_call <= self.ai_chips:
            legal_actions.append('call')
            amounts['call'] = need_to_call
        
        min_raise = need_to_call + 20
        if min_raise <= self.ai_chips:
            legal_actions.append('raise')
            amounts['raise'] = min(min_raise + random.randint(0, 80), self.ai_chips)
        
        # AI 행동 결정
        action, amount = self.ai.decide_action(observation, legal_actions, amounts)
        
        # 경험 저장 (즉시 보상은 0, 나중에 핸드 결과로 업데이트)
        action_idx = {'fold': 0, 'call': 1, 'raise': 2}.get(action, 1)
        self.ai.memory.add_experience(observation, action_idx, 0.0)
        
        return action, amount
    
    def deal_flop(self):
        """플롭 (처음 3장의 커뮤니티 카드) 딜"""
        for _ in range(3):
            self.community_cards.append(self.deck.deal_card())
    
    def deal_turn(self):
        """턴 (4번째 커뮤니티 카드) 딜"""
        self.community_cards.append(self.deck.deal_card())
    
    def deal_river(self):
        """리버 (5번째 커뮤니티 카드) 딜"""
        self.community_cards.append(self.deck.deal_card())
    
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

    def determine_winner(self) -> str:
        """승자 결정"""
        player_best = self.get_best_hand(self.player_hand)
        ai_best = self.get_best_hand(self.ai_hand)

        if player_best is None or ai_best is None:
            return "tie"

        if player_best > ai_best:
            return "player"
        elif ai_best > player_best:
            return "ai"
        else:
            return "tie"
    
    def display_game_state(self, hide_ai_cards=True):
        """게임 상태 출력"""
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
    
    def betting_round(self) -> bool:
        """베팅 라운드 진행. True: 계속, False: 핸드 종료"""
        action_taken = False
        
        while True:
            # 플레이어 턴
            self.display_game_state()
            
            need_to_call = self.current_bet - self.player_bet
            if need_to_call > 0:
                print(f"\n💡 콜하려면 {need_to_call}칩이 필요합니다.")
            
            print("\n🎮 행동을 선택하세요:")
            print("1. 폴드 (포기)")
            print(f"2. {'콜 (' + str(need_to_call) + '칩)' if need_to_call > 0 else '체크'}")
            print("3. 레이즈")
            print("4. AI 학습 현황 보기")
            
            try:
                choice = int(input("선택 (1-4): "))
                
                if choice == 4:  # AI 학습 현황
                    print(self.ai.get_learning_progress())
                    continue
                
                elif choice == 1:  # 폴드
                    print("❌ 플레이어가 폴드했습니다.")
                    self.ai_chips += self.pot
                    
                    # AI에게 승리 보상
                    self.ai.learn_from_hand(1.0)
                    self.ai.update_stats(True)
                    return False
                
                elif choice == 2:  # 콜/체크
                    if need_to_call > 0:
                        call_amount = min(need_to_call, self.player_chips)
                        self.player_chips -= call_amount
                        self.player_bet += call_amount
                        self.pot += call_amount
                        print(f"✅ 플레이어가 {call_amount}칩을 콜했습니다.")
                    else:
                        print("✅ 플레이어가 체크했습니다.")
                    action_taken = True
                    break
                
                elif choice == 3:  # 레이즈
                    min_raise = max(20, need_to_call + 20)
                    max_raise = self.player_chips
                    
                    if min_raise > max_raise:
                        print("❌ 레이즈할 수 없습니다. 올인하거나 콜하세요.")
                        continue
                    
                    print(f"레이즈 가능 범위: {min_raise} ~ {max_raise}칩")
                    raise_amount = int(input(f"레이즈 금액 입력: "))
                    
                    if raise_amount < min_raise or raise_amount > max_raise:
                        print("❌ 잘못된 금액입니다.")
                        continue
                    
                    self.player_chips -= raise_amount
                    self.player_bet += raise_amount
                    self.pot += raise_amount
                    self.current_bet = self.player_bet
                    print(f"🚀 플레이어가 {raise_amount}칩을 레이즈했습니다.")
                    action_taken = True
                    break
                
                else:
                    print("❌ 잘못된 선택입니다.")
                    continue
                    
            except ValueError:
                print("❌ 숫자를 입력하세요.")
                continue
        
        # AI 턴
        need_to_call = self.current_bet - self.ai_bet
        
        if need_to_call == 0 and action_taken:
            print("🤖 AI가 체크했습니다.")
            return True
        
        ai_action, ai_amount = self.ai_action()
        
        if ai_action == 'fold':
            print("❌ AI가 폴드했습니다.")
            self.player_chips += self.pot
            
            # AI에게 패배 페널티
            self.ai.learn_from_hand(-1.0)
            self.ai.update_stats(False)
            return False
        
        elif ai_action == 'call':
            if ai_amount > 0:
                self.ai_chips -= ai_amount
                self.ai_bet += ai_amount
                self.pot += ai_amount
                print(f"✅ AI가 {ai_amount}칩을 콜했습니다.")
            else:
                print("✅ AI가 체크했습니다.")
            return True
        
        else:  # raise
            self.ai_chips -= ai_amount
            self.ai_bet += ai_amount
            self.pot += ai_amount
            self.current_bet = self.ai_bet
            print(f"🚀 AI가 {ai_amount}칩을 레이즈했습니다!")
            
            # 플레이어가 다시 응답해야 함
            return self.betting_round()
    
    def play_hand(self):
        """한 핸드 플레이"""
        print("\n🎴 새로운 핸드를 시작합니다!")
        self.start_new_hand()
        
        # 프리플롭 베팅
        print("\n🎯 프리플롭 베팅 라운드")
        if not self.betting_round():
            return
        
        # 플롭
        print("\n🎲 플롭 카드를 공개합니다!")
        self.deal_flop()
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        
        if not self.betting_round():
            return
        
        # 턴
        print("\n🎲 턴 카드를 공개합니다!")
        self.deal_turn()
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        
        if not self.betting_round():
            return
        
        # 리버
        print("\n🎲 리버 카드를 공개합니다!")
        self.deal_river()
        self.current_bet = 0
        self.player_bet = 0
        self.ai_bet = 0
        
        if not self.betting_round():
            return
        
        # 쇼다운
        print("\n" + "="*20 + " 🎭 쇼다운 " + "="*20)
        self.display_game_state(hide_ai_cards=False)
        
        player_best = self.get_best_hand(self.player_hand)
        ai_best = self.get_best_hand(self.ai_hand)
        
        if player_best:
            print(f"\n👤 플레이어 베스트 핸드: {player_best.cards} ({player_best.rank.name})")
        else:
            print("\n👤 플레이어 베스트 핸드: (핸드 없음)")

        if ai_best:
            print(f"🤖 AI 베스트 핸드: {ai_best.cards} ({ai_best.rank.name})")
        else:
            print("🤖 AI 베스트 핸드: (핸드 없음)")
        
        winner = self.determine_winner()
        
        # 승부 결과에 따른 보상 계산
        pot_ratio = self.pot / 2000.0  # 팟 크기에 따른 보상 조정
        
        if winner == "player":
            print("\n🎉 플레이어 승리! 팟을 가져갑니다!")
            self.player_chips += self.pot
            
            # AI에게 패배 페널티 (팟이 클수록 큰 페널티)
            reward = -0.5 - pot_ratio
            self.ai.learn_from_hand(reward)
            self.ai.update_stats(False)
            
        elif winner == "ai":
            print("\n🤖 AI 승리! AI가 팟을 가져갑니다!")
            self.ai_chips += self.pot
            
            # AI에게 승리 보상 (팟이 클수록 큰 보상)
            reward = 0.5 + pot_ratio
            self.ai.learn_from_hand(reward)
            self.ai.update_stats(True)
            
        else:
            print("\n🤝 무승부! 팟을 나누어 가져갑니다!")
            self.player_chips += self.pot // 2
            self.ai_chips += self.pot // 2
            
            # AI에게 무승부 보상 (작은 양의 보상)
            reward = 0.1
            self.ai.learn_from_hand(reward)
            # 무승부는 승률 계산에서 제외
    
    def play_game(self):
        """메인 게임 루프"""
        print("🃏 실시간 학습 포커 AI 대전에 오신 것을 환영합니다! 🃏")
        print("텍사스 홀덤 포커로 AI와 1대1 대결을 펼치세요!")
        print("🧠 AI는 매 핸드마다 당신의 플레이를 학습하며 점점 강해집니다!")
        print(f"🤖 AI 모델: {'PPO 강화학습' if self.ai.model else '적응형 규칙 기반'}")
        
        hand_count = 0
        save_interval = 5  # 5핸드마다 자동 저장
        
        while self.player_chips > 0 and self.ai_chips > 0:
            print(f"\n💰 현재 칩 상황 - 👤 플레이어: {self.player_chips}, 🤖 AI: {self.ai_chips}")
            
            print("\n🎮 메뉴:")
            print("1. 새 핸드 시작")
            print("2. AI 학습 현황 보기")
            print("3. 모델 저장")
            print("4. 게임 종료")
            
            try:
                choice = int(input("선택 (1-4): "))
                
                if choice == 1:  # 새 핸드 시작
                    self.play_hand()
                    hand_count += 1
                    
                    # 정기적으로 모델 저장
                    if hand_count % save_interval == 0:
                        print(f"\n💾 {save_interval}핸드 완료! 자동 저장 중...")
                        self.ai.save_model()
                        print("📊 현재까지의 학습 현황:")
                        print(self.ai.get_learning_progress())
                
                elif choice == 2:  # AI 학습 현황
                    print(self.ai.get_learning_progress())
                    
                    # 승률 변화 그래프 (간단한 텍스트 버전)
                    if len(self.ai.learning_stats['win_rate_history']) > 1:
                        print("\n📈 최근 승률 변화:")
                        history = self.ai.learning_stats['win_rate_history'][-10:]
                        for i, rate in enumerate(history):
                            bar_length = int(rate * 20)
                            bar = "█" * bar_length + "░" * (20 - bar_length)
                            print(f"  {i+1:2d}: {bar} {rate:.1%}")
                
                elif choice == 3:  # 모델 저장
                    self.ai.save_model()
                    print("✅ 모델이 저장되었습니다!")
                
                elif choice == 4:  # 게임 종료
                    break
                
                else:
                    print("❌ 잘못된 선택입니다.")
                    
            except ValueError:
                print("❌ 숫자를 입력하세요.")
                continue
        
        # 게임 종료
        print("\n" + "="*50)
        print("🏁 게임 종료!")
        
        # 최종 저장
        print("💾 최종 모델 저장 중...")
        self.ai.save_model()
        
        # 결과 출력
        if self.player_chips > self.ai_chips:
            print("🎉 축하합니다! 플레이어가 최종 승리했습니다!")
        elif self.ai_chips > self.player_chips:
            print("🤖 AI가 최종 승리했습니다!")
        else:
            print("🤝 무승부입니다!")
        
        print(f"💰 최종 칩 - 👤 플레이어: {self.player_chips}, 🤖 AI: {self.ai_chips}")
        
        # 최종 학습 통계
        print("\n🧠 AI 최종 학습 통계:")
        print(self.ai.get_learning_progress())
        
        # 학습 성과 분석
        if self.ai.games_played > 5:
            recent_games = min(5, len(self.ai.learning_stats['win_rate_history']))
            recent_win_rate = np.mean(self.ai.learning_stats['win_rate_history'][-recent_games:])
            initial_win_rate = np.mean(self.ai.learning_stats['win_rate_history'][:recent_games])
            
            improvement = recent_win_rate - initial_win_rate
            if improvement > 0.1:
                print(f"📈 AI가 크게 성장했습니다! 승률이 {improvement:.1%} 향상되었습니다.")
            elif improvement > 0.05:
                print(f"📊 AI가 조금 성장했습니다. 승률이 {improvement:.1%} 향상되었습니다.")
            elif improvement < -0.1:
                print(f"📉 AI가 어려움을 겪고 있습니다. 승률이 {abs(improvement):.1%} 하락했습니다.")
            else:
                print("📊 AI가 안정적인 성능을 보이고 있습니다.")

# 사용 예시
if __name__ == "__main__":
    # 모델 파일 경로 (확장자 없이)
    model_path = "learning_poker_ai"
    
    print("🚀 실시간 학습 포커 AI를 시작합니다!")
    print("=" * 50)
    
    # 게임 시작
    game = LearningPokerGame(model_path)
    game.play_game()
    
    print("\n🎯 게임을 종료합니다. 다음에 다시 시작하면 AI가 학습한 내용을 기억합니다!")
