import numpy as np
import random
from typing import List, Tuple, Dict
from enum import Enum
from itertools import combinations

# stable_baselines3가 설치되어 있어야 합니다
try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    print("Warning: stable_baselines3가 설치되지 않았습니다. pip install stable-baselines3로 설치해주세요.")
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

class PokerGame:
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
        
        # PPO 모델 로드
        self.ai_model = None
        if ai_model_path and PPO_AVAILABLE:
            try:
                self.ai_model = PPO.load(ai_model_path)
                print(f"✅ PPO AI 모델을 성공적으로 로드했습니다: {ai_model_path}")
            except Exception as e:
                print(f"❌ AI 모델 로드 실패: {e}")
                print("🎲 기본 랜덤 AI를 사용합니다.")
                self.ai_model = None
        else:
            if not PPO_AVAILABLE:
                print("⚠️ stable_baselines3가 설치되지 않았습니다.")
            print("🎲 기본 랜덤 AI를 사용합니다.")
            self.ai_model = None
    
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
        
        return obs
    
    def ai_action(self) -> Tuple[str, int]:
        """AI의 행동 결정"""
        need_to_call = max(0, self.current_bet - self.ai_bet)
        
        if self.ai_model is None:
            # 랜덤 AI
            if need_to_call > self.ai_chips:
                return 'fold', 0
            
            actions = ['fold', 'call', 'raise']
            weights = [0.2, 0.5, 0.3]  # 폴드: 20%, 콜: 50%, 레이즈: 30%
            action = random.choices(actions, weights=weights)[0]
            
            if action == 'fold':
                return 'fold', 0
            elif action == 'call':
                call_amount = min(need_to_call, self.ai_chips)
                return 'call', call_amount
            else:  # raise
                min_raise = need_to_call + 20
                max_raise = min(self.ai_chips, need_to_call + 100)
                if min_raise <= max_raise:
                    raise_amount = random.randint(min_raise, max_raise)
                    return 'raise', raise_amount
                else:
                    # 레이즈할 수 없으면 콜
                    call_amount = min(need_to_call, self.ai_chips)
                    return 'call', call_amount
        
        # PPO 모델 사용
        try:
            obs = self.get_observation()
            action, _ = self.ai_model.predict(obs, deterministic=False)
            
            # 액션 해석 (0: fold, 1: call, 2: raise)
            if action == 0:  # fold
                return 'fold', 0
            elif action == 1:  # call
                call_amount = min(need_to_call, self.ai_chips)
                return 'call', call_amount
            else:  # raise (action == 2)
                min_raise = need_to_call + 20
                max_raise = min(self.ai_chips, need_to_call + random.randint(20, 100))
                if min_raise <= max_raise and min_raise <= self.ai_chips:
                    raise_amount = min_raise
                    return 'raise', raise_amount
                else:
                    # 레이즈할 수 없으면 콜
                    call_amount = min(need_to_call, self.ai_chips)
                    return 'call', call_amount
                    
        except Exception as e:
            print(f"PPO 모델 예측 오류: {e}")
            # 오류 시 랜덤 행동
            if need_to_call > self.ai_chips:
                return 'fold', 0
            else:
                call_amount = min(need_to_call, self.ai_chips)
                return 'call', call_amount
    
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
            
            try:
                choice = int(input("선택 (1-3): "))
                
                if choice == 1:  # 폴드
                    print("❌ 플레이어가 폴드했습니다.")
                    self.ai_chips += self.pot
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
        
        if winner == "player":
            print("\n🎉 플레이어 승리! 팟을 가져갑니다!")
            self.player_chips += self.pot
        elif winner == "ai":
            print("\n🤖 AI 승리! AI가 팟을 가져갑니다!")
            self.ai_chips += self.pot
        else:
            print("\n🤝 무승부! 팟을 나누어 가져갑니다!")
            self.player_chips += self.pot // 2
            self.ai_chips += self.pot // 2
    
    def play_game(self):
        """메인 게임 루프"""
        print("🃏 포커 AI 대전에 오신 것을 환영합니다! 🃏")
        print("텍사스 홀덤 포커로 AI와 1대1 대결을 펼치세요!")
        print(f"학습된 PPO 모델: {'✅ 사용 중' if self.ai_model else '❌ 랜덤 AI 사용'}")
        
        while self.player_chips > 0 and self.ai_chips > 0:
            print(f"\n💰 현재 칩 상황 - 👤 플레이어: {self.player_chips}, 🤖 AI: {self.ai_chips}")
            
            play_again = input("\n🎮 새 핸드를 시작하시겠습니까? (y/n): ").lower()
            if play_again != 'y':
                break
            
            self.play_hand()
        
        # 게임 종료
        print("\n" + "="*50)
        print("🏁 게임 종료!")
        if self.player_chips > self.ai_chips:
            print("🎉 축하합니다! 플레이어가 최종 승리했습니다!")
        elif self.ai_chips > self.player_chips:
            print("🤖 AI가 최종 승리했습니다!")
        else:
            print("🤝 무승부입니다!")
        
        print(f"💰 최종 칩 - 👤 플레이어: {self.player_chips}, 🤖 AI: {self.ai_chips}")

# 사용 예시
if __name__ == "__main__":
    # PPO 모델 파일 경로 (확장자 없이)
    model_path = "final_poker_model"  # final_poker_model.zip 파일
    
    # 게임 시작
    game = PokerGame(model_path)
    game.play_game()