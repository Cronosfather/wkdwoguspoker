import numpy as np
from typing import Dict, Tuple, List
from itertools import combinations

from .core import Deck, PokerHand, Card, Suit
from .ai_agent import AdaptivePokerAI

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
        self.ai = AdaptivePokerAI(ai_model_path)
        self.current_observation = None
        print("🤖 실시간 학습 포커 AI(v2)가 준비되었습니다!")
        print(self.ai.get_learning_progress())

    def _get_legal_actions(self) -> Dict[int, int]:
        legal_actions = {0: 0}
        need_to_call = self.current_bet - self.ai_bet

        if self.ai_chips >= need_to_call:
            legal_actions[1] = need_to_call

            raise_50_amount = int(self.pot * 0.5)
            if self.ai_chips >= need_to_call + raise_50_amount:
                legal_actions[2] = need_to_call + raise_50_amount

            raise_100_amount = self.pot
            if self.ai_chips >= need_to_call + raise_100_amount:
                legal_actions[3] = need_to_call + raise_100_amount

            legal_actions[4] = self.ai_chips

        return legal_actions

    def ai_action(self) -> Tuple[str, int]:
        observation = self.get_observation()
        self.current_observation = observation
        self.ai.ai_chips = self.ai_chips

        legal_actions = self._get_legal_actions()
        
        action_code, amount = self.ai.decide_action(observation, legal_actions)

        self.ai.memory.add_experience(observation, action_code, 0.0)
        
        action_map = {0: 'fold', 1: 'call', 2: 'raise', 3: 'raise', 4: 'raise'}
        return action_map[action_code], amount

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

    def get_observation(self) -> np.ndarray:
        obs = np.zeros(60, dtype=np.float32)
        for i, card in enumerate(self.ai_hand):
            if i < 2:
                suit_idx = list(Suit).index(card.suit)
                obs[i*4 + suit_idx] = 1.0
                obs[8 + i] = card.rank.value / 14.0
        for i, card in enumerate(self.community_cards):
            if i < 5:
                suit_idx = list(Suit).index(card.suit)
                obs[10 + i*4 + suit_idx] = 1.0
                obs[30 + i] = card.rank.value / 14.0
        
        total_chips = self.ai_chips + self.player_chips + self.pot
        total_chips = max(total_chips, 1)
        obs[35] = np.clip(self.pot / total_chips, 0.0, 1.0)
        obs[36] = np.clip(self.ai_chips / total_chips, 0.0, 1.0)
        obs[37] = np.clip(self.player_chips / total_chips, 0.0, 1.0)
        obs[38] = np.clip(self.current_bet / max(self.pot, 1), 0.0, 1.0)
        obs[39] = np.clip(self.ai_bet / total_chips, 0.0, 1.0)
        obs[40] = np.clip(self.player_bet / total_chips, 0.0, 1.0)
        obs[41] = len(self.community_cards) / 5.0
        obs[42] = 1.0 if len(self.community_cards) == 0 else 0.0
        obs[43] = 1.0 if len(self.community_cards) == 3 else 0.0
        obs[44] = 1.0 if len(self.community_cards) == 4 else 0.0
        obs[45] = 1.0 if len(self.community_cards) == 5 else 0.0
        need_to_call = max(0, self.current_bet - self.ai_bet)
        obs[46] = np.clip(need_to_call / max(self.pot, 1), 0.0, 1.0)
        obs[47] = 1.0 if self.current_bet > 0 else 0.0
        obs[48] = self.ai.win_rate
        obs[49] = len(self.ai.memory.experiences) / 10000.0
        return obs

    def deal_flop(self):
        for _ in range(3):
            self.community_cards.append(self.deck.deal_card())

    def deal_turn(self):
        self.community_cards.append(self.deck.deal_card())

    def deal_river(self):
        self.community_cards.append(self.deck.deal_card())

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

    def determine_winner(self) -> str:
        player_best = self.get_best_hand(self.player_hand)
        ai_best = self.get_best_hand(self.ai_hand)
        if player_best is None or ai_best is None: return "tie"
        if player_best > ai_best: return "player"
        elif ai_best > player_best: return "ai"
        else: return "tie"

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
                elif choice == 1:
                    self.ai_chips += self.pot
                    self.ai.learn_from_hand(1.0)
                    self.ai.update_stats(True)
                    return False
                elif choice == 2:
                    if need_to_call > 0:
                        call_amount = min(need_to_call, self.player_chips)
                        self.player_chips -= call_amount
                        self.player_bet += call_amount
                        self.pot += call_amount
                    action_taken = True
                    break
                elif choice == 3:
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

        ai_action, ai_amount = self.ai_action()
        action_map_str = {0: "폴드", 1: "콜", 2: "팟의 50% 레이즈", 3: "팟의 100% 레이즈", 4: "올인"}
        print(f"🤖 AI의 선택: {action_map_str.get(ai_action, '알 수 없음')} ({ai_amount}칩)")

        if ai_action == 0:
            self.player_chips += self.pot
            self.ai.learn_from_hand(-1.0)
            self.ai.update_stats(False)
            return False
        elif ai_action == 1:
            if ai_amount > 0:
                self.ai_chips -= ai_amount
                self.ai_bet += ai_amount
                self.pot += ai_amount
            return True
        else:
            self.ai_chips -= ai_amount
            self.ai_bet += ai_amount
            self.pot += ai_amount
            self.current_bet = self.ai_bet
            return self.betting_round()

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
            self.ai.learn_from_hand(-1.0)
            self.ai.update_stats(False)
        elif winner == "ai":
            print("\n🤖 AI 승리!")
            self.ai_chips += self.pot
            self.ai.learn_from_hand(1.0)
            self.ai.update_stats(True)
        else:
            print("\n🤝 무승부!")
            self.player_chips += self.pot // 2
            self.ai_chips += self.pot // 2
            self.ai.learn_from_hand(0.1)
            
        
        if player_best:
            print(f"\n👤 플레이어 베스트 핸드: {player_best.cards} ({player_best.rank.name})")
        else:
            print("\n👤 플레이어 베스트 핸드: (핸드 없음)")

        if ai_best:
            print(f"🤖 AI 베스트 핸드: {ai_best.cards} ({ai_best.rank.name})")
        else:
            print("🤖 AI 베스트 핸드: (핸드 없음)")

    def play_game(self):
        print("🃏 실시간 학습 포커 AI(v2) 대전에 오신 것을 환영합니다! 🃏")
        hand_count = 0
        save_interval = 10
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
        self.ai.save_model()
        print(self.ai.get_learning_progress())
