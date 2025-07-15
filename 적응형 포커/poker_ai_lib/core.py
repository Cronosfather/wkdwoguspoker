from enum import Enum
from itertools import combinations
import random
from typing import List, Tuple

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
        
        if unique_ranks[-1] - unique_ranks[0] == 4:
            return True
        
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
