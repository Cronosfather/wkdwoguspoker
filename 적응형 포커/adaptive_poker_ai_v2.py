from poker_ai_lib.game import LearningPokerGame

if __name__ == "__main__":
    model_path = "learning_poker_ai"
    
    print("🚀 실시간 학습 포커 AI를 시작합니다!")
    print("=" * 50)
    
    game = LearningPokerGame(model_path)
    game.play_game()
    
    print("\n🎯 게임을 종료합니다. 다음에 다시 시작하면 AI가 학습한 내용을 기억합니다!")
