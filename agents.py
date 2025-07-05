import numpy as np
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from NoLimitHoldemGymWrapper import NoLimitHoldemGymWrapper

class BaseAgent(ABC):
    """Base class for poker agents."""
    
    def __init__(self, name: str = "Agent"):
        self.name = name
    
    @abstractmethod
    def get_action(self, observation: np.ndarray, legal_actions: List[int], 
                   info: Dict[str, Any]) -> int:
        """Select an action given the current state."""
        pass


class RandomAgent(BaseAgent):
    """Agent that selects actions uniformly at random from legal actions."""
    
    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int], 
                   info: Dict[str, Any]) -> int:
        """Randomly select from legal actions."""
        if not legal_actions:
            return 1  # Default to CHECK/CALL if no legal actions
        return np.random.choice(legal_actions)


class SimplePotOddsAgent(BaseAgent):
    """
    Simple rule-based agent that makes decisions based on pot odds.
    
    Strategy:
    - Always check when possible (action 1 when current_bet is 0)
    - Calculate pot odds and compare to a threshold
    - Fold if pot odds are unfavorable
    - Call/raise based on hand strength estimate
    """
    
    def __init__(self, name: str = "PotOddsAgent", 
                 fold_threshold: float = 0.3,
                 raise_threshold: float = 0.7):
        super().__init__(name)
        self.fold_threshold = fold_threshold
        self.raise_threshold = raise_threshold
    
    def estimate_hand_strength(self, observation: np.ndarray, info: Dict[str, Any]) -> float:
        """
        Estimate hand strength based on cards (simplified).
        Returns a value between 0 and 1.
        """
        # Extract hand cards from observation (first 104 elements, 2 cards one-hot encoded)
        hand_indices = []
        for i in range(52):
            if observation[i] > 0:
                hand_indices.append(i)
        
        if len(hand_indices) < 2:
            return 0.5  # Default if we can't read cards
        
        # Simple heuristic based on card ranks
        # Cards are indexed 0-51, where rank = index % 13
        ranks = [idx % 13 for idx in hand_indices]
        
        # Pair detection
        if ranks[0] == ranks[1]:
            # Pocket pair - higher ranks are better
            return 0.6 + (ranks[0] / 13) * 0.3
        
        # High cards
        max_rank = max(ranks)
        if max_rank >= 11:  # Ace or King
            return 0.5 + (max_rank - 10) * 0.1
        elif max_rank >= 9:  # Queen or Jack
            return 0.4 + (max_rank - 8) * 0.05
        
        # Default for lower cards
        return 0.3 + (max_rank / 13) * 0.2
    
    def calculate_pot_odds(self, pot_size: int, bet_to_call: int) -> float:
        """Calculate pot odds as a ratio."""
        if bet_to_call == 0:
            return 1.0  # Free to play
        return bet_to_call / (pot_size + bet_to_call)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int], 
                   info: Dict[str, Any]) -> int:
        """Make decision based on pot odds and estimated hand strength."""
        if not legal_actions:
            return 1  # Default to CHECK/CALL
        
        pot_size = info.get('pot_size', 0)
        current_bet = info.get('raw_state', {}).get('current_bet', 0)
        
        # If we can check (no bet to call), always check
        if current_bet == 0 and 1 in legal_actions:
            return 1  # CHECK
        
        # Calculate pot odds
        pot_odds = self.calculate_pot_odds(pot_size, current_bet)
        
        # Estimate our hand strength
        hand_strength = self.estimate_hand_strength(observation, info)
        
        # Decision logic
        if pot_odds > self.fold_threshold and 0 in legal_actions:
            # Pot odds are bad, fold if possible
            return 0  # FOLD
        
        if hand_strength > self.raise_threshold:
            # Strong hand, try to raise
            if 4 in legal_actions:  # ALL_IN available
                return 4
            elif 3 in legal_actions:  # RAISE_POT available
                return 3
            elif 2 in legal_actions:  # RAISE_HALF_POT available
                return 2
        
        # Default to call/check
        if 1 in legal_actions:
            return 1  # CHECK/CALL
        
        # If we can't call, fold
        return 0 if 0 in legal_actions else legal_actions[0]


class TightAgent(BaseAgent):
    """
    A tight agent that only plays premium hands.
    Folds most hands, only plays strong starting hands.
    """
    
    def __init__(self, name: str = "TightAgent"):
        super().__init__(name)
        # Premium starting hands (simplified)
        self.premium_ranks = {12, 11, 10, 9}  # A, K, Q, J
    
    def has_premium_hand(self, observation: np.ndarray) -> bool:
        """Check if we have a premium starting hand."""
        hand_indices = []
        for i in range(52):
            if observation[i] > 0:
                hand_indices.append(i)
        
        if len(hand_indices) < 2:
            return False
        
        ranks = [idx % 13 for idx in hand_indices]
        
        # Pocket pair of 9s or better
        if ranks[0] == ranks[1] and ranks[0] >= 8:
            return True
        
        # Both cards are high cards
        return all(rank in self.premium_ranks for rank in ranks)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int], 
                   info: Dict[str, Any]) -> int:
        """Play only premium hands, fold everything else pre-flop."""
        if not legal_actions:
            return 1
        
        # Check if we're in pre-flop (no community cards)
        community_cards_start = 104  # After 2*52 for hand cards
        has_community_cards = any(observation[community_cards_start:community_cards_start+52] > 0)
        
        if not has_community_cards:  # Pre-flop
            if self.has_premium_hand(observation):
                # Play aggressively with premium hands
                if 3 in legal_actions:  # RAISE_POT
                    return 3
                elif 2 in legal_actions:  # RAISE_HALF_POT
                    return 2
                elif 1 in legal_actions:  # CALL
                    return 1
            else:
                # Fold non-premium hands
                if 0 in legal_actions:
                    return 0
                elif 1 in legal_actions and info.get('raw_state', {}).get('current_bet', 0) == 0:
                    return 1  # Check if free
                else:
                    return 0 if 0 in legal_actions else 1
        
        # Post-flop: play more conservatively
        current_bet = info.get('raw_state', {}).get('current_bet', 0)
        if current_bet == 0 and 1 in legal_actions:
            return 1  # Check when possible
        
        # Only continue with strong hands post-flop
        if self.has_premium_hand(observation):
            if 1 in legal_actions:
                return 1  # Call
        
        return 0 if 0 in legal_actions else 1


# Example usage and testing
if __name__ == "__main__":
    from NoLimitHoldemGymWrapper import NoLimitHoldemGymWrapper
    
    # Create environment
    env = NoLimitHoldemGymWrapper(num_players=2, init_chips=100)
    
    # Create agents
    agents = [
        RandomAgent("Random Player"),
        SimplePotOddsAgent("Pot Odds Player"),
        TightAgent("Tight Player")
    ]
    
    # Test each agent
    for agent in agents:
        print(f"\nTesting {agent.name}...")
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 50:
            info = {
                'current_player': env.current_player,
                'legal_actions': env._get_legal_actions(),
                'raw_state': env.raw_state,
                'pot_size': env.raw_state['pot'],
                'player_chips': env.raw_state['stakes']
            }
            
            action = agent.get_action(obs, info['legal_actions'], info)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 10 == 0:
                print(f"  Step {steps}: Total reward = {total_reward}")
        
        print(f"  Final: {steps} steps, total reward = {total_reward}")