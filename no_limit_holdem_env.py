import numpy as np
import gym
from gym import spaces
import rlcard
from rlcard.envs.nolimitholdem import NolimitholdemEnv
from rlcard.games.nolimitholdem.round import Action
from rlcard.games.nolimitholdem.game import Stage
from typing import Dict, List, Tuple, Optional, Any


class NoLimitHoldemGymWrapper(gym.Env):
    def __init__(
        self,
        num_players: int = 2,
        init_chips: int = 100,
        small_blind: int = 1,
        big_blind: int = 2,
    ):
        super().__init__()
        self.num_players = num_players
        self.init_chips = init_chips
        self.small_blind = small_blind
        self.big_blind = big_blind

        config = {
            "game_num_players": num_players,
            "chips_for_each": init_chips,
            "small_blind": small_blind,
            "big_blind": big_blind,
        }
        self.env: NolimitholdemEnv = rlcard.make("no-limit-holdem", config=config)

        self.action_space = spaces.Discrete(5)  # FOLD, CHECK/CALL, ½-pot, pot, all-in

        obs_dim = 2 * 52 + 5 * 52 + 1 + 1 + 1 + num_players + 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.current_player = 0
        self.done = False
        self.state: Optional[Dict[str, Any]] = None
        self.raw_state: Optional[Dict[str, Any]] = None

    def reset(self) -> np.ndarray:
        self.state, self.current_player = self.env.reset()
        self.done = False
        self.raw_state = self._get_raw_state()
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise ValueError("Episode done; call reset()")
        rlcard_action = self._convert_action(action)
        self.state, self.current_player = self.env.step(rlcard_action)
        reward = 0.0
        if self.env.is_over():
            self.done = True
            rewards = self.env.get_payoffs()
            reward = rewards[self.current_player]
        self.raw_state = self._get_raw_state()
        observation = self._get_observation()
        info = {
            "current_player": self.current_player,
            "legal_actions": self._get_legal_actions(),
            "raw_state": self.raw_state,
            "pot_size": self.raw_state["pot"],
            "player_chips": self.raw_state["stakes"],
        }
        return observation, reward, self.done, info

    @staticmethod
    def _convert_action(action: int) -> Action:
        if action == 0:
            return Action.FOLD
        if action == 1:
            return Action.CHECK_CALL
        if action == 2:
            return Action.RAISE_HALF_POT
        if action == 3:
            return Action.RAISE_POT
        if action == 4:
            return Action.ALL_IN
        raise ValueError("Illegal action id")

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        idx = 0

        hand_cards = self.state.get("hand", [])
        for card in hand_cards:
            if card < 52:
                obs[idx + card] = 1.0
        idx += 2 * 52

        public_cards = self.state.get("public_cards", [])
        for card in public_cards:
            if card < 52:
                obs[idx + card] = 1.0
        idx += 5 * 52

        obs[idx] = self.raw_state["stakes"][self.current_player] / self.init_chips
        idx += 1
        obs[idx] = self.raw_state["pot"] / (self.init_chips * self.num_players)
        idx += 1
        obs[idx] = self.raw_state["current_bet"] / self.init_chips
        idx += 1

        obs[idx + self.current_player] = 1.0
        idx += self.num_players

        round_idx = self.raw_state["round"]
        if round_idx < 4:
            obs[idx + round_idx] = 1.0

        return obs

    def _get_raw_state(self) -> Dict[str, Any]:
        raised = self.state.get("all_chips", [0] * self.num_players)
        diff = max(raised) - raised[self.current_player]
        return {
            "stakes": self.state.get("stakes", [self.init_chips] * self.num_players),
            "pot": self.state.get("pot", 0),
            "current_bet": diff,
            "community_cards": self.state.get("public_cards", []),
            "hand": self.state.get("hand", []),
            "round": self.state.get("stage", Stage.PREFLOP).value,
        }

    def _get_legal_actions(self) -> List[int]:
        action_ids = []
        for a in self.state.get("legal_actions", []):
            mapping = {
                Action.FOLD: 0,
                Action.CHECK_CALL: 1,
                Action.RAISE_HALF_POT: 2,
                Action.RAISE_POT: 3,
                Action.ALL_IN: 4,
            }
            if a in mapping:
                action_ids.append(mapping[a])
        return action_ids

    def render(self, mode: str = "human") -> Optional[str]:
        if mode != "human":
            return None
        out = []
        out.append("\n" + "=" * 50)
        out.append(f"No-Limit Hold'em — Player {self.current_player}'s turn")
        out.append("=" * 50)
        out.append(f"Community: {self.raw_state['community_cards']}")
        out.append(f"Pot: ${self.raw_state['pot']}")
        out.append(f"To call: ${self.raw_state['current_bet']}")
        for i, stake in enumerate(self.raw_state["stakes"]):
            mark = " ←" if i == self.current_player else ""
            out.append(f"Player {i}: ${stake}{mark}")
        out.append(f"Your hand: {self.raw_state['hand']}")
        out.append(f"Legal actions: {self._get_legal_actions()}")
        return "\n".join(out)

    def close(self):
        pass


if __name__ == "__main__":
    env = NoLimitHoldemGymWrapper(num_players=2, init_chips=100)
    print("Testing wrapper…")
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    obs = env.reset()
    print("Initial obs shape:", obs.shape)
    done, step, total = False, 0, 0
    while not done and step < 100:
        legal = env._get_legal_actions() or [1]
        action = np.random.choice(legal)
        obs, reward, done, info = env.step(action)
        total += reward
        print(f"\nStep {step}")
        print(env.render())
        print("Action:", action, "Reward:", reward)
        step += 1
    print("\nFinished after", step, "steps  total reward:", total)
