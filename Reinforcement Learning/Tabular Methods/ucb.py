"""
This file creates an UCB Agent.
Note: The helper functions and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Optional, Union

# Third Party
import numpy as np


# Private


# -------------------------------------------------------------------------------------------------------------------- #
class UCB:
    
    """
    Initialise UCB agent. 
    - This agent returns an action between 0 and 'number_of_arms'.
    - This agent uses uncertainty in the action-value estimates for balancing exploration and exploitation.
    """
    def __init__(self, name: str, number_of_arms: int, bonus_multiplier: float):
        self._number_of_arms = number_of_arms
        self._bonus_multiplier = bonus_multiplier
        self.name = name
        self.reset()

    """
    Execute UCB agent's next action and update UCB's action-state values.
    """
    def step(self, previous_action: Optional[int], reward: Union[float, int]) -> int:
        # Execute UCB
        if previous_action != None:
        # Update action count for previous action
            self.N_t[previous_action] += 1
        # Use iterative form of Q_t(a)
        self.Q_t[previous_action] += (1 / self.N_t[previous_action]) * (reward - self.Q_t[previous_action])
        # All actions must be selected at least once before UCB is applied
        if np.any(self.N_t == 0):
        # Select non-explored action
            action = np.random.choice(np.where(self.N_t== 0)[0])
        else:
        # Calculate expected reward values
            reward_values = self.Q_t + self._bonus_multiplier * np.sqrt(np.log(self.t) / self.N_t)
        # A_t(a) is the 'action' chosen at time step 't'
        action = np.random.choice(np.where(reward_values == np.max(reward_values))[0])
        # Update time step counter
        self.t += 1
        return action

    """
    Reset UCB agent.
    """
    def reset(self):
        # Q_t(a) is the estimated value of action ‘a’ at time step ‘t’
        self.Q_t = np.zeros(self._number_of_arms)
        # N_t(a) is the number of times that action ‘a’ has been selected, prior to time ‘t’
        self.N_t = np.zeros(self._number_of_arms)
        # Set time step counter
        self.t = 1