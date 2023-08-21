"""
This file creates an Epsilon Greedy Agent.
Note: The helper functions and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Optional, Union

# Third Party
import numpy as np


# Private


# -------------------------------------------------------------------------------------------------------------------- #

class EpsilonGreedy:

    """
    Initialise epsilon-greedy agent.
    - This agent returns an action between 0 and 'number_of_arms'.
    - It does so with probability `(1-epsilon)` it chooses the action with the highest estimated value, while
    with probability `epsilon`, it samples an action uniformly at random.
    """
    def __init__(self, name: str, number_of_arms: int, epsilon=Union[0.1, callable]):
        self.name = name
        self._number_of_arms = number_of_arms
        self._epsilon = epsilon
        self.reset()

    """
    Execute Epsilon-Greedy agent's next action and update Epsilon Greedy's action-state values.
    """
    def step(self, previous_action: Optional[int], reward: float) -> int:
        # Execute Epsilon-Greedy
        if previous_action != None:
            # Update action count for previous action
            self.N_t[previous_action] += 1
            # Use iterative form of Q_t(a)
            self.Q_t[previous_action] += (reward - self.Q_t[previous_action]) / self.N_t[previous_action]
        # Check if epsilon is scalar or callable
        new_epsilon = self._epsilon if np.isscalar(self._epsilon) else self._epsilon(self.t)
        # A_t(a) is the 'action' chosen at time step 't'
        action = np.random.choice(np.where(self.Q_t == np.max(self.Q_t))[0]) if np.random.uniform() < 1 - new_epsilon else np.random.randint(0, self.Q_t.shape[0])
        # Update time step counter
        self.t += 1
        return action

    """
    Reset Epsilon Greedy agent.
    """
    def reset(self):
        # Q_t(a) is the estimated value of action ‘a’ at time step ‘t’
        self.Q_t = np.zeros(self._number_of_arms)
        # N_t(a) is the number of times that action ‘a’ has been selected, prior to time ‘t’
        self.N_t = np.zeros(self._number_of_arms)
        # Set time step counter
        self.t = 1