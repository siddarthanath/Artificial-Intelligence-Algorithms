"""
This file creates a Reinforce Agent.
Note: The helper functions and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Optional

# Third Party
import numpy as np


# Private


# -------------------------------------------------------------------------------------------------------------------- #

class REINFORCE:
     
    """
    Initialise Reinforce agent. 
    This agent uses preference policy and takes a probabilistic approach.
    """
    def __init__(self, name, number_of_arms: str, step_size=0.1, baseline=False):
        self.name = name
        self._number_of_arms = number_of_arms
        self._step_size = step_size
        self._baseline = baseline
        self.average_reward = 0
        self.reset()

    """
    Execute Reinforce agent's next action and update Reinforce's preference policy.
    """
    def step(self, previous_action: Optional[int], reward: float) -> int:
        # If action gets executed
        if previous_action is not None:
            # Indicator for which action to update
            ind = np.zeros_like(self.p)
            ind[previous_action] = 1
            # If baseline is used
            if self._baseline:
                # Update average reward
                self.average_reward += self._step_size * (reward - self.average_reward)
                # Update p(a)
                self.p += 2 * self._step_size * (reward - self.average_reward) * (ind - self.pi) / self.p
            else:
                # Update p(a)
                self.p += 2 * self._step_size * reward  * (ind - self.pi) / self.p
        # Update π_t(a)
        self.pi = self.pi_action_probas(p=self.p)
        # A_t(a) is the 'action' chosen at time step 't'
        action = np.random.choice(self._number_of_arms, p=self.pi)
        # Update time step counter
        self.t += 1
        return action

    """
    Reset Reinforce agent.
    """
    def reset(self):
        # p(a) is the action preference of ‘a’
        self.p = np.ones(self._number_of_arms)
        # π_t(a) is the probability of taking action 'a' at time 't'
        self.pi = self.pi_action_probas(p=self.p)
        # Initialise average reward
        self.average_reward = 0
        # Set time step counter
        self.t = 1

    """
    Helper function to calculate generic form of policy i.e. softmax approach.
    """
    def pi_action_probas(self, p: float) -> float:
        # π_t(a) is the probability of taking action 'a' at time 't'
        return (p ** 2) / (np.sum(p ** 2))