import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    # Expected Sarsa(Failed)
    # def select_action(self, state, policy):
    #     """ Given the state, select an action.
    #
    #     Params
    #     ======
    #     - state: the current state of the environment
    #
    #     Returns
    #     =======
    #     - action: an integer, compatible with the task's action space
    #     """
    #
    #     return np.random.choice(np.arange(self.nA), p=policy)
    #
    # def step(self, state, action, reward, next_state, done, i_episode):
    #     """ Update the agent's knowledge, using the most recently sampled tuple.
    #
    #     Params
    #     ======
    #     - state: the previous state of the environment
    #     - action: the agent's previous choice of action
    #     - reward: last reward received
    #     - next_state: the current state of the environment
    #     - done: whether the episode is complete (True or False)
    #     """
    #     alpha = .01
    #     gamma = 1.0
    #     policy_s = self.epsilon_greedy_probs(next_state, i_episode, 0.005)
    #     self.Q[state][action] += alpha * (reward + (gamma * np.dot(self.Q[next_state], policy_s)) - self.Q[state][action])

    # Sarsamax algorithm
    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        policy_s = self.epsilon_greedy_probs(state, i_episode)
        return np.random.choice(np.arange(self.nA), p=policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        alpha = .015
        gamma = .95
        self.Q[state][action] += alpha * (reward + (gamma * np.max(self.Q[next_state])) - self.Q[state][action])

    def epsilon_greedy_probs(self, state, i_episode, eps=None):
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA)
        return policy_s
