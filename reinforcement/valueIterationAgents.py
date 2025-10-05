# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        states = self.mdp.getStates()
        for _ in range(0,self.iterations):
            newValues = util.Counter()
            for state in states:
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue
                actions = self.mdp.getPossibleActions(state)
                V = float('-inf')
                for action in actions:
                    Q = self.getQValue(state, action)
                    if Q > V:
                        V = Q
                newValues[state] = V
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        Q = 0
        StatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in StatesAndProbs:
            reward = self.mdp.getReward(state,action,nextState)
            value = self.getValue(nextState)
            Q += (reward + value * self.discount) * prob
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        best_action = None
        maxQ = float('-inf')
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        for action in actions:
            Q = self.getQValue(state, action)
            if Q > maxQ:
                maxQ = Q
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        predecessors = {}
        for s in states:
            predecessors[s] = set()

        for s in states:
            if self.mdp.isTerminal(s):
                continue
            for action in self.mdp.getPossibleActions(s):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                    if prob > 0:
                        predecessors[nextState].add(s)

        pq = util.PriorityQueue()
        for s in states:
            if self.mdp.isTerminal(s):
                continue
            actions = self.mdp.getPossibleActions(s)
            maxQ = max([self.getQValue(s, a) for a in actions])
            diff = abs(self.values[s] - maxQ)
            pq.update(s, -diff)

        for _ in range(self.iterations):
            if pq.isEmpty():
                break

            s = pq.pop()
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                self.values[s] = max([self.getQValue(s, a) for a in actions])

            for p in predecessors[s]:
                if self.mdp.isTerminal(p):
                    continue
                actions = self.mdp.getPossibleActions(p)
                maxQ = max([self.getQValue(p, a) for a in actions])
                diff = abs(self.values[p] - maxQ)
                if diff > self.theta:
                    pq.update(p, -diff)

