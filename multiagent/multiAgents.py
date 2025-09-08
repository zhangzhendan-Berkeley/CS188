# multiAgents.py
# --------------
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
from scipy.cluster.hierarchy import average

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, foodPos) for foodPos in foodList])
            foodScore = 1.0 / (minFoodDistance + 1)
        else:
            foodScore = 0
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        ghostScore = 0
        for dist, scaredTime in zip(ghostDistances, newScaredTimes):
            if dist <= 1 and scaredTime == 0:
                ghostScore -= 1000
            if scaredTime > 0 and dist > 0:
                ghostScore += 10 / dist
        foodLeftScore = -len(foodList)
        capsules = currentGameState.getCapsules()
        capsuleScore = 0
        if capsules:
            minCapsuleDistance = min([manhattanDistance(newPos, cap) for cap in capsules])
            capsuleScore = 1.0 / (minCapsuleDistance + 1)
        score = (successorGameState.getScore()
                 + foodScore * 10
                 + ghostScore
                 + foodLeftScore * 5
                 + capsuleScore * 2)
        if action == Directions.STOP:
            score -= 2
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()

            if agentIndex == 0:
                bestScore = float('-inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(1, depth, successor)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                if depth == 0:
                    return bestAction
                else:
                    return bestScore
            else:
                bestScore = float('inf')
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgent, nextDepth, successor)
                    bestScore = min(bestScore, score)
                return bestScore

        return minimax(0, 0, gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        def minimax(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()

            if agentIndex == 0:  # Pacman (Max)
                bestScore = float('-inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(1, depth, successor, alpha, beta)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                    alpha = max(alpha, bestScore)
                    if beta < alpha:  # 剪枝
                        break
                return bestAction if depth == 0 else bestScore

            else:  # Ghost (Min)
                bestScore = float('inf')
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgent, nextDepth, successor, alpha, beta)
                    bestScore = min(bestScore, score)
                    beta = min(beta, bestScore)
                    if beta < alpha:  # 剪枝
                        break
                return bestScore

        return minimax(0, 0, gameState, float('-inf'), float('inf'))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def Expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()

            if agentIndex == 0:  # Pacman (Max)
                bestScore = float('-inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = Expectimax(1, depth, successor)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                return bestAction if depth == 0 else bestScore

            else:  # Ghost (Min)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth
                total = 0
                actions = gameState.getLegalActions(agentIndex)
                if not actions:
                    return self.evaluationFunction(gameState)
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = Expectimax(nextAgent, nextDepth, successor)
                    total += score
                return total / len(actions)

        return Expectimax(0, 0, gameState)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    This evaluation function considers multiple factors to guide Pacman’s behavior:

    1. Food proximity: Pacman prefers closer food, calculated as the reciprocal of the Manhattan distance to the nearest food.
    2. Ghost threat: If a ghost is too close and not scared, a large penalty is applied to avoid immediate danger.
       Conversely, if a ghost is scared, Pacman is rewarded for approaching it, proportional to the inverse distance.
    3. Remaining food: Fewer remaining food items increase the score, encouraging Pacman to eat all the dots.
    4. Capsules: Closer capsules increase the score, guiding Pacman to eat power pellets strategically.
    5. Base game score: The original game score is included to maintain the standard Pacman scoring.

    Weights are applied to balance the importance of each factor, ensuring Pacman efficiently collects food, avoids ghosts, and utilizes capsules when advantageous.

    """
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    foodList = Food.asList()
    if foodList:
        minFoodDistance = min([manhattanDistance(Pos, foodPos) for foodPos in foodList])
        foodScore = 1.0 / (minFoodDistance + 1)
    else:
        foodScore = 0
    ghostDistances = [manhattanDistance(Pos, ghost.getPosition()) for ghost in GhostStates]
    ghostScore = 0
    for dist, scaredTime in zip(ghostDistances, ScaredTimes):
        if dist <= 1 and scaredTime == 0:
            ghostScore -= 1000
        if scaredTime > 0 and dist > 0:
            ghostScore += 100 / dist
    foodLeftScore = -len(foodList)
    capsules = currentGameState.getCapsules()
    capsuleScore = 0
    if capsules:
        minCapsuleDistance = min([manhattanDistance(Pos, cap) for cap in capsules])
        capsuleScore = 1.0 / (minCapsuleDistance + 1)
    score = (currentGameState.getScore()
             + foodScore * 10
             + ghostScore * 3
             + foodLeftScore * 5
             + capsuleScore * 10)
    return score

# Abbreviation
better = betterEvaluationFunction
