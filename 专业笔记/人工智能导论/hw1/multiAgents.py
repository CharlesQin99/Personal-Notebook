from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
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
    Your minimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        we assume ghosts act in turn after the pacman takes an action
        so your minimax tree will have multiple min layers (one for each ghost)
        for every max layer

        gameState.generateChild(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state

        self.evaluationFunction(state)
        Returns pacman SCORE in current state (useful to evaluate leaf nodes)

        self.depth
        limits your minimax tree depth (note that depth increases one means
        the pacman and all ghosts has already decide their actions)
        """
        maxVal = -float('inf')
        bestAction = None
        agentsNum = gameState.getNumAgents()
        for action in gameState.getLegalActions(0):
            value = self._getMin(gameState.generateChild(0, action), self.depth, agentsNum-1)
            if value is not None and value > maxVal:
                maxVal = value
                bestAction = action
        return bestAction

    def _getMax(self, gameState, currentDepth, agentIndex):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(agentIndex)
        if currentDepth == self.depth or len(legalActions) == 0:
            return self.evaluationFunction(gameState)
        maxVal = -float('inf')
        for action in legalActions:
            value = self._getMin(gameState.generateChild(agentIndex, action), currentDepth, agentIndex)
            if value is not None and value > maxVal:
                maxVal = value
        return maxVal

    def _getMin(self, gameState, currentDepth, agentIndex):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(agentIndex)
        if currentDepth == self.depth or len(legalActions) == 0:
            return self.evaluationFunction(gameState)
        minVal = float('inf')
        for action in legalActions:
            if agentIndex == gameState.getNumAgents() - 1:
                value = self._getMax(gameState.generateChild(agentIndex, action), currentDepth + 1, agentIndex)
            else:
                value = self._getMin(gameState.generateChild(agentIndex, action), currentDepth, agentIndex + 1)
            if value is not None and value < minVal:
                minVal = value
        return minVal


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
