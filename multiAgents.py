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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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

        # print('successorGameState:', successorGameState)
        # print('newPos:', newPos)
        # print('newFood:', newFood)
        # print('newGhostStates:', newGhostStates)
        # print('newScaredTimes:', newScaredTimes)

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabeta(gameState)
        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)

        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))
        # return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        def alphaBetaSearch(gameState, turn, alpha, beta):
            numAgents = gameState.getNumAgents()
            agentIndex = turn % numAgents
            depth = turn // numAgents

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            
            # N???u Agent l?? Ghost
            if agentIndex == 0:
                v = float('-inf')
            else: # L?? Pacman
                v = float('inf')

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex > 0:
                    v = min(v, alphaBetaSearch(successor, turn + 1, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)

                else:
                    v = max(v, alphaBetaSearch(successor, turn + 1, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
            
            return v

        alpha, beta = float('-inf'), float('inf')
        actions = gameState.getLegalActions(0)
        vals = []

        for action in actions:
            v = alphaBetaSearch(gameState.generateSuccessor(0, action), 1, alpha, beta)
            alpha = max(alpha, v)
            vals.append(v)

        for i in range(len(vals)):
            if alpha == vals[i]:
                return actions[i]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        def expectimaxSearch(gameState, turn):
            numAgents = gameState.getNumAgents()
            agentIndex = turn % numAgents
            depth = turn // numAgents

            # N???u tr???ng th??i k???t th??c game
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # N???u kh??ng l?? tr???ng th??i k???t th??c game
            actions = gameState.getLegalActions(agentIndex)

            # T??nh evaluation cho t???ng Successor
            evaluations = []
            for action in actions:
                evaluations += [expectimaxSearch(gameState.generateSuccessor(agentIndex, action), turn + 1)]

            # N???u agent l?? Ghost (Min-Agent)
            if agentIndex > 0:
                return sum(evaluations) * 1.0 / len(evaluations)

            # N???u agent l?? Pacman (Max-Agent)
            return max(evaluations)

        actions = gameState.getLegalActions(0)
        return max(actions, key=lambda x: expectimaxSearch(gameState.generateSuccessor(0, x), 1))


# def betterEvaluationFunction(currentGameState):
#     """
#       Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
#       evaluation function (question 5).
#       DESCRIPTION: <write something here so we know what you did>
#     """
#     "*** YOUR CODE HERE ***"
#     # util.raiseNotDefined()
#     newPos = currentGameState.getPacmanPosition()
#     newFood = currentGameState.getFood()
#     newGhostStates = currentGameState.getGhostStates()
#     newCapsules = currentGameState.getCapsules()
#     newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

#     closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
#     if newCapsules:
#         closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
#     else:
#         closestCapsule = 0

#     if closestCapsule:
#         closest_capsule = -3 / closestCapsule
#     else:
#         closest_capsule = 100

#     if closestGhost:
#         ghost_distance = -2 / closestGhost
#     else:
#         ghost_distance = -500

#     foodList = newFood.asList()
#     if foodList:
#         closestFood = min([manhattanDistance(newPos, food) for food in foodList])
#     else:
#         closestFood = 0

#     return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule

#################################
#### H??m l?????ng gi?? t??? c??i ?????t ###
#################################
def myBetterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: 6 Features to compute the linear combination (Estimated Value):
        * Current Game Score
        * Closest Food Distance
        * Inverse of Closest Ghost Distance
        * Number of Foods
        * Number of Capsules 
    """
    # Useful Information
    pacmanPosition = currentGameState.getPacmanPosition()               
    foodPositionList = currentGameState.getFood().asList()
    ghostPositions = currentGameState.getGhostPositions()
    ghostStates = currentGameState.getGhostStates()
    scaredGhostTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    foodLeft = len(foodPositionList)
    capsuleLeft = len(currentGameState.getCapsules())
    score = currentGameState.getScore()
    
    scaredGhosts = list()
    enemyGhosts = list()
    scaredGhostPositions = list()
    enemyGhostPositions = list()
    closestEnemyGhost = float('+Inf')
    closestScaredGhost = float('+Inf')
    closestFood = float('+Inf')

    foodDistances = [manhattanDistance(pacmanPosition, foodPosition) for foodPosition in foodPositionList]

    # N???u tr??n b???n ????? v???n c??n ch???m th???c ??n 
    # Score tr??? ??i kho???ng c??ch ???? (V???i tr???ng s??? l?? 1)
    if len(foodDistances) != 0:
        closestFood = min(foodDistances)
        score = score - 1.0 * closestFood
        
    # Duy???t qua t???t c??? c??c Ghosts hi???n c?? tr??n b???n ?????
    # Tr??ch xa 2 m???ng Ghost hi???n c??: 
    #    * Enemy Ghosts
    #    * Scared Ghosts
    for ghost in ghostStates:
        if ghost.scaredTimer != 0:
            enemyGhosts.append(ghost)
        else:
            scaredGhosts.append(ghost)
            
    # Duy???t qua t???t c??? c??c Enemy Ghosts
    # L???y v??? tr?? tr??n b???n ????? v???i t???t c??? Enemy Ghosts
    #   * Enemy Ghost Positions
    for enemyGhost in enemyGhosts:
        enemyGhostPositions.append(enemyGhost.getPosition())

    # N???u tr??n b???n ????? hi???n t???i c?? b???t c??? Enemy Ghost n??o
    # T??m kho???ng c??ch c???a Enemy Ghost g???n nh???t b???ng Manhattan Distance
    # Score tr??? ??i "ngh???ch ?????o" kho???ng c??ch ???? (V???i tr???ng s??? l?? 2)
    if len(enemyGhostPositions) != 0:
        enemyGhostDistances = [manhattanDistance(pacmanPosition, enemyGhostPosition) for enemyGhostPosition in enemyGhostPositions]
        closestEnemyGhost = min(enemyGhostDistances)

        # Kho???ng c??ch v???i Enemy Ghost c??ng l???n th?? Score c??ng l???n
        score = score - 2.0 * (1 / closestEnemyGhost)

    # Duy???t qua t???t c??? c??c Scared Ghosts
    # L???y v??? tr?? tr??n b???n ????? v???i t???t c??? Scared Ghosts
    #   * Scared Ghost Positions
    for scaredGhost in scaredGhosts:
        scaredGhostPositions.append(scaredGhost.getPosition())

    # N???u tr??n b???n ????? hi???n t???i c?? b???t c??? Scared Ghost n??o
    # T??m kho???ng c??ch c???a Scared Ghost g???n nh???t b???ng Manhattan Distance
    # Score tr??? ??i "ngh???ch ?????o" kho???ng c??ch ???? (V???i tr???ng s??? l?? 2)
    if len(scaredGhostPositions) != 0:
        scaredGhostDistances = [manhattanDistance(pacmanPosition, scaredGhostPosition) for scaredGhostPosition in scaredGhostPositions]
        closestscaredGhost = min(scaredGhostDistances)

        # Kho???ng c??ch v???i Scared Ghost c??ng nh??? th?? Score c??ng l???n
        score = score - 3.0 * closestscaredGhost

    # ?????i v???i s??? l?????ng Foods c??n l???i (C??ng nhi???u Foods th?? Score c??ng nh???)
    score = score - 4.0 * foodLeft
    # ?????i v???i s??? l?????ng Casules c??n l???i (C??ng nhi???u Casules th?? Score c??ng nh???)
    score = score - 20.0 * capsuleLeft

    return score
    # util.raiseNotDefined()
    

# Abbreviation
better = myBetterEvaluationFunction
