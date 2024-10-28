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
from pacman import GameState
import util

from game import Agent



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
    Votre agent minimax (question 2)
    """

    def getAction(self, state: GameState):
        """
        Retourne l'action minimax à partir de l'état actuel du jeu en utilisant self.depth
        et self.evaluationFunction.
        """
        def minimax(agentIndex, depth, gameState):
            # Conditions de fin : victoire, défaite ou profondeur maximale atteinte
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Si l'agent est Pacman (Maximisation)
            if agentIndex == 0:
                return max(minimax(1, depth, gameState.getNextState(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))

            # Si l'agent est un fantôme (Minimisation)
            else:
                nextAgent = agentIndex + 1  # prochain fantôme
                if nextAgent == gameState.getNumAgents():  # si dernier agent, reset to Pacman
                    nextAgent, depth = 0, depth + 1  # Pacman et +1 profondeur
                return min(minimax(nextAgent, depth, gameState.getNextState(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))

        # Retourne l'action avec la valeur minimax maximale pour Pacman
        legalMoves = state.getLegalActions(0)
        bestAction = max(legalMoves, key=lambda action: minimax(1, 0, state.getNextState(0, action)))
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Votre agent minimax avec élagage alpha-bêta (question 3)
    """

    def getAction(self, state: GameState):
        """
        Retourne l'action minimax avec élagage alpha-bêta en utilisant self.depth et self.evaluationFunction.
        """
        def alpha_beta(agentIndex, depth, gameState, alpha, beta):
            # Conditions de fin : victoire, défaite, ou profondeur maximale atteinte
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman joue (Maximisation)
            if agentIndex == 0:
                maxEval = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    evalValue = alpha_beta(1, depth, gameState.getNextState(agentIndex, action), alpha, beta)
                    maxEval = max(maxEval, evalValue)
                    alpha = max(alpha, evalValue)
                    if beta <= alpha:  # Élagage beta
                        break
                return maxEval

            # Fantômes jouent (Minimisation)
            else:
                minEval = float('inf')
                nextAgent = agentIndex + 1  # prochain fantôme
                if nextAgent == gameState.getNumAgents():  # si dernier agent, revient à Pacman
                    nextAgent, depth = 0, depth + 1
                for action in gameState.getLegalActions(agentIndex):
                    evalValue = alpha_beta(nextAgent, depth, gameState.getNextState(agentIndex, action), alpha, beta)
                    minEval = min(minEval, evalValue)
                    beta = min(beta, evalValue)
                    if beta <= alpha:  # Élagage alpha
                        break
                return minEval

        # Obtenir l'action avec la valeur minimax optimisée pour Pacman
        legalMoves = state.getLegalActions(0)
        bestAction = None
        alpha, beta = float('-inf'), float('inf')
        maxEval = float('-inf')

        for action in legalMoves:
            evalValue = alpha_beta(1, 0, state.getNextState(0, action), alpha, beta)
            if evalValue > maxEval:
                maxEval = evalValue
                bestAction = action
            alpha = max(alpha, evalValue)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Votre agent Expectimax (question 4)
    """

    def getAction(self, state: GameState):
        """
        Retourne l'action expectimax en utilisant self.depth et self.evaluationFunction
        """

        def expectimax(agentIndex, depth, gameState):
            # Conditions de fin : victoire, défaite, ou profondeur maximale atteinte
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman joue (Maximisation)
            if agentIndex == 0:
                maxEval = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    evalValue = expectimax(1, depth, gameState.getNextState(agentIndex, action))
                    maxEval = max(maxEval, evalValue)
                return maxEval

            # Les fantômes jouent (Valeur moyenne des résultats possibles)
            else:
                totalValue = 0
                actions = gameState.getLegalActions(agentIndex)
                probability = 1 / len(actions)  # probabilité uniforme
                nextAgent = agentIndex + 1  # prochain agent
                if nextAgent == gameState.getNumAgents():  # si dernier agent, revient à Pacman
                    nextAgent, depth = 0, depth + 1

                for action in actions:
                    evalValue = expectimax(nextAgent, depth, gameState.getNextState(agentIndex, action))
                    totalValue += evalValue * probability

                return totalValue

        # Obtenir l'action avec la valeur maximisée pour Pacman
        legalMoves = state.getLegalActions(0)
        bestAction = None
        maxEval = float('-inf')

        for action in legalMoves:
            evalValue = expectimax(1, 0, state.getNextState(0, action))
            if evalValue > maxEval:
                maxEval = evalValue
                bestAction = action

        return bestAction
    
def betterEvaluationFunction(state: GameState):
    """
    Évaluation améliorée pour Pacman.
    
    Prend en compte plusieurs facteurs de l'état du jeu :
    - Distance de la nourriture la plus proche
    - Distance aux fantômes (et leur état "effrayé")
    - Présence de capsules de puissance
    - Score de base de l'état actuel
    """
    # Score de base de l'état actuel
    score = state.getScore()
    
    # Position de Pacman et de la nourriture
    pacmanPos = state.getPacmanPosition()
    foodList = state.getFood().asList()
    
    # Distance à la nourriture la plus proche
    if foodList:
        closestFoodDist = min(util.manhattanDistance(pacmanPos, foodPos) for foodPos in foodList)
        foodScore = 1 / (closestFoodDist + 1)  # Plus la nourriture est proche, meilleur est le score
    else:
        foodScore = 0  # Plus de nourriture à manger
    
    # Distance aux capsules
    capsuleList = state.getCapsules()
    if capsuleList:
        closestCapsuleDist = min(util.manhattanDistance(pacmanPos, capsulePos) for capsulePos in capsuleList)
        capsuleScore = 1 / (closestCapsuleDist + 1)
    else:
        capsuleScore = 0

    # Distance aux fantômes et état "effrayé"
    ghostStates = state.getGhostStates()
    ghostScore = 0
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDist = util.manhattanDistance(pacmanPos, ghostPos)
        
        if ghost.scaredTimer > 0:  # Fantôme effrayé
            ghostScore += 10 / (ghostDist + 1)  # Encourage Pacman à se rapprocher des fantômes effrayés
        else:  # Fantôme normal
            if ghostDist > 0:
                ghostScore -= 10 / ghostDist  # Pacman doit éviter les fantômes proches

    # Pondération des scores
    totalScore = score + foodScore * 10 + capsuleScore * 5 + ghostScore

    return totalScore


# Abbreviation
better = betterEvaluationFunction
