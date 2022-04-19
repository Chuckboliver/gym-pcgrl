from gym_pcgrl.envs.probs.problem import Problem

"""
Generate a fully connected level for coding game similar to CodeCombat
"""
class CodingGameProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super.__init__()
