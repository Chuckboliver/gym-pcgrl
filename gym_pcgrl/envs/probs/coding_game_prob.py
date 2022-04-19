from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_tile_locations, calc_num_regions, calc_certain_tile, run_dijkstra, get_range_reward
import numpy as np
from PIL import Image
import os

"""
Generate a fully connected level for coding game

"""
class CodingGameProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 14
        self._height = 14
        self._prob = {"empty": 0.5, "solid":0.35, "player":0.05, "key":0.05, "door":0.05}
        self._border_tile = "solid"

        self._target_path = 8
        self._random_probs = True

        self._rewards = {
            "player": 3,
            "key": 3,
            "door": 3,
            "regions": 5,
            "path-length": 1
        }


    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "solid", "player", "key", "door"]


    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intialization, the names are "empty", "solid"
        target_path (int): the current path length that the episode turn when it reaches
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._target_path = kwargs.get('target_path', self._target_path)
        self._random_probs = kwargs.get('random_probs', self._random_probs)


    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "regions": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "key": calc_certain_tile(map_locations, ["key"]),
            "door": calc_certain_tile(map_locations, ["door"]),
            "regions": calc_num_regions(map, map_locations, ["empty", "player", "key"]),
            "path-length": 0
        }
        if map_stats["player"] == 1 and map_stats["regions"] == 1:
            p_x,p_y = map_locations["player"][0]
            if map_stats["key"] == 1 and map_stats["door"] == 1:
                k_x,k_y = map_locations["key"][0]
                d_x,d_y = map_locations["door"][0]
                dijkstra,_ = run_dijkstra(p_x, p_y, map, ["empty", "key", "player"])
                map_stats["path-length"] += dijkstra[k_y][k_x]
                dijkstra,_ = run_dijkstra(k_x, k_y, map, ["empty", "player", "key", "door"])
                map_stats["path-length"] += dijkstra[d_y][d_x]

        return map_stats

    
    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "key": get_range_reward(new_stats["key"], old_stats["key"], 1, 1),
            "door": get_range_reward(new_stats["door"], old_stats["door"], 1, 1),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["key"] * self._rewards["key"] +\
            rewards["door"] * self._rewards["door"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["path-length"] * self._rewards["path-length"]

    
    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, old_stats):
        return new_stats["path-length"] >= self._target_path


    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "key": new_stats["key"],
            "door": new_stats["door"],
            "regions": new_stats["regions"],
            "path-length": new_stats["path-length"]
        }


    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/zelda/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/zelda/solid.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/zelda/player.png").convert('RGBA'),
                "key": Image.open(os.path.dirname(__file__) + "/zelda/key.png").convert('RGBA'),
                "door": Image.open(os.path.dirname(__file__) + "/zelda/door.png").convert('RGBA'),
            }
        return super().render(map)