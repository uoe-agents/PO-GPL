import random
import numpy as np
from coopnavigation.navigation.agent import Agent
from enum import IntEnum

class Action(IntEnum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

class HeuristicAgent(Agent):
    name = "Heuristic Agent"

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        return np.rint(coords.mean(axis=0))

    def _furthest_player(self, players, pos):
        coords = [player.position for player in players]
        if len(coords) != 0:
            dists = [(loc[0]-pos[0])**2 + (loc[1]-pos[1])**2 for loc in coords]
            max_dists = max(dists)
            idx_max = np.random.choice([idx for idx, dist in enumerate(dists) if max_dists == dist], 1)[0]

            return coords[idx_max]

        return None

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        if r < y and Action.NORTH in allowed:
            return Action.NORTH
        elif r > y and Action.SOUTH in allowed:
            return Action.SOUTH
        elif c > x and Action.EAST in allowed:
            return Action.EAST
        elif c < x and Action.WEST in allowed:
            return Action.WEST
        else:
            raise ValueError("No simple path found")

    def step(self, obs):
        raise NotImplemented("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
    """
	H1 agent always goes to the closest food
	"""

    name = "H1"

    def step(self, obs):
        try:
            r, c = self._closest_food(obs)
        except TypeError:
            return random.choice(obs.actions)

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H2(HeuristicAgent):
    """
	H2 Agent goes to the one visible dest which is furthest to the centre of visible players
	"""

    name = "H2"

    def step(self, obs):

        players_center = self._center_of_players(obs.players)

        try:
            r, c = self._farthest_food(obs, None, players_center)
        except TypeError:
            return random.choice(obs.actions)

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)

class H3(HeuristicAgent):
    """
	H3 Agent goes to the one visible dest which is closest to the centre of visible players
	"""

    name = "H3"

    def step(self, obs):

        players_center = self._center_of_players(obs.players)

        try:
            r, c = self._closest_food(obs, None, players_center)
        except TypeError:
            return random.choice(obs.actions)

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)

class H4(HeuristicAgent):
    """
    Agent goes to farthest food according to agent furthest from the agent.
    """
    name = "H4"
    def step(self, obs):
        farthest_player = self._furthest_player(obs.players, self.observed_position)
        if farthest_player is None:
            return random.choice(obs.actions)

        if not (obs.field > 0).any():
            try:
                return self._move_towards(farthest_player, obs.actions)
            except ValueError:
                return random.choice(obs.actions)

        try:
            r, c = self._farthest_food(obs, start=farthest_player)
        except TypeError:
            return random.choice(obs.actions)

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)
