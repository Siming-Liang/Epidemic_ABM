import mesa_geo as mg
import shapely
from shapely.geometry import Point
from math import exp


class PersonAgent(mg.GeoAgent):
    def __init__(
            self,
            model,
            geometry,
            crs,
            state,
            home_neighborhood
    ):
        super().__init__(model, geometry, crs)
        self.state = state
        self.time_until_infected = None
        self.home_neighborhood = home_neighborhood
        self.hub = None
        self.hub_index = None
        self.time_at_hub = None
        self.nearby_hubs = None
        self.weights = None
        self.hubs_left = 0
        self.current_weights = []
        home_neighborhood.agents.append(self)

    def __repr__(self):
        return f'Person {self.unique_id}'

    def bake_movement_distribution(self):
        self.nearby_hubs = []
        self.weights = []

        for hub in self.model.all_hubs.values():
            d = shapely.distance(self.geometry, hub.geometry) / 1000
            if d <= self.model.max_travel_distance:
                self.nearby_hubs.append(hub)
                self.weights.append(exp(-d**2 / (2 * self.model.selection_length**2)))

        assert len(self.nearby_hubs) > 0, 'Agent has no nearby hubs!'

        self.current_weights = self.weights[:]
        self.hubs_left = len(self.nearby_hubs)

    def spread_chance(self):
        return self.hub.neighborhood.spread_chance

    def recovery_chance(self):
        return self.home_neighborhood.recovery_chance

    def gain_resistance_chance(self):
        return self.home_neighborhood.gain_resistance_chance

    def movement_step(self):
        if self.hub is None or self.time_at_hub == self.hub.duration:
            if self.hub is not None:
                self.current_weights[self.hub_index] = 0
                self.hub.agents.discard(self)
                self.hubs_left -= 1

            if self.hubs_left == 0:
                self.hub_index = None
                self.hub = None
            else:
                self.hub_index = self.random.choices(
                    range(len(self.nearby_hubs)),
                    weights=self.current_weights
                )[0]
                self.hub = self.nearby_hubs[self.hub_index]
                self.hub.agents.add(self)
                self.time_at_hub = 1
        else:
            self.time_at_hub += 1

    def infection_step(self):
        if self.state != 'I' or self.hub is None:
            return

        for agent in self.hub.agents:
            if agent.state != 'S':
                continue

            if self.random.random() < self.spread_chance():
                agent.time_until_infected = self.model.incubation_period

    def recovery_step(self):
        if self.time_until_infected is not None:
            self.time_until_infected -= 1

        if self.state == 'I' and self.random.random() < self.recovery_chance():
            if self.random.random() < self.gain_resistance_chance():
                self.state = 'R'
            else:
                self.state = 'S'
        elif self.time_until_infected is not None and self.time_until_infected == 0:
            self.state = 'I'
            self.time_until_infected = None

    def home_step(self):
        self.current_weights = self.weights[:]
        self.hubs_left = len(self.nearby_hubs)

        if self.hub is not None:
            self.hub.agents.discard(self)
            self.hub = None
            self.hub_index = None

    def dump_state(self):
        return {
            'unique_id': self.unique_id,
            'geometry': [self.geometry.x, self.geometry.y],
            'state': self.state,
            'time_until_infected': self.time_until_infected,
            'home_neighborhood': self.home_neighborhood.HOODNUM,
            'hub': None if self.hub is None else self.hub.unique_id,
            'hub_index': self.hub_index,
            'weights': self.weights,
            'current_weights': self.current_weights,
            'nearby_hubs': [h.unique_id for h in self.nearby_hubs],
            'hubs_left': self.hubs_left
        }

    def load_state(self, state):
        self.unique_id = state['unique_id']
        self.geometry = Point(state['geometry'])
        self.state = state['state']
        self.time_until_infected = state['time_until_infected']
        self.home_neighborhood = self.model.neighborhoods[state['home_neighborhood']]
        self.home_neighborhood.agents.append(self)
        self.hub = None if state['hub'] is None else self.model.all_hubs[state['hub']]
        if self.hub is not None:
            self.hub.agents.add(self)
        self.hub_index = state['hub_index']
        self.weights = state['weights']
        self.current_weights = state['current_weights']
        self.nearby_hubs = [
            self.model.all_hubs[unique_id] for unique_id in state['nearby_hubs']
        ]
        self.hubs_left = state['hubs_left']


class HubAgent(mg.GeoAgent):
    def __init__(
            self,
            model,
            geometry,
            crs,
            neighborhood,
            duration=None
    ):
        super().__init__(model, geometry, crs)
        self.neighborhood = neighborhood
        neighborhood.hubs.append(self)

        if duration is None:
            duration = self.model.random.randint(1, self.model.cycle_period)
        self.duration = duration

        self.agents = set()

    def __repr__(self):
        return f'Hub {self.unique_id}'

    def dump_state(self):
        return {
            'neighborhood': self.neighborhood.HOODNUM,
            'duration': self.duration,
            'unique_id': self.unique_id,
            'geometry': [self.geometry.x, self.geometry.y]
        }

    def load_state(self, state):
        self.neighborhood = self.model.neighborhoods[state['neighborhood']]
        self.neighborhood.hubs.append(self)
        self.duration = state['duration']
        self.unique_id = state['unique_id']
        self.geometry = Point(state['geometry'])


class NeighborhoodAgent(mg.GeoAgent):
    def __init__(
            self,
            model,
            geometry,
            crs,
            spread_chance=0.1,
            recovery_chance=0.3,
            gain_resistance_chance=0.5
    ):
        super().__init__(model, geometry, crs)

        self.agents = []
        self.hubs = []

        self.spread_chance = spread_chance
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

    def __repr__(self):
        return f'Neighborhood {self.HOODNUM}'

    def dump_state(self):
        return {
            'HOODNUM': self.HOODNUM,
            'spread_chance': self.spread_chance,
            'recovery_chance': self.recovery_chance,
            'gain_resistance_chance': self.gain_resistance_chance
        }

    def load_state(self, state):
        self.spread_chance = state['spread_chance']
        self.recovery_chance = state['recovery_chance']
        self.gain_resistance_chance = state['gain_resistance_chance']

    def bake_movement_distributions(self):
        for agent in self.agents:
            agent.bake_movement_distribution()


class MeasurementAgent(mg.GeoAgent):
    def __init__(self, model, geometry, crs):
        super().__init__(model, geometry, crs)
        self.people = []

    def measure(self, state):
        return sum(1 for p in self.people if p.state == state)
