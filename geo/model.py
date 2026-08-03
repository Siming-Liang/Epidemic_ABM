import mesa
import mesa_geo as mg
import pointpats
import numpy as np
import shapely
import matplotlib.pyplot as plt
from matplotlib import animation
from shapely.geometry import Point

from .agents import NeighborhoodAgent, HubAgent, PersonAgent, MeasurementAgent

NEIGHBORHOOD_FILE = 'geo/data/TorontoNeighbourhoods.geojson'


def measure_local(model, state):
    result = np.zeros(len(model.measurement_sites), dtype=int)
    for i, site in enumerate(model.measurement_sites):
        result[i] = site.measure(state)

    return result


def measure_local_susceptible(model):
    return measure_local(model, 'S')


def measure_local_infected(model):
    return measure_local(model, 'I')


def measure_local_resistant(model):
    return measure_local(model, 'R')


def measure_global(model, state):
    return sum(1 for p in model.all_people if p.state == state)


def measure_global_susceptible(model):
    return measure_global(model, 'S')


def measure_global_infected(model):
    return measure_global(model, 'I')


def measure_global_resistant(model):
    return measure_global(model, 'R')


class GeoDiseaseModel(mesa.Model):
    def __init__(
            self,
            cycle_period=10,
            max_travel_distance=5,
            selection_length=1,
            incubation_period=5,
            pop_density=50,
            hub_density=5,
            num_measures=20,
            num_initial_infected=1,
            retain_states_if_possible=True
    ):
        super().__init__()
        self.space = mg.GeoSpace(warn_crs_conversion=False)
        self.cycle_period = cycle_period
        self.max_travel_distance = max_travel_distance
        self.selection_length = selection_length
        self.incubation_period = incubation_period
        self.retain_states_if_possible = retain_states_if_possible

        self.neighborhoods = {}
        self.all_people = []
        self.all_hubs = {}
        self.measurement_sites = []

        ac = mg.AgentCreator(NeighborhoodAgent, model=self)
        neighborhoods = ac.from_file(NEIGHBORHOOD_FILE)
        self.space.add_agents(neighborhoods)

        self.neighborhoods = {n.HOODNUM: n for n in neighborhoods}

        if pop_density is not None:
            self.randomly_initialize(pop_density, hub_density, num_measures, num_initial_infected)

    def clear_agents(self):
        all_agents = list(self.space.agents)
        for agent in all_agents:
            self.space.remove_agent(agent)

        self.space.add_agents(self.neighborhoods.values())

    def randomly_initialize(self, pop_density, hub_density, num_measures, num_initial_infected):
        self.clear_agents()
        self.all_people = []
        self.all_hubs = {}
        self.measurement_sites = []

        for neighborhood in self.neighborhoods.values():
            area_km = neighborhood.geometry.area / 1e6
            xy = np.stack(neighborhood.geometry.exterior.coords.xy, axis=-1)
            window = pointpats.Window([xy.tolist()])
            homes = pointpats.PoissonPointProcess(
                window,
                n=pop_density * area_km,
                samples=1,
                conditioning=True,
                asPP=False
            ).realizations[0]

            ac = mg.AgentCreator(
                PersonAgent,
                model=self,
                crs=self.space.crs,
                agent_kwargs=dict(state='S', home_neighborhood=neighborhood)
            )
            people = [ac.create_agent(Point(location)) for location in homes]
            self.all_people.extend(people)
            self.space.add_agents(people)

            hub_locations = pointpats.PoissonPointProcess(
                window,
                n=hub_density * area_km,
                samples=1,
                conditioning=True,
                asPP=False
            ).realizations[0]

            ac = mg.AgentCreator(
                HubAgent,
                model=self,
                crs=self.space.crs,
                agent_kwargs=dict(neighborhood=neighborhood)
            )
            hubs = [ac.create_agent(Point(location)) for location in hub_locations]
            self.all_hubs.update({h.unique_id: h for h in hubs})
            self.space.add_agents(hubs)

            neighborhood.bake_movement_distributions()

        whole_area = shapely.union_all([n.geometry for n in self.neighborhoods.values()])
        window = pointpats.Window([np.stack(whole_area.exterior.coords.xy, axis=-1).tolist()])
        measurement_locations = pointpats.PoissonPointProcess(
            window,
            n=num_measures,
            samples=1,
            conditioning=False,
            asPP=False
        ).realizations[0]

        ac = mg.AgentCreator(MeasurementAgent, model=self, crs=self.space.crs)
        self.measurement_sites = [
            ac.create_agent(Point(location)) for location in measurement_locations
        ]
        self.space.add_agents(self.measurement_sites)
        self.bake_measurement_sites()

        for agent in self.random.sample(self.all_people, num_initial_infected):
            agent.state = 'I'

    def step(self):
        if self.steps % self.cycle_period == 0:
            for agent in self.all_people:
                agent.home_step()

        for agent in self.all_people:
            agent.movement_step()

        for agent in self.all_people:
            agent.infection_step()

        for agent in self.all_people:
            agent.recovery_step()

    def get_measurement_sites_by_neighborhood(self):
        indices = []
        for neighborhood_key in self.neighborhood_keys:
            n_indices = []
            for i, site in enumerate(self.measurement_sites):
                neighborhood = self.neighborhoods[neighborhood_key].geometry
                if shapely.contains(neighborhood, site.geometry):
                    n_indices.append(i)
            indices.append(n_indices)

        return indices

    @property
    def hub_ids(self):
        return sorted(self.all_hubs)

    @property
    def neighborhood_keys(self):
        return sorted(self.neighborhoods)

    def get_spread_chance(self):
        return np.array([self.neighborhoods[k].spread_chance for k in self.neighborhood_keys])

    def get_recovery_chance(self):
        return np.array([self.neighborhoods[k].recovery_chance for k in self.neighborhood_keys])

    def get_gain_resistance_chance(self):
        return np.array([self.neighborhoods[k].gain_resistance_chance for k in self.neighborhood_keys])

    def set_spread_chance(self, spread_chance):
        for k, value in zip(self.neighborhood_keys, spread_chance):
            self.neighborhoods[k].spread_chance = value

    def set_recovery_chance(self, recovery_chance):
        for k, value in zip(self.neighborhood_keys, recovery_chance):
            self.neighborhoods[k].recovery_chance = value

    def set_gain_resistance_chance(self, gain_resistance_chance):
        for k, value in zip(self.neighborhood_keys, gain_resistance_chance):
            self.neighborhoods[k].gain_resistance_chance = value

    def match_sir_statistics_in_region(self, measurement_site, s, i, r):
        agents = list(measurement_site.people)

        if self.retain_states_if_possible:
            self.random.shuffle(agents)
            agents_by_state = {
                state: [a for a in agents if a.state == state]
                for state in 'SIR'
            }

            ds = s - len(agents_by_state['S'])
            di = i - len(agents_by_state['I'])
            dr = r - len(agents_by_state['R'])

            correction_agents = (
                agents_by_state['S'][:max(0, -ds)]
                + agents_by_state['I'][:max(0, -di)]
                + agents_by_state['R'][:max(0, -dr)]
            )
            self.random.shuffle(correction_agents)

            states = 'S' * max(0, ds) + 'I' * max(0, di) + 'R' * max(0, dr)

            for agent, state in zip(correction_agents, states):
                agent.state = state
                agent.time_until_infected = None  # Cancel pending infection
        else:
            states = 'S' * s + 'I' * i + 'R' * r
            self.random.shuffle(states)
            for agent, reassigned_state in zip(agents, states):
                agent.state = reassigned_state
                agent.time_until_infected = None  # Cancel pending infection

    def match_sir_statistics(self, s, i, r):
        for site, s_site, i_site, r_site in zip(self.measurement_sites, s, i, r):
            self.match_sir_statistics_in_region(site, s_site, i_site, r_site)

    def bake_measurement_sites(self):
        distance_matrix = np.array([
            [
                shapely.distance(site.geometry, p.geometry)
                for site in self.measurement_sites
            ]
            for p in self.all_people
        ])
        closest_sites = np.argmin(distance_matrix, axis=1)
        for p, i_closest_site in zip(self.all_people, closest_sites):
            self.measurement_sites[i_closest_site].people.append(p)

    def dump_state(self):
        return {
            'steps': self.steps,
            'neighborhoods': [n.dump_state() for n in self.neighborhoods.values()],
            'hubs': [h.dump_state() for h in self.all_hubs.values()],
            'people': [p.dump_state() for p in self.all_people],
            'measurement_sites': [
                [site.geometry.x, site.geometry.y]
                for site in self.measurement_sites
            ]
        }

    def load_state(self, state):
        self.clear_agents()

        self.steps = state['steps']
        for neighborhood_state in state['neighborhoods']:
            self.neighborhoods[neighborhood_state['HOODNUM']].load_state(neighborhood_state)

        hubs = [
            HubAgent(
                model=self,
                geometry=Point(h['geometry']),
                crs=self.space.crs,
                neighborhood=self.neighborhoods[h['neighborhood']],
                duration=h['duration']
            )
            for h in state['hubs']
        ]
        self.space.add_agents(hubs)

        for hub, h in zip(hubs, state['hubs']):
            hub.unique_id = h['unique_id']

        self.all_hubs = {hub.unique_id: hub for hub in hubs}

        self.all_people = [
            PersonAgent(
                model=self,
                geometry=Point(p['geometry']),
                crs=self.space.crs,
                state=p['state'],
                home_neighborhood=self.neighborhoods[p['home_neighborhood']]
            )
            for p in state['people']
        ]
        self.space.add_agents(self.all_people)

        for person, p in zip(self.all_people, state['people']):
            person.load_state(p)

        self.measurement_sites = [
            MeasurementAgent(
                model=self,
                geometry=Point(site),
                crs=self.space.crs
            )
            for site in state['measurement_sites']
        ]
        self.space.add_agents(self.measurement_sites)
        self.bake_measurement_sites()


class GeoDiseaseMovie:
    def __init__(self, model, steps, frame_length=500):
        self.model = model
        fig, ax = plt.subplots()

        x, y, c = self.get_scatter_data()
        self.scatter_points = ax.scatter(x, y, c=c)

        self.animation = animation.FuncAnimation(
            fig,
            self.update,
            steps,
            interval=frame_length
        )

    def get_scatter_data(self):
        x, y = [], []
        c = []
        color_map = {
            'S': 'green',
            'I': 'red',
            'R': 'gray'
        }
        for agent in self.model.all_people:
            xy = agent.geometry.xy
            x.append(xy[0])
            y.append(xy[1])
            c.append(color_map[agent.state])

        return x, y, c

    def update(self, _):
        self.model.step()
        _, _, c = self.get_scatter_data()
        self.scatter_points.set_color(c)
