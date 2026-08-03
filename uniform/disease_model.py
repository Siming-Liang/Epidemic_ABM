"""
Disease model implementations
"""
import mesa
from mesa.discrete_space import OrthogonalMooreGrid, CellCollection
from mesa.model import Model
import numpy as np

import disease_agent
from disease_data import (
    DiseaseState,
    BaseDiseaseProps,
    ConstantDiseaseProps,
    MeasurementRegionExponentialDiseaseProps, MeasurementRegionLookupTableDiseaseProps
)


def measure_global_state(model, state):
    """Measures the number of agents in the whole simulation that are in the given state"""
    return sum(1 for a in model.grid.all_cells.agents if a.state is state)


def measure_global_susceptible(model):
    """Measures the number of susceptible agents in the whole simulation"""
    return measure_global_state(model, DiseaseState.SUSCEPTIBLE)


def measure_global_infected(model):
    """Measure the number of infected agents in the whole simulation"""
    return measure_global_state(model, DiseaseState.INFECTED)


def measure_global_resistant(model):
    """Measures the number of resistant agents in the whole simulation"""
    return measure_global_state(model, DiseaseState.RESISTANT)


def measure_region_state(model, state, region_x, region_y):
    """Measures the number of agents in the given measurement region and given disease state"""
    return sum(
        1 for a in model.cells_in_region(region_x, region_y).agents
        if a.state is state
    )


def measure_local_state(model, state):
    """
    Measures the number of agents in the given disease state by region.

    :param model: The model to measure
    :param state: Which state to count the number of agents in
    :return: (measurement_grid_width, measurement_grid_height) array of
        measurements. The first index is the region x coordinate and the second
        index is the region y coordinate, and the value is the number of agents
        in the given state at the region with those coordinates.
    """
    m = np.zeros((model.measurement_grid_width, model.measurement_grid_height), dtype=int)
    for region_x in range(model.measurement_grid_width):
        for region_y in range(model.measurement_grid_height):
            m[region_x, region_y] = measure_region_state(
                model, state,
                region_x, region_y
            )

    return m


def measure_local_susceptible(model):
    """Measures the number of susceptible agents by region (see measure_local_state)"""
    return measure_local_state(model, DiseaseState.SUSCEPTIBLE)


def measure_local_infected(model):
    """Measures the number of infected agents by region (see measure_local_state)"""
    return measure_local_state(model, DiseaseState.INFECTED)


def measure_local_resistant(model):
    """Measures the number of resistant agents by region (see measure_local_state)"""
    return measure_local_state(model, DiseaseState.RESISTANT)


def _cover(a, b):
    # Get the smallest multiple of b greater than or equal to a
    return ((a + b - 1) // b) * b


class BaseDiseaseModel(Model):
    def __init__(
            self,
            num_agents: int,
            measurement_grid_width: int,
            measurement_grid_height: int,
            state_grid_width: int,
            state_grid_height: int,
            initial_outbreak_size: int,
            influence_radius: int,
            agents_can_move: bool,
            disease_props: BaseDiseaseProps,
            retain_states_if_possible: bool = True,
            collect_initial: bool = True,
            seed=None
    ):
        """
        Base class for all disease models.

        Models the spread of a disease among moving agents on a grid. The grid
        on which the agents move is called the *state grid*. The model collects
        data on a coarser grid called the *measurement grid*. Each cell of the
        measurement grid (which I call a *measurement region*) comprises a fixed
        number of cells in the state grid. We measure the number of susceptible,
        infected, and resistant in each measurement grid cell, i.e., the total
        number in each state grid cell within the measurement grid cell.

        :param num_agents: Total number of agents in simulation
        :param measurement_grid_width: Number of measurement regions
            horizontally
        :param measurement_grid_height: Number of measurement regions vertically
        :param state_grid_width: Number of cells in the state grid horizontally.
            Must have measurement_grid_width | state_grid_width; if not,
            state_grid_width is set to the nearest multiple of
            measurement_grid_width greater than the given width.
        :param state_grid_height: Number of cells in the state grid vertically.
            Must have measurement_grid_height | state_grid_height; if not,
            state_grid_height is set to the nearest multiple of
            measurement_grid_height greater than the given height.
        :param initial_outbreak_size: Initial number of infected agents
        :param influence_radius: How many cells away an agent can infect another
            agent from
        :param agents_can_move: Whether agents can move or not
        :param disease_props: Properties of the disease model. Must be an
            instance of BaseDiseaseProps, which provides the parameters on
            a per-state-grid-cell basis (see disease_data.BaseDiseaseProps for
            information on what the model parameters are).
        :param retain_states_if_possible: Whether to retain as many agent states
            as possible when matching given SIR statistics. If False, all
            agents will be randomly reassigned to match the statistics; if True,
            only a minimal number of randomly-selected agents will be
            reassigned.
        :param collect_initial: Whether to collect data for the initial state
        :param seed: The random seed

        """
        super().__init__(seed=seed)
        self.disease_props = disease_props
        self.retain_states_if_possible = retain_states_if_possible
        self.influence_radius = influence_radius
        self.agents_can_move = agents_can_move

        self.num_agents = num_agents
        self.measurement_grid_width = measurement_grid_width
        self.measurement_grid_height = measurement_grid_height
        self.state_grid_width = _cover(state_grid_width, measurement_grid_width)
        self.state_grid_height = _cover(state_grid_height, measurement_grid_height)
        self.measurement_region_width = self.state_grid_width // measurement_grid_width
        self.measurement_region_height = self.state_grid_height // measurement_grid_height
        self.grid = OrthogonalMooreGrid(
            (self.state_grid_width, self.state_grid_height),
            random=self.random
        )

        self.initial_outbreak_size = min(
            initial_outbreak_size,
            num_agents
        )

        disease_agent.DiseaseAgent.create_agents(
            self,
            num_agents,
            list(self.random.choices(list(self.grid.all_cells), k=num_agents)),
        )

        # Infect some agents
        for a in self.random.sample(list(self.agents), initial_outbreak_size):
            a.state = DiseaseState.INFECTED

        self.datacollector = mesa.DataCollector(
            {
                "Infected": measure_global_infected,
                "Susceptible": measure_global_susceptible,
                "Resistant": measure_global_resistant,
                "Infected-local": measure_local_infected,
                "Susceptible-local": measure_local_susceptible,
                "Resistant-local": measure_local_resistant
            }
        )

        # Collect data for initial state
        if collect_initial:
            self.datacollector.collect(self)

        self.running = True

    def dump_state(self):
        state = {
            'steps': self.steps,
            'disease_props': self.disease_props.dump_state(),
            'random': self.random.getstate()
        }

        agents = []
        for agent in self.agents:
            agents.append({
                'coordinate': list(agent.cell.coordinate),
                'disease_state': agent.state.value,
                'new_state': agent.new_state.value
            })
        state['agents'] = agents

        return state

    def load_state(self, state):
        if 'steps' in state:
            self.steps = state['steps']
        if 'disease_props' in state:
            self.disease_props.load_state(state['disease_props'])
        if 'random' in state:
            self.random.setstate(state['random'])

        # Agents are fungible, so we only need to ensure that all the same
        # states are represented (identity of the agents doesn't matter)
        if 'agents' in state:
            for agent, agent_state in zip(self.agents, state['agents']):
                agent.cell = self.grid[tuple(agent_state['coordinate'])]
                agent.state = DiseaseState(agent_state['disease_state'])

    def reassign_agents_in_region(self, region_x, region_y, s, i, r):
        agents = list(self.cells_in_region(region_x, region_y).agents)
        self.random.shuffle(agents)

        if self.retain_states_if_possible:
            agents_by_state = {
                state: [a for a in agents if a.state is state]
                for state in DiseaseState
            }

            ds = s - len(agents_by_state[DiseaseState.SUSCEPTIBLE])
            di = i - len(agents_by_state[DiseaseState.INFECTED])
            dr = r - len(agents_by_state[DiseaseState.RESISTANT])

            correction_agents = (
                agents_by_state[DiseaseState.SUSCEPTIBLE][:max(0, -ds)]
                + agents_by_state[DiseaseState.INFECTED][:max(0, -di)]
                + agents_by_state[DiseaseState.RESISTANT][:max(0, -dr)]
            )
            self.random.shuffle(correction_agents)

            states = (
                [DiseaseState.SUSCEPTIBLE] * max(0, ds)
                + [DiseaseState.INFECTED] * max(0, di)
                + [DiseaseState.RESISTANT] * max(0, dr)
            )
            for agent, state in zip(correction_agents, states):
                agent.state = state
                agent.new_state = state
        else:
            states = ([DiseaseState.SUSCEPTIBLE] * s
                      + [DiseaseState.INFECTED] * i
                      + [DiseaseState.RESISTANT] * r)
            self.random.shuffle(states)
            for agent, reassigned_state in zip(agents, states):
                agent.state = reassigned_state
                agent.new_state = reassigned_state

    def match_sir_statistics(self, s, i, r):
        for region_x in range(self.measurement_grid_width):
            for region_y in range(self.measurement_grid_height):
                self.reassign_agents_in_region(
                    region_x,
                    region_y,
                    s[region_x, region_y],
                    i[region_x, region_y],
                    r[region_x, region_y]
                )

    def cells_in_region(self, region_x, region_y):
        x_lo = region_x * self.measurement_region_width
        y_lo = region_y * self.measurement_region_height
        cell_list = []
        for y in range(y_lo, y_lo + self.measurement_region_height):
            for x in range(x_lo, x_lo + self.measurement_region_width):
                cell_list.append(self.grid[(x, y)])

        return CellCollection(cell_list, random=self.random)

    def get_spread_chance(self):
        spread_chance = np.zeros((self.state_grid_width, self.state_grid_height))
        for x in range(spread_chance.shape[0]):
            for y in range(spread_chance.shape[1]):
                spread_chance[x, y] = self.disease_props.spread_chance(
                    self, self.grid[(x, y)]
                )

        return spread_chance

    def get_recovery_chance(self):
        recovery_chance = np.zeros((self.state_grid_width, self.state_grid_height))
        for x in range(recovery_chance.shape[0]):
            for y in range(recovery_chance.shape[1]):
                recovery_chance[x, y] = self.disease_props.recovery_chance(
                    self, self.grid[(x, y)]
                )

        return recovery_chance

    def get_gain_resistance_chance(self):
        gain_resistance_chance = np.zeros((self.state_grid_width, self.state_grid_height))
        for x in range(gain_resistance_chance.shape[0]):
            for y in range(gain_resistance_chance.shape[1]):
                gain_resistance_chance[x, y] = self.disease_props.gain_resistance_chance(
                    self, self.grid[(x, y)]
                )

        return gain_resistance_chance

    def step(self):
        self.agents.shuffle_do("step")
        self.agents.do("update_state")

        # collect data, if required
        if hasattr(self, "datacollector"):
            self.datacollector.collect(self)


class ConstantDiseaseModel(BaseDiseaseModel):
    def __init__(
            self,
            num_agents=10,
            measurement_grid_width=4,
            measurement_grid_height=4,
            state_grid_width=8,
            state_grid_height=8,
            initial_outbreak_size=1,
            influence_radius=1,
            agents_can_move=True,
            spread_chance=0.4,
            recovery_chance=0.3,
            gain_resistance_chance=0.5,
            retain_states_if_possible=True,
            collect_initial=True,
            seed=None,
    ):
        """
        Agent-based SIR model of disease spread in which model parameters are
        constants in space and time.

        :param num_agents: Total number of agents in simulation
        :param measurement_grid_width: Number of measurement regions
            horizontally
        :param measurement_grid_height: Number of measurement regions vertically
        :param state_grid_width: Number of cells in the state grid horizontally.
            Must have measurement_grid_width | state_grid_width; if not,
            state_grid_width is set to the nearest multiple of
            measurement_grid_width greater than the given width.
        :param state_grid_height: Number of cells in the state grid vertically.
            Must have measurement_grid_height | state_grid_height; if not,
            state_grid_height is set to the nearest multiple of
            measurement_grid_height greater than the given height.
        :param initial_outbreak_size: Initial number of infected agents
        :param influence_radius: How many cells away an agent can infect another
            agent from
        :param agents_can_move: Whether agents can move or not
        :param spread_chance: Chance of virus spreading from an infected to a
            susceptible agent that are in neighboring cells
        :param recovery_chance: Chance of an infected agent recovering
        :param gain_resistance_chance: Chance that an infected agent gains
            resistance, given that it recovers (if resistance is not gained,
            the agent becomes susceptible again)
        :param retain_states_if_possible: Whether to retain as many agent states
            as possible when matching given SIR statistics. If False, all
            agents will be randomly reassigned to match the statistics; if True,
            only a minimal number of randomly-selected agents will be
            reassigned.
        :param collect_initial: Whether to collect data for the initial state
        :param seed: The random seed
        """
        super().__init__(
            num_agents=num_agents,
            measurement_grid_width=measurement_grid_width,
            measurement_grid_height=measurement_grid_height,
            state_grid_width=state_grid_width,
            state_grid_height=state_grid_height,
            initial_outbreak_size=initial_outbreak_size,
            influence_radius=influence_radius,
            agents_can_move=agents_can_move,
            disease_props=ConstantDiseaseProps(
                spread_chance=spread_chance,
                recovery_chance=recovery_chance,
                gain_resistance_chance=gain_resistance_chance,
            ),
            retain_states_if_possible=retain_states_if_possible,
            collect_initial=collect_initial,
            seed=seed
        )


class ExponentialDiseaseModel(BaseDiseaseModel):
    def __init__(
            self,
            num_agents=10,
            measurement_grid_width=4,
            measurement_grid_height=4,
            state_grid_width=8,
            state_grid_height=8,
            initial_outbreak_size=1,
            influence_radius=1,
            agents_can_move=True,
            max_spread_chance=0.6,
            spread_chance_x=1/3,
            spread_chance_y=1/3,
            max_recovery_chance=0.5,
            recovery_chance_x=1/3,
            recovery_chance_y=1/3,
            max_gain_resistance_chance=0.5,
            gain_resistance_chance_x=0.0,
            gain_resistance_chance_y=0.0,
            retain_states_if_possible=True,
            collect_initial=True,
            seed=None,
    ):
        """
        Agent-based SIR model of disease spread in which model parameters
        decay exponentially in space but are constant within measurement regions

        :param num_agents: Total number of agents in simulation
        :param measurement_grid_width: Number of measurement regions
            horizontally
        :param measurement_grid_height: Number of measurement regions vertically
        :param state_grid_width: Number of cells in the state grid horizontally.
            Must have measurement_grid_width | state_grid_width; if not,
            state_grid_width is set to the nearest multiple of
            measurement_grid_width greater than the given width.
        :param state_grid_height: Number of cells in the state grid vertically.
            Must have measurement_grid_height | state_grid_height; if not,
            state_grid_height is set to the nearest multiple of
            measurement_grid_height greater than the given height.
        :param initial_outbreak_size: Initial number of infected agents
        :param influence_radius: How many cells away an agent can infect another
            agent from
        :param agents_can_move: Whether agents can move or not
        :param max_spread_chance: Maximum spread chance (at region (0, 0))
        :param spread_chance_x: Decay coefficient for x for the spread chance
        :param spread_chance_y: Decay coefficient for y for the spread chance
        :param max_recovery_chance: Maximum recovery chance (at region (0, 0))
        :param recovery_chance_x: Decay coefficient for x for the recovery chance
        :param recovery_chance_y: Decay coefficient for y for the recovery chance
        :param max_gain_resistance_chance: Maximum gain resistance chance
            (at region (0, 0))
        :param gain_resistance_chance_x: Decay coefficient for x for the gain
            resistance chance
        :param gain_resistance_chance_y: Decay coefficient for y for the gain
            resistance chance
        :param retain_states_if_possible: Whether to retain as many agent states
            as possible when matching given SIR statistics. If False, all
            agents will be randomly reassigned to match the statistics; if True,
            only a minimal number of randomly-selected agents will be
            reassigned.
        :param collect_initial: Whether to collect data for the initial state
        :param seed: The random seed
        """
        super().__init__(
            num_agents=num_agents,
            measurement_grid_width=measurement_grid_width,
            measurement_grid_height=measurement_grid_height,
            state_grid_width=state_grid_width,
            state_grid_height=state_grid_height,
            initial_outbreak_size=initial_outbreak_size,
            influence_radius=influence_radius,
            agents_can_move=agents_can_move,
            disease_props=MeasurementRegionExponentialDiseaseProps(
                max_spread_chance=max_spread_chance,
                spread_chance_x=spread_chance_x,
                spread_chance_y=spread_chance_y,
                max_recovery_chance=max_recovery_chance,
                recovery_chance_x=recovery_chance_x,
                recovery_chance_y=recovery_chance_y,
                max_gain_resistance_chance=max_gain_resistance_chance,
                gain_resistance_chance_x=gain_resistance_chance_x,
                gain_resistance_chance_y=gain_resistance_chance_y,
            ),
            retain_states_if_possible=retain_states_if_possible,
            seed=seed
        )


class LookupTableDiseaseModel(BaseDiseaseModel):
    def __init__(
            self,
            spread_chance_table,
            recovery_chance_table,
            gain_resistance_chance_table,
            num_agents=4000,
            measurement_grid_width=10,
            measurement_grid_height=10,
            state_grid_width=20,
            state_grid_height=20,
            initial_outbreak_size=1,
            influence_radius=2,
            agents_can_move=False,
            retain_states_if_possible=True,
            collect_initial=True,
            seed=None
    ):
        """
        Agent-based SIR model of disease spread in which model parameters
        vary by measurement region and are given by look-up tables

        :param spread_chance_table: (W, H) matrix of spread chances,
            where W = measurement_grid_width, H = measurement_grid_height
        :param recovery_chance_table: (W, H) matrix of recovery chances
        :param gain_resistance_chance_table: (W, H) matrix of chances to gain
            resistance
        :param num_agents: Total number of agents in simulation
        :param measurement_grid_width: Number of measurement regions
            horizontally
        :param measurement_grid_height: Number of measurement regions vertically
        :param state_grid_width: Number of cells in the state grid horizontally.
            Must have measurement_grid_width | state_grid_width; if not,
            state_grid_width is set to the nearest multiple of
            measurement_grid_width greater than the given width.
        :param state_grid_height: Number of cells in the state grid vertically.
            Must have measurement_grid_height | state_grid_height; if not,
            state_grid_height is set to the nearest multiple of
            measurement_grid_height greater than the given height.
        :param initial_outbreak_size: Initial number of infected agents
        :param influence_radius: How many cells away an agent can infect another
            agent from
        :param agents_can_move: Whether agents can move or not
        :param retain_states_if_possible: Whether to retain as many agent states
            as possible when matching given SIR statistics. If False, all
            agents will be randomly reassigned to match the statistics; if True,
            only a minimal number of randomly-selected agents will be
            reassigned.
        :param collect_initial: Whether to collect data for the initial state
        :param seed: The random seed
        """
        super(LookupTableDiseaseModel, self).__init__(
            num_agents=num_agents,
            measurement_grid_width=measurement_grid_width,
            measurement_grid_height=measurement_grid_height,
            state_grid_width=state_grid_width,
            state_grid_height=state_grid_height,
            initial_outbreak_size=initial_outbreak_size,
            influence_radius=influence_radius,
            agents_can_move=agents_can_move,
            disease_props=MeasurementRegionLookupTableDiseaseProps(
                spread_chance_table=spread_chance_table,
                recovery_chance_table=recovery_chance_table,
                gain_resistance_chance_table=gain_resistance_chance_table
            ),
            retain_states_if_possible=retain_states_if_possible,
            collect_initial=collect_initial,
            seed=seed
        )
