"""
Disease agent implementation
"""
from mesa.discrete_space import CellAgent

from disease_data import DiseaseState


class DiseaseAgent(CellAgent):
    def __init__(self, model, cell):
        super(DiseaseAgent, self).__init__(model)
        self.cell = cell
        self.state = DiseaseState.SUSCEPTIBLE
        self.new_state = self.state

    def __repr__(self):
        return f'[{self.state} @ {self.cell.coordinate}]'

    def move(self):
        if self.model.agents_can_move:
            self.cell = self.cell.neighborhood.select_random_cell()

    def try_to_infect_neighbors(self):
        for agent in self.cell.get_neighborhood(radius=self.model.influence_radius).agents:
            if (agent.state is DiseaseState.SUSCEPTIBLE) and (
                self.random.random() < self.model.disease_props.agent_spread_chance(self)
            ):
                agent.new_state = DiseaseState.INFECTED

    def try_to_remove_infection(self):
        if self.random.random() < self.model.disease_props.agent_gain_resistance_chance(self):
            self.new_state = DiseaseState.RESISTANT
        else:
            self.new_state = DiseaseState.SUSCEPTIBLE

    def update_state(self):
        self.state = self.new_state

    def step(self):
        self.move()
        if self.state is DiseaseState.INFECTED:
            self.try_to_infect_neighbors()

            if self.random.random() < self.model.disease_props.agent_recovery_chance(self):
                self.try_to_remove_infection()
