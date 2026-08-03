"""
Objects related to disease model and agent properties
"""
import math
from enum import Enum
from abc import ABC, abstractmethod


class DiseaseState(Enum):
    """Represents what state of disease an agent is in"""
    SUSCEPTIBLE = 0
    INFECTED = 1
    RESISTANT = 2


class BaseDiseaseProps(ABC):
    """Represents the properties of a disease model"""

    def agent_spread_chance(self, agent) -> float:
        """Query spread chance for an agent"""
        return self.spread_chance(agent.model, agent.cell)

    def agent_recovery_chance(self, agent) -> float:
        """Query recovery chance for an agent"""

        return self.recovery_chance(agent.model, agent.cell)

    def agent_gain_resistance_chance(self, agent) -> float:
        """Query gain resistance chance for an agent"""
        return self.gain_resistance_chance(agent.model, agent.cell)

    @abstractmethod
    def spread_chance(self, model, cell) -> float:
        """
        Get the spread chance in a given cell
        :param model: Model for which to query the spread chance
        :param cell: Cell at which to query the spread chance
        :return: The spread chance at the given cell
        """
        raise NotImplemented

    @abstractmethod
    def recovery_chance(self, model, cell) -> float:
        """
        Get the recovery chance in a given cell
        :param model: Model for which to query the recovery chance
        :param cell: Cell at which to query the recovery chance
        :return: The recovery chance at the given cell
        """
        raise NotImplemented

    @abstractmethod
    def gain_resistance_chance(self, model, cell) -> float:
        """
        Get the gain resistance chance in a given cell
        :param model: Model for which to query the gain resistance chance
        :param cell: Cell at which to query the gain resistance chance
        :return: The gain resistance chance at the given cell
        """
        raise NotImplemented

    @abstractmethod
    def dump_state(self):
        """
        Dumps disease properties state for serialization
        :return: A state dictionary representing the current state
        """
        raise NotImplemented

    @abstractmethod
    def load_state(self, state):
        """
        Sets current state from a given state dictionary produced by
        dump_state
        :param state: State dictionary to load
        """
        raise NotImplemented


class ConstantDiseaseProps(BaseDiseaseProps):
    def __init__(
            self,
            spread_chance: float,
            recovery_chance: float,
            gain_resistance_chance: float,
    ):
        """
        Disease model properties that are constant for all cells

        :param spread_chance: Chance of disease spreading from an infected agent
            to a susceptible agent
        :param recovery_chance: Chance of an infected agent recovering
        :param gain_resistance_chance: Given that an infected agent recovers,
            the probability of the agent gaining resistance
        """
        self.spread_chance_constant = spread_chance
        self.recovery_chance_constant = recovery_chance
        self.gain_resistance_chance_constant = gain_resistance_chance

    def spread_chance(self, *_):
        return self.spread_chance_constant

    def recovery_chance(self, *_):
        return self.recovery_chance_constant

    def gain_resistance_chance(self, *_):
        return self.gain_resistance_chance_constant

    def dump_state(self):
        return dict(
            spread_chance=self.spread_chance_constant,
            recovery_chance=self.recovery_chance_constant,
            gain_resistance_chance=self.gain_resistance_chance_constant
        )

    def load_state(self, state):
        self.spread_chance_constant = state['spread_chance']
        self.recovery_chance_constant = state['recovery_chance']
        self.gain_resistance_chance_constant = state['gain_resistance_chance']


class MeasurementRegionExponentialDiseaseProps(BaseDiseaseProps):
    def __init__(
            self,
            max_spread_chance: float,
            spread_chance_x: float,
            spread_chance_y: float,
            max_recovery_chance: float,
            recovery_chance_x: float,
            recovery_chance_y: float,
            max_gain_resistance_chance: float,
            gain_resistance_chance_x: float,
            gain_resistance_chance_y: float
    ):
        """
        Disease properties that vary by measurement region, with exponential
        decay in the measurement region coordinates
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
        """
        self.max_spread_chance = max_spread_chance
        self.spread_chance_x = spread_chance_x
        self.spread_chance_y = spread_chance_y

        self.max_recovery_chance = max_recovery_chance
        self.recovery_chance_x = recovery_chance_x
        self.recovery_chance_y = recovery_chance_y

        self.max_gain_resistance_chance = max_gain_resistance_chance
        self.gain_resistance_chance_x = gain_resistance_chance_x
        self.gain_resistance_chance_y = gain_resistance_chance_y

    def dump_state(self):
        return dict(
            max_spread_chance=self.max_spread_chance,
            spread_chance_x=self.spread_chance_x,
            spread_chance_y=self.spread_chance_y,
            max_recovery_chance=self.max_recovery_chance,
            recovery_chance_x=self.recovery_chance_x,
            recovery_chance_y=self.recovery_chance_y,
            max_gain_resistance_chance=self.max_gain_resistance_chance,
            gain_resistance_chance_x=self.gain_resistance_chance_x,
            gain_resistance_chance_y=self.gain_resistance_chance_y,
        )

    def load_state(self, state):
        self.max_spread_chance = state['max_spread_chance']
        self.spread_chance_x = state['spread_chance_x']
        self.spread_chance_y = state['spread_chance_y']

        self.max_recovery_chance = state['max_recovery_chance']
        self.recovery_chance_x = state['recovery_chance_x']
        self.recovery_chance_y = state['recovery_chance_y']

        self.max_gain_resistance_chance = state['max_gain_resistance_chance']
        self.gain_resistance_chance_x = state['gain_resistance_chance_x']
        self.gain_resistance_chance_y = state['gain_resistance_chance_y']

    def spread_chance(self, model, cell) -> float:
        x, y = cell.coordinate
        x = x // model.measurement_region_width
        y = y // model.measurement_region_height
        e = math.exp(-self.spread_chance_x * x - self.spread_chance_y * y)
        return self.max_spread_chance * e

    def recovery_chance(self, model, cell) -> float:
        x, y = cell.coordinate
        x = x // model.measurement_region_width
        y = y // model.measurement_region_height
        e = math.exp(-self.recovery_chance_x * x - self.recovery_chance_y * y)
        return self.max_recovery_chance * e

    def gain_resistance_chance(self, model, cell) -> float:
        x, y = cell.coordinate
        x = x // model.measurement_region_width
        y = y // model.measurement_region_height
        e = math.exp(-self.gain_resistance_chance_x * x - self.gain_resistance_chance_y * y)
        return self.max_gain_resistance_chance * e


class MeasurementRegionLookupTableDiseaseProps(BaseDiseaseProps):
    def __init__(
            self,
            spread_chance_table,
            recovery_chance_table,
            gain_resistance_chance_table
    ):
        self.spread_chance_table = spread_chance_table
        self.recovery_chance_table = recovery_chance_table
        self.gain_resistance_chance_table = gain_resistance_chance_table

    def dump_state(self):
        return dict(
            spread_chance_table=self.spread_chance_table,
            recovery_chance_table=self.recovery_chance_table,
            gain_resistance_chance_table=self.gain_resistance_chance_table
        )

    def load_state(self, state):
        self.spread_chance_table = state['spread_chance_table']
        self.recovery_chance_table = state['recovery_chance_table']
        self.gain_resistance_chance_table = state['gain_resistance_chance_table']

    def spread_chance(self, model, cell) -> float:
        x, y = cell.coordinate
        x = x // model.measurement_region_width
        y = y // model.measurement_region_height
        return self.spread_chance_table[x, y]

    def recovery_chance(self, model, cell) -> float:
        x, y = cell.coordinate
        x = x // model.measurement_region_width
        y = y // model.measurement_region_height
        return self.recovery_chance_table[x, y]

    def gain_resistance_chance(self, model, cell) -> float:
        x, y = cell.coordinate
        x = x // model.measurement_region_width
        y = y // model.measurement_region_height
        return self.gain_resistance_chance_table[x, y]
