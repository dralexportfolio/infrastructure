##########################################
### Import needed general dependencies ###
##########################################
# Add paths for internal modules
# Import dependencies
from pathlib import Path
from sys import path
# Get the shared infrastructure folder
infrastructure_folder = Path(__file__).parent.parent
# Add the needed paths
path.insert(0, str(infrastructure_folder.joinpath("common_needs")))

# Built-in modules
from abc import ABC, abstractmethod
from typing import Tuple

# Internal modules
from privacy_helper import privacyDecorator
from type_helper import isListWithNumericEntries


####################################################################################
### Define an abstract agent class to be used for concrete agent implementations ###
####################################################################################
# Define the list of private agent attributes
agent_private_attributes = ["_lower_bounds",	# class variables
							"_n_inputs",
							"_n_outputs",
							"_upper_bounds"]

# Create the decorator needed for making the attributes private
agent_decorator = privacyDecorator(agent_private_attributes)

# Define the class with private attributes
@agent_decorator
class Agent(ABC):
	### Initialize the class ###
	def __init__(self, n_inputs:int, n_outputs:int, lower_bounds:list, upper_bounds:list):
		# Verify the inputs
		assert type(n_inputs) == int, "Agent::__init__: Provided value for 'n_inputs' must be an int object"
		assert n_inputs > 0, "Agent::__init__: Provided value for 'n_inputs' must be positive"
		assert type(n_outputs) == int, "Agent::__init__: Provided value for 'n_outputs' must be an int object"
		assert n_outputs > 0, "Agent::__init__: Provided value for 'n_outputs' must be positive"
		assert isListWithNumericEntries(lower_bounds, include_numpy_flag = True) == True, "Agent::__init__: Provided value for 'lower_bounds' must be a list of numeric entries"
		assert isListWithNumericEntries(upper_bounds, include_numpy_flag = True) == True, "Agent::__init__: Provided value for 'upper_bounds' must be a list of numeric entries"
		assert len(lower_bounds) == n_inputs, "Agent::__init__: Provided value for 'lower_bounds' must be a list of length 'n_inputs"
		assert len(upper_bounds) == n_inputs, "Agent::__init__: Provided value for 'upper_bounds' must be a list of length 'n_inputs"
		for input_index in range(n_inputs):
			assert -float("inf") < lower_bounds[input_index] and lower_bounds[input_index] < float("inf"), "Agent::__init__: Provided value for 'lower_bounds' must be a list of finite numbers"
			assert -float("inf") < upper_bounds[input_index] and upper_bounds[input_index] < float("inf"), "Agent::__init__: Provided value for 'upper_bounds' must be a list of finite numbers"
			assert lower_bounds[input_index] < upper_bounds[input_index], "Agent::__init__: Entries in provided value for 'lower_bounds' must be less than their corresponding values in 'upper_bounds'"

		# Store the provided values
		self._n_inputs = n_inputs
		self._n_outputs = n_outputs
		self._lower_bounds = lower_bounds
		self._upper_bounds = upper_bounds

	### Define needed concrete methods ###
	def getInputCount(self) -> int:
		# Return the stored number of inputs
		return self._n_inputs

	def getLowerBounds(self) -> list:
		# Return the stored list of lower bounds
		return self._lower_bounds

	def getOutputCount(self) -> int:
		# Return the stored number of outputs
		return self._n_outputs

	def getUpperBounds(self) -> list:
		# Return the stored list of upper bounds
		return self._upper_bounds

	### Define needed abstract methods ###
	@abstractmethod
	def evaluate(self, all_inputs:list) -> Tuple:
		# Evaluate the agent on the provided inputs
		# Verify the inputs
		assert isListWithNumericEntries(all_inputs, include_numpy_flag = True) == True, "Agent::evaluate: Provided value for 'all_inputs' must be a list of numeric entries"
		assert len(all_inputs) == self._n_inputs, "Agent::evaluate: Provided value for 'all_inputs' must be a list of length equal to the stored number of inputs (in this case " + str(self._n_inputs) + ")"
		for input_index in range(self._n_inputs):
			assert self._lower_bounds[input_index] <= all_inputs[input_index] and all_inputs[input_index] <= self._upper_bounds[input_index], "Agent::evaluate: Entries in provided value for 'all_entries' must fall between the stored lower and upper bounds for each input index"