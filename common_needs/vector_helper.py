##########################################
### Import needed general dependencies ###
##########################################
# Add paths for internal modules
# Import dependencies
from pathlib import Path
from sys import path
# Get the shared parent folder
parent_folder = Path(__file__).parent.parent
# Add the needed paths
path.insert(0, str(parent_folder.joinpath("common_needs")))

# Internal modules
from type_helper import isNumeric

# External modules
from PrivateAttributesDecorator import private_attributes_dec
from typing import Any


###################################################
### Define the 2-dimensional vector field class ###
###################################################
# Create the decorator needed for making the attributes private
vector_field_2d_decorator = private_attributes_dec("_n_cols",			# class variables
												   "_n_rows",
												   "_overwrites",
												   "_getValue")			# internal functions

# Define the class with private attributes
class VectorField2D:
	### Set the default values for the class ###
	_DEFAULT_VALUES = {
		# Minimum and mean numbers of base vectors
		"n_base_vectors_min": 5,
		"n_base_vectors_mean": 10,
		# Minimum and maximum base vector lengths
		"base_vector_length_min": 0.1,
		"base_vector_length_max": 10
	}
	
	### Initialize the class ###
	def __init__(self, n_rows:int, n_cols:int, **kwargs):
		# Verify the inputs
		assert type(n_rows) == int, "VectorField2D::__init__: Provided value for 'n_rows' must be an int object"
		assert type(n_cols) == int, "VectorField2D::__init__: Provided value for 'n_cols' must be an int object"
		for key in kwargs:
			assert key in self._DEFAULT_VALUES, "VectorField2D::__init__: Provided keyword arguments must be keys in '_DEFAULT_VALUES'"
		
		# Store the provided values
		self._n_rows = n_rows
		self._n_cols = n_cols
		self._overwriten_values = {}
		for key in kwargs:
			self._overwrites[key] = kwargs[key]
		
		# Make sure that the combined default and overwrite values are valid
		# Fetch the needed values
		n_base_vectors_min = self._getValue(key = "n_base_vectors_min")
		n_base_vectors_mean = self._getValue(key = "n_base_vectors_mean")
		base_vector_length_min = self._getValue(key = "base_vector_length_min")
		base_vector_length_mean = self._getValue(key = "base_vector_length_mean")
		# Minimum and mean numbers of base vectors
		assert type(n_base_vectors_min) in [float, int], "VectorField2D::__init__: Provided value for 'n_base_vectors_min' must be a float or int object"
		assert type(n_base_vectors_mean) in [float, int], "VectorField2D::__init__: Provided value for 'n_base_vectors_mean' must be a float or int object"
		assert 1 <= n_base_vectors_min and n_base_vectors_min < float("inf"), "VectorField2D::__init__: Provided value for 'n_base_vectors_min' must be >= 1 and finite"
		assert n_base_vectors_min <= n_base_vectors_mean and n_base_vectors_mean < float("inf"), "VectorField2D::__init__: Provided value for 'n_base_vectors_mean' must be >= 'n_base_vectors_min' and finite"
		# Minimum and maximum base vector lengths
		assert type(base_vector_length_min) in [float, int], "VectorField2D::__init__: Provided value for 'base_vector_length_min' must be a float or int object"
		assert type(base_vector_length_max) in [float, int], "VectorField2D::__init__: Provided value for 'base_vector_length_max' must be a float or int object"
		assert 0 < base_vector_length_min and base_vector_length_min < float("inf"), "VectorField2D::__init__: Provided value for 'base_vector_length_min' must be positive and finite"
		assert base_vector_length_min <= base_vector_length_max and base_vector_length_max < float("inf"), "VectorField2D::__init__: Provided value for 'base_vector_length_max' must be >= 'base_vector_length_min' and finite"
		
	### Define internal function for combining default and overwrite values ###
	def _getValue(self, key:str) -> Any:
		# Combine the default and overwrite values
		# Verify the inputs
		assert type(key) == str, "VectorField2D::_getValue: Provided value for 'key' must be a str object"
		assert key in self._DEFAULT_VALUES, "VectorField2D::_getValue: Provided value for 'key' must appear in '_DEFAULT_VALUES'"
		
		# Return the result
		return self._overwrites.get(key, self._DEFAULT_VALUES[key])
