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
from math import cos, pi, sin
from numpy import random, zeros
from PrivateAttributesDecorator import private_attributes_dec
from typing import Any


###################################################
### Define the 2-dimensional vector field class ###
###################################################
# Create the decorator needed for making the attributes private
vector_field_2d_decorator = private_attributes_dec("_all_base_vector_locations_col",	# class variables
												   "_all_base_vector_locations_row",
												   "_all_base_vectors_x",
												   "_all_base_vectors_y",
												   "_base_generated_flag",
												   "_n_base_vectors",
												   "_n_cols",
												   "_n_rows",
												   "_overwrites",
												   "_seed",
												   "_getValue")							# internal functions

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
		self._overwrites = {}
		for key in kwargs:
			self._overwrites[key] = kwargs[key]
		
		# Make sure that the combined default and overwrite values are valid
		# Fetch the needed generating values
		n_base_vectors_min = self._getValue(key = "n_base_vectors_min")
		n_base_vectors_mean = self._getValue(key = "n_base_vectors_mean")
		base_vector_length_min = self._getValue(key = "base_vector_length_min")
		base_vector_length_max = self._getValue(key = "base_vector_length_max")
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

		# Initialize additional needed storage
		self._all_base_vector_locations_col = None
		self._all_base_vector_locations_row = None
		self._all_base_vectors_x = None
		self._all_base_vectors_y = None
		self._base_generated_flag = False
		self._n_base_vectors = None
		self._seed = None

	### Define internal function for combining default and overwrite values ###
	def _getValue(self, key:str) -> Any:
		# Combine the default and overwrite values
		# Verify the inputs
		assert type(key) == str, "VectorField2D::_getValue: Provided value for 'key' must be a str object"
		assert key in self._DEFAULT_VALUES, "VectorField2D::_getValue: Provided value for 'key' must appear in '_DEFAULT_VALUES'"
		
		# Return the result
		return self._overwrites.get(key, self._DEFAULT_VALUES[key])

	### Define function used for generating vector field information ###
	def generateBaseVectors(self, seed:int = None):
		# Generate the base vectors of the vector field (using the given seed if needed)
		# Verify the inputs
		assert type(seed) == int, "VectorField2D::generateBaseVectors: Provided value for 'seed' must be a int object"
		assert 0 <= seed and seed < 2**32, "VectorField2D::generateBaseVector: Provided value for 'seed' must be >= 0 and < 2^32"

		# Mark that all bae vectors have been not been generated
		self._base_generated_flag = False

		# Set the random seed (if needed)
		if seed is not None:
			self._seed = seed
			random.seed(seed = seed)

		# Fetch the needed generating values
		n_base_vectors_min = self._getValue(key = "n_base_vectors_min")
		n_base_vectors_mean = self._getValue(key = "n_base_vectors_mean")
		base_vector_length_min = self._getValue(key = "base_vector_length_min")
		base_vector_length_max = self._getValue(key = "base_vector_length_max")

		# Generate the number of base vectors
		self._n_base_vectors = random.poisson(lam = n_base_vectors_mean - n_base_vectors_min) + n_base_vectors_min

		# Initialize the arrays for the base vectors and their locations
		self._all_base_vectors_x = zeros(self._n_base_vectors, dtype = float)
		self._all_base_vectors_y = zeros(self._n_base_vectors, dtype = float)
		self._all_base_vector_locations_row = zeros(self._n_base_vectors, dtype = int)
		self._all_base_vector_locations_col = zeros(self._n_base_vectors, dtype = int)

		# Generate the base vectors and their locations on the image
		for base_vector_index in range(self._n_base_vectors):
			# Randomly generate the angle and length of the base vector
			base_vector_angle = 2 * pi * random.rand()
			base_vector_length = base_vector_length_min + (base_vector_length_max - base_vector_length_min) * random.rand()

			# Create the x-value and y-value of the base vector
			base_vector_x = base_vector_length * cos(base_vector_angle)
			base_vector_y = base_vector_length * sin(base_vector_angle)

			# Randomly generate the base vector location
			base_vector_row = int(random.randint(0, self._n_rows))
			base_vector_col = int(random.randint(0, self._n_cols))

			# Add these to the relevant arrays
			self._all_base_vectors_x[base_vector_index] = base_vector_x
			self._all_base_vectors_y[base_vector_index] = base_vector_y
			self._all_base_vector_locations_row[base_vector_index] = base_vector_row
			self._all_base_vector_locations_col[base_vector_index] = base_vector_col

		# Mark that all bae vectors have been generated
		self._base_generated_flag = True

	def generateAllVectors(self):
		# Generate all other vectors in the vector field
		# Only proceed if the base vectors have been generated
		assert self._base_generated_flag == True, "VectorField2D::generateAllVectors: Only able to generate all vectors once base vectors have been generated"


a = VectorField2D(n_rows = 1080, n_cols = 1920)
a.generateBaseVectors(seed = 0)