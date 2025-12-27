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
from tkinter_helper import askSaveFilename
from type_helper import isNumeric

# External modules
from math import cos, pi, sin
from multiprocessing import Pool
from numpy import dot, random, uint8, zeros
from numpy import max as np_max
from numpy import min as np_min
from os import cpu_count
from PIL.Image import fromarray
from PrivateAttributesDecorator import private_attributes_dec
from scipy.special import softmax
from typing import Any, Tuple


###################################################
### Define the 2-dimensional vector field class ###
###################################################
# Define a wrapper function so that the class can use multiprocessing to compute vectors
def proxyComputeVector(payload:Tuple[Any, Tuple[int, int]]) -> Tuple[float, float]:
	# Allow for computation of all vectors in the field to be computed in parallel
	# Extract the needed values
	vector_field = payload[0]
	input_pair = payload[1]

	# Extract the row and column indices from the input pair
	row_index = input_pair[0]
	col_index = input_pair[1]

	# Evaluate the vector field at the given input pair
	return vector_field.computeVector(row_index = row_index, col_index = col_index)

# Define a wrapper function so that the class can use multiprocessing to compute curl and divergence
def proxyComputeCurlDivergence(payload:Tuple[Any, Tuple[int, int]]) -> Tuple[float, float]:
	# Allow for computation of all curl and divergence values in the field to be computed in parallel
	# Extract the needed values
	vector_field = payload[0]
	input_pair = payload[1]

	# Extract the row and column indices from the input pair
	row_index = input_pair[0]
	col_index = input_pair[1]

	# Evaluate the vector field at the given input pair
	return vector_field.computeCurlDivergence(row_index = row_index, col_index = col_index)

# Create the decorator needed for making the attributes private
vector_field_2d_decorator = private_attributes_dec("_all_base_vector_locations_col",	# class variables
												   "_all_base_vector_locations_row",
												   "_all_base_vectors_x",
												   "_all_base_vectors_y",
												   "_all_curls",
												   "_all_divergences",
												   "_all_generated_flag",
												   "_all_vectors_x",
												   "_all_vectors_y",
												   "_base_generated_flag",
												   "_curl_divergence_generated_flag",
												   "_n_base_vectors",
												   "_n_cols",
												   "_n_rows",
												   "_overwrites",
												   "_seed",
												   "_softmax_normalizer",
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
		self._all_curls = None
		self._all_divergences = None
		self._all_generated_flag = False
		self._all_vectors_x = None
		self._all_vectors_y = None
		self._base_generated_flag = False
		self._curl_divergence_generated_flag = False
		self._n_base_vectors = None
		self._seed = None
		self._softmax_normalizer = None

	### Define internal function for combining default and overwrite values ###
	def _getValue(self, key:str) -> Any:
		# Combine the default and overwrite values
		# Verify the inputs
		assert type(key) == str, "VectorField2D::_getValue: Provided value for 'key' must be a str object"
		assert key in self._DEFAULT_VALUES, "VectorField2D::_getValue: Provided value for 'key' must appear in '_DEFAULT_VALUES'"
		
		# Return the result
		return self._overwrites.get(key, self._DEFAULT_VALUES[key])

	### Define functions used for vectors in the field ###
	def generateBaseVectors(self, seed:int = None):
		# Generate the base vectors of the vector field (using the given seed if needed)
		# Verify the inputs
		assert type(seed) == int, "VectorField2D::generateBaseVectors: Provided value for 'seed' must be a int object"
		assert 0 <= seed and seed < 2**32, "VectorField2D::generateBaseVector: Provided value for 'seed' must be >= 0 and < 2^32"

		# Mark that needed information has not been generated
		self._base_generated_flag = False
		self._all_generated_flag = False
		self._curl_divergence_generated_flag = False

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

		# Mark that the base vectors have been generated
		self._base_generated_flag = True

	def computeVector(self, row_index:int, col_index:int) -> Tuple[float, float]:
		# Compute the vector at the given row and column indices
		# Verify the inputs
		assert type(row_index) == int, "VectorField2D::computeVector: Provided value for 'row_index' must be an int object"
		assert type(col_index) == int, "VectorField2D::computeVector: Provided value for 'col_index' must be an int object"
		assert 0 <= row_index and row_index < self._n_rows, "VectorField2D::computeVector: Provided value for 'row_index' must be >= 0 and < the number of rows in the field"
		assert 0 <= col_index and col_index < self._n_cols, "VectorField2D::computeVector: Provided value for 'col_index' must be >= 0 and < the number of columns in the field"

		# Compute the squared distances from this point to each base location
		all_squared_distances = zeros(self._n_base_vectors, dtype = float)
		for base_vector_index in range(self._n_base_vectors):
			delta_row = self._all_base_vector_locations_row[base_vector_index] - row_index
			delta_col = self._all_base_vector_locations_col[base_vector_index] - col_index
			all_squared_distances[base_vector_index] = delta_row**2 + delta_col**2

		# Compute the weights using the softmax
		all_weights = softmax(-all_squared_distances / self._softmax_normalizer**2)

		# Compute the x-value and y-value of the vector
		vector_x = dot(all_weights, self._all_base_vectors_x)
		vector_y = dot(all_weights, self._all_base_vectors_y)

		# Return the results
		return (vector_x, vector_y)

	def generateAllVectors(self, softmax_normalizer:float):
		# Generate all other vectors in the vector field
		# Only proceed if the base vectors have been generated
		assert self._base_generated_flag == True, "VectorField2D::generateAllVectors: Only able to generate all vectors once base vectors have been generated"

		# Mark that needed information has not been generated
		self._all_generated_flag = False
		self._curl_divergence_generated_flag = False

		# Verify the inputs
		assert isNumeric(softmax_normalizer, include_numpy_flag = False) == True, "VectorField2D::generateAllVectors: Provided value for 'softmax_normalizer' must be a float or int object"
		assert 0 < softmax_normalizer and softmax_normalizer < float("inf"), "VectorField2D::generateAllVectors: Provided value for 'softmax_normalizer' must be positive and finite"

		# Store the provided softmax normalizer value
		self._softmax_normalizer = softmax_normalizer

		# Create the input tuples for the process
		all_inputs = []
		for row_index in range(self._n_rows):
			for col_index in range(self._n_cols):
				all_inputs.append((self, (row_index, col_index)))

		# Initialize a pool with the needed number of processes, run the computation in parallel, and end by closing the pool
		pool = Pool(processes = max(cpu_count() - 1, 1))
		all_outputs = pool.map(proxyComputeVector, all_inputs)
		pool.close()

		# Compute the vector associated with each point
		self._all_vectors_x = zeros((self._n_rows, self._n_cols), dtype = float)
		self._all_vectors_y = zeros((self._n_rows, self._n_cols), dtype = float)
		for index in range(self._n_rows * self._n_cols):
			input_pair = all_inputs[index]
			output_pair = all_outputs[index]
			self._all_vectors_x[input_pair[1][0], input_pair[1][1]] = output_pair[0]
			self._all_vectors_y[input_pair[1][0], input_pair[1][1]] = output_pair[1]

		# Mark that all vectors have been generated
		self._all_generated_flag = True

	### Define functions for computing and displaying information about the vector field ###
	def computeCurlDivergence(self, row_index:int, col_index:int) -> Tuple[float, float]:
		# Compute the curl and divergence of the vector field at the given location
		# Verify the inputs
		assert type(row_index) == int, "VectorField2D::computeCurlDivergence: Provided value for 'row_index' must be an int object"
		assert type(col_index) == int, "VectorField2D::computeCurlDivergence: Provided value for 'col_index' must be an int object"
		assert 0 <= row_index and row_index < self._n_rows, "VectorField2D::computeCurlDivergence: Provided value for 'row_index' must be >= 0 and < the number of rows in the field"
		assert 0 <= col_index and col_index < self._n_cols, "VectorField2D::computeCurlDivergence: Provided value for 'col_index' must be >= 0 and < the number of columns in the field"

		# Numerically compute the vector field's derivatives with respect to x at this index pair
		if col_index == 0:
			d_vector_x_dx = self._all_vectors_x[row_index, col_index + 1] - self._all_vectors_x[row_index, col_index]
			d_vector_y_dx = self._all_vectors_y[row_index, col_index + 1] - self._all_vectors_y[row_index, col_index]
		elif col_index == self._n_cols - 1:
			d_vector_x_dx = self._all_vectors_x[row_index, col_index] - self._all_vectors_x[row_index, col_index - 1]
			d_vector_y_dx = self._all_vectors_y[row_index, col_index] - self._all_vectors_y[row_index, col_index - 1]
		else:
			d_vector_x_dx = (self._all_vectors_x[row_index, col_index + 1] - self._all_vectors_x[row_index, col_index - 1]) / 2
			d_vector_y_dx = (self._all_vectors_y[row_index, col_index + 1] - self._all_vectors_y[row_index, col_index - 1]) / 2

		# Numerically compute the vector field's derivatives with respect to y at this index pair
		if row_index == 0:
			d_vector_x_dy = self._all_vectors_x[row_index + 1, col_index] - self._all_vectors_x[row_index, col_index]
			d_vector_y_dy = self._all_vectors_y[row_index + 1, col_index] - self._all_vectors_y[row_index, col_index]
		elif row_index == self._n_rows - 1:
			d_vector_x_dy = self._all_vectors_x[row_index, col_index] - self._all_vectors_x[row_index - 1, col_index]
			d_vector_y_dy = self._all_vectors_y[row_index, col_index] - self._all_vectors_y[row_index - 1, col_index]
		else:
			d_vector_x_dy = (self._all_vectors_x[row_index + 1, col_index] - self._all_vectors_x[row_index - 1, col_index]) / 2
			d_vector_y_dy = (self._all_vectors_y[row_index + 1, col_index] - self._all_vectors_y[row_index - 1, col_index]) / 2

		# Compute the curl and divergence
		curl = d_vector_y_dx - d_vector_x_dy
		divergence = d_vector_x_dx + d_vector_y_dy

		# Return the results
		return (curl, divergence)

	def generateAllCurlDivergence(self):
		# Compute all curl and divergence values for the vector field
		# Only proceed if all vectors have been generated
		assert self._all_generated_flag == True, "VectorField2D::generateAllCurlDivergence: Only able to generate curl and divergence once all vectors have been generated"

		# Mark that needed information has not been generated
		self._curl_divergence_generated_flag = False

		# Create the input tuples for the process
		all_inputs = []
		for row_index in range(self._n_rows):
			for col_index in range(self._n_cols):
				all_inputs.append((self, (row_index, col_index)))

		# Initialize a pool with the needed number of processes, run the computation in parallel, and end by closing the pool
		pool = Pool(processes = max(cpu_count() - 1, 1))
		all_outputs = pool.map(proxyComputeCurlDivergence, all_inputs)
		pool.close()

		# Compute the vector associated with each point
		self._all_curls = zeros((self._n_rows, self._n_cols), dtype = float)
		self._all_divergences = zeros((self._n_rows, self._n_cols), dtype = float)
		for index in range(self._n_rows * self._n_cols):
			input_pair = all_inputs[index]
			output_pair = all_outputs[index]
			self._all_curls[input_pair[1][0], input_pair[1][1]] = output_pair[0]
			self._all_divergences[input_pair[1][0], input_pair[1][1]] = output_pair[1]

		# Mark that the curl and divergence have been generated
		self._curl_divergence_generated_flag = True

	def plotCurl(self, show_flag:bool = True, save_flag:bool = False):
		# Plot the curl of the vector field
		# Only proceed if curl and divergence have been generated
		assert self._curl_divergence_generated_flag == True, "VectorField2D::plotCurl: Only able to plot curl and divergence once all curl and divergence values have been generated"

		# Verify the inputs
		assert type(show_flag) == bool, "VectorField2D::plotCurl: Provided value for 'show_flag' must be a bool object"
		assert type(save_flag) == bool, "VectorField2D::plotCurl: Provided value for 'save_flag' must be a bool object"
		assert show_flag == True or save_flag == True, "VectorField2D::plotCurl: At least of the provided values for 'show_flag' and 'save_flag' must be True"

		# Compute the maximum and minimum values of the curl
		max_curl = np_max(self._all_curls)
		min_curl = np_min(self._all_curls)

		# Compute the maximum magnitude curl
		max_magnitude_curl = max(abs(max_curl), abs(min_curl))

		# Initialize the RGB array used for the image
		curl_rgb_array = zeros((self._n_rows, self._n_cols, 3), dtype = float)

		# Compute the raw RGB values associated with each pixel
		for row_index in range(self._n_rows):
			for col_index in range(self._n_cols):
				# Get the curl for this point
				curl = self._all_curls[row_index, col_index]

				# Compute the RGB values for the image
				if curl > 0:
					curl_rgb_array[row_index, col_index, 2] = int(255 * curl / max_magnitude_curl)
				elif curl < 0:
					curl_rgb_array[row_index, col_index, 0] = int(-255 * curl / max_magnitude_curl)

		# Create the image from the RGB array
		curl_image = fromarray(curl_rgb_array.astype(uint8), "RGB")

		# Show the image (if needed)
		if show_flag == True:
			curl_image.show()

		# Save the image (if needed)
		if save_flag == True:
			# Get a path to which the image should be saved and make sure cancel wasn't clicked
			image_path = askSaveFilename(allowed_extensions = ["png"])
			assert image_path is not None, "VectorField2D::plotCurl: Unable to save image because cancel button was clicked"
			# Save the image to this location
			curl_image.save(image_path, "PNG")

	def plotDivergence(self, show_flag:bool = True, save_flag:bool = False):
		# Plot the divergence of the vector field
		# Only proceed if curl and divergence have been generated
		assert self._curl_divergence_generated_flag == True, "VectorField2D::plotDivergence: Only able to plot curl and divergence once all curl and divergence values have been generated"

		# Verify the inputs
		assert type(show_flag) == bool, "VectorField2D::plotDivergence: Provided value for 'show_flag' must be a bool object"
		assert type(save_flag) == bool, "VectorField2D::plotDivergence: Provided value for 'save_flag' must be a bool object"
		assert show_flag == True or save_flag == True, "VectorField2D::plotDivergence: At least of the provided values for 'show_flag' and 'save_flag' must be True"

		# Compute the maximum and minimum values of the divergence
		max_divergence = np_max(self._all_divergences)
		min_divergence = np_min(self._all_divergences)

		# Compute the maximum magnitude divergence
		max_magnitude_divergence = max(abs(max_divergence), abs(min_divergence))

		# Initialize the RGB array used for the image
		divergence_rgb_array = zeros((self._n_rows, self._n_cols, 3), dtype = float)

		# Compute the raw RGB values associated with each pixel
		for row_index in range(self._n_rows):
			for col_index in range(self._n_cols):
				# Get the divergence for this point
				divergence = self._all_divergences[row_index, col_index]

				# Compute the RGB values for the image
				if divergence > 0:
					divergence_rgb_array[row_index, col_index, 2] = int(255 * divergence / max_magnitude_divergence)
				elif divergence < 0:
					divergence_rgb_array[row_index, col_index, 0] = int(-255 * divergence / max_magnitude_divergence)

		# Create the image from the RGB array
		divergence_image = fromarray(divergence_rgb_array.astype(uint8), "RGB")

		# Show the image (if needed)
		if show_flag == True:
			divergence_image.show()

		# Save the image (if needed)
		if save_flag == True:
			# Get a path to which the image should be saved and make sure cancel wasn't clicked
			image_path = askSaveFilename(allowed_extensions = ["png"])
			assert image_path is not None, "VectorField2D::plotDivergence: Unable to save image because cancel button was clicked"
			# Save the image to this location
			divergence_image.save(image_path, "PNG")

if __name__ == "__main__":
	a = VectorField2D(n_rows = 1080, n_cols = 1920)
	a.generateBaseVectors(seed = 0)
	a.generateAllVectors(softmax_normalizer = 320)
	a.generateAllCurlDivergence()
	a.plotCurl(show_flag = False, save_flag = True)
	a.plotDivergence(show_flag = False, save_flag = True)