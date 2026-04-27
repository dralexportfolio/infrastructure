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
path.insert(0, str(infrastructure_folder.joinpath("dimensional_analysis")))

# Built-in modules
from math import cos, pi, sin, sqrt
from multiprocessing import Pool
from os import cpu_count
from typing import Any, Tuple

# Internal modules
from dimension_reduction import performPCA
from privacy_helper import privacyDecorator
from tkinter_helper import askSaveFilename
from type_helper import isNumeric

# External modules
import matplotlib.pyplot as plt
from numpy import all, any, dot, isnan, isreal, meshgrid, ndarray, random, uint8, zeros
from numpy import max as np_max
from numpy import min as np_min
from PIL import Image
from scipy.special import softmax


############################################################################################
### Define helper functions needed for running the vector field computations in parallel ###
############################################################################################
# Define the global variables which will be needed for parallelization
global_ALL_BASE_VECTOR_LOCATIONS_COL = None
global_ALL_BASE_VECTOR_LOCATIONS_ROW = None
global_ALL_BASE_VECTORS_X = None
global_ALL_BASE_VECTORS_Y = None
global_ALL_VECTORS_X = None
global_ALL_VECTORS_Y = None
global_N_BASE_VECTORS = None
global_N_COLS = None
global_N_ROWS = None
global_SOFTMAX_NORMALIZER = None

# Define the worker initializer for computing vectors
def _initializeComputeVector(payload:Tuple[ndarray, ndarray, ndarray, ndarray, int, int, int, Any]):
	# Store the vector field parameters needed for vector computation in the global scope
	# Set the needed variables to be global in scope
	global global_ALL_BASE_VECTOR_LOCATIONS_COL
	global global_ALL_BASE_VECTOR_LOCATIONS_ROW
	global global_ALL_BASE_VECTORS_X
	global global_ALL_BASE_VECTORS_Y
	global global_N_BASE_VECTORS
	global global_N_COLS
	global global_N_ROWS
	global global_SOFTMAX_NORMALIZER

	# Store the provided values in the global scope
	global_ALL_BASE_VECTOR_LOCATIONS_COL = payload[0]
	global_ALL_BASE_VECTOR_LOCATIONS_ROW = payload[1]
	global_ALL_BASE_VECTORS_X = payload[2]
	global_ALL_BASE_VECTORS_Y = payload[3]
	global_N_BASE_VECTORS = payload[4]
	global_N_COLS = payload[5]
	global_N_ROWS = payload[6]
	global_SOFTMAX_NORMALIZER = payload[7]

# Define a wrapper function so that the class can use multiprocessing to compute vectors
def _proxyComputeVector(payload:Tuple[int, int]) -> Tuple[float, float]:
	# Allow for computation of all vectors in the field to be computed in parallel
	# Extract the needed values from the payload
	row_index = payload[0]
	col_index = payload[1]

	# Verify the inputs for row and column indices only (and just assume global values are correct coming from the VectorFieldGenerator class)
	assert type(row_index) == int, "_proxyComputeVector: Provided value for 'row_index' must be an int object"
	assert type(col_index) == int, "_proxyComputeVector: Provided value for 'col_index' must be an int object"
	assert 0 <= row_index and row_index < global_N_ROWS, "_proxyComputeVector: Provided value for 'row_index' must be >= 0 and < the number of rows in the field"
	assert 0 <= col_index and col_index < global_N_COLS, "_proxyComputeVector: Provided value for 'col_index' must be >= 0 and < the number of columns in the field"

	# Compute the squared distances from this point to each base location
	all_squared_distances = zeros(global_N_BASE_VECTORS, dtype = float)
	for base_vector_index in range(global_N_BASE_VECTORS):
		delta_row = global_ALL_BASE_VECTOR_LOCATIONS_ROW[base_vector_index] - row_index
		delta_col = global_ALL_BASE_VECTOR_LOCATIONS_COL[base_vector_index] - col_index
		all_squared_distances[base_vector_index] = delta_row**2 + delta_col**2

	# Compute the weights using the softmax
	all_weights = softmax(-all_squared_distances / global_SOFTMAX_NORMALIZER**2)

	# Compute the x-value and y-value of the vector
	vector_x = dot(all_weights, global_ALL_BASE_VECTORS_X)
	vector_y = dot(all_weights, global_ALL_BASE_VECTORS_Y)

	# Return the results
	return (vector_x, vector_y)

# Define the worker initializer for computing derivative-related values
def _initializeComputeDerivatives(payload:Tuple[ndarray, ndarray, int, int]):
	# Store the vector field parameters needed for derivative-related computation in the global scope
	# Set the needed variables to be global in scope
	global global_ALL_VECTORS_X
	global global_ALL_VECTORS_Y
	global global_N_COLS
	global global_N_ROWS

	# Store the provided values in the global scope
	global_ALL_VECTORS_X = payload[0]
	global_ALL_VECTORS_Y = payload[1]
	global_N_COLS = payload[2]
	global_N_ROWS = payload[3]

# Define a wrapper function so that the class can use multiprocessing to compute derivative-related values
def _proxyComputeDerivatives(payload:Tuple[int, int]) -> Tuple[float, float, float]:
	# Allow for computation of all derivative-related values in the field to be computed in parallel
	# Extract the needed values from the payload
	row_index = payload[0]
	col_index = payload[1]

	# Verify the inputs for row and column indices only (and just assume global values are correct coming from the VectorField class)
	assert type(row_index) == int, "_proxyComputeDerivatives: Provided value for 'row_index' must be an int object"
	assert type(col_index) == int, "_proxyComputeDerivatives: Provided value for 'col_index' must be an int object"
	assert 0 <= row_index and row_index < global_N_ROWS, "_proxyComputeDerivatives: Provided value for 'row_index' must be >= 0 and < the number of rows in the field"
	assert 0 <= col_index and col_index < global_N_COLS, "_proxyComputeDerivatives: Provided value for 'col_index' must be >= 0 and < the number of columns in the field"

	# Numerically compute the vector field's derivatives with respect to x at this index pair
	if col_index == 0:
		d_vector_x_dx = global_ALL_VECTORS_X[row_index, col_index + 1] - global_ALL_VECTORS_X[row_index, col_index]
		d_vector_y_dx = global_ALL_VECTORS_Y[row_index, col_index + 1] - global_ALL_VECTORS_Y[row_index, col_index]
	elif col_index == global_N_COLS - 1:
		d_vector_x_dx = global_ALL_VECTORS_X[row_index, col_index] - global_ALL_VECTORS_X[row_index, col_index - 1]
		d_vector_y_dx = global_ALL_VECTORS_Y[row_index, col_index] - global_ALL_VECTORS_Y[row_index, col_index - 1]
	else:
		d_vector_x_dx = (global_ALL_VECTORS_X[row_index, col_index + 1] - global_ALL_VECTORS_X[row_index, col_index - 1]) / 2
		d_vector_y_dx = (global_ALL_VECTORS_Y[row_index, col_index + 1] - global_ALL_VECTORS_Y[row_index, col_index - 1]) / 2

	# Numerically compute the vector field's derivatives with respect to y at this index pair
	if row_index == 0:
		d_vector_x_dy = global_ALL_VECTORS_X[row_index, col_index] - global_ALL_VECTORS_X[row_index + 1, col_index]
		d_vector_y_dy = global_ALL_VECTORS_Y[row_index, col_index] - global_ALL_VECTORS_Y[row_index + 1, col_index]
	elif row_index == global_N_ROWS - 1:
		d_vector_x_dy = global_ALL_VECTORS_X[row_index - 1, col_index] - global_ALL_VECTORS_X[row_index, col_index]
		d_vector_y_dy = global_ALL_VECTORS_Y[row_index - 1, col_index] - global_ALL_VECTORS_Y[row_index, col_index]
	else:
		d_vector_x_dy = (global_ALL_VECTORS_X[row_index - 1, col_index] - global_ALL_VECTORS_X[row_index + 1, col_index]) / 2
		d_vector_y_dy = (global_ALL_VECTORS_Y[row_index - 1, col_index] - global_ALL_VECTORS_Y[row_index + 1, col_index]) / 2

	# Compute the needed derivative-related values
	curl = d_vector_y_dx - d_vector_x_dy
	divergence = d_vector_x_dx + d_vector_y_dy
	jacobian = d_vector_x_dx * d_vector_y_dy - d_vector_x_dy * d_vector_y_dx

	# Return the results
	return (curl, divergence, jacobian)


###################################################
### Define the 2-dimensional vector field class ###
###################################################
# Create the decorator needed for making the attributes private
vector_field_decorator = privacyDecorator(["_all_curls",							# class variables
										   "_all_divergences",
										   "_all_jacobians",
										   "_all_vectors_x",
										   "_all_vectors_y",
										   "_derivatives_computed_flag",
										   "_n_cols",
										   "_n_rows"])

# Define the class with private attributes
@vector_field_decorator
class VectorField:
	### Initialize the class ###
	def __init__(self, n_rows:int, n_cols:int, all_vectors_x:ndarry, all_vectors_y:ndarry):
		# Verify the inputs
		# Row and column count
		assert type(n_rows) == int, "VectorField::__init__: Provided value for 'n_rows' must be an int object"
		assert 100 <= n_rows and n_rows <= 5000, "VectorField::__init__: Provided value for 'n_rows' must be >= 100 and <= 5000"
		assert type(n_cols) == int, "VectorField::__init__: Provided value for 'n_cols' must be an int object"
		assert 100 <= n_cols and n_cols <= 5000, "VectorField::__init__: Provided value for 'n_cols' must be >= 100 and <= 5000"
		# Vector array for x-values
		assert type(all_vectors_x) == ndarray, "VectorField::__init__: Provided value for 'all_vectors_x' must be a numpy.ndarray object"
		assert len(all_vectors_x.shape) == 2, "VectorField::__init__: Provided value for 'all_vectors_x' must be 2-dimensional array"
		assert all_vectors_x.shape[0] == n_rows, "VectorField::__init__: Provided value for 'all_vectors_x' must have a row count equal to 'n_rows'"
		assert all_vectors_x.shape[1] == n_cols, "VectorField::__init__: Provided value for 'all_vectors_x' must have a column count equal to 'n_cols'"
		assert any(isnan(all_vectors_x)) == False, "VectorField::__init__: Provided value for 'all_vectors_x' cannot have any nan values"
		assert all(isreal(all_vectors_x)) == True, "VectorField::__init__: Provided value for 'all_vectors_x' must have only real-valued entries"
		assert -float("inf") < np_min(all_vectors_x) and np_max(all_vectors_x) < float("inf"), "VectorField::__init__: Provided value for 'all_vectors_x' must have only finite entries"
		# Vector array for y-values
		assert type(all_vectors_y) == ndarray, "VectorField::__init__: Provided value for 'all_vectors_y' must be a numpy.ndarray object"
		assert len(all_vectors_y.shape) == 2, "VectorField::__init__: Provided value for 'all_vectors_y' must be 2-dimensional array"
		assert all_vectors_y.shape[0] == n_rows, "VectorField::__init__: Provided value for 'all_vectors_y' must have a row count equal to 'n_rows'"
		assert all_vectors_y.shape[1] == n_cols, "VectorField::__init__: Provided value for 'all_vectors_y' must have a column count equal to 'n_cols'"
		assert any(isnan(all_vectors_y)) == False, "VectorField::__init__: Provided value for 'all_vectors_y' cannot have any nan values"
		assert all(isreal(all_vectors_y)) == True, "VectorField::__init__: Provided value for 'all_vectors_y' must have only real-valued entries"
		assert -float("inf") < np_min(all_vectors_y) and np_max(all_vectors_y) < float("inf"), "VectorField::__init__: Provided value for 'all_vectors_y' must have only finite entries"
		
		# Store the provided values
		self._n_rows = n_rows
		self._n_cols = n_cols
		self._all_vectors_x = all_vectors_x
		self._all_vectors_y = all_vectors_y

		# Initialize additional needed storage
		self._all_curls = None
		self._all_divergences = None
		self._all_jacobians = None
		self._derivatives_computed_flag = False

	### Define a function for applying an affine transformation to the vector field ###
	def applyAffineTransformation(self, m_11:Any, m_12:Any, m_21:Any, m_22:Any, b_1:Any = 0, b_2:Any = 0):
		# Apply an affine transformation to each vector in the field
		# Verify the inputs
		# Types
		assert isNumeric(m_11, include_numpy_flag = True), "VectorField::applyAffineTransformation: Provided value for 'm_11' must be numeric"
		assert isNumeric(m_12, include_numpy_flag = True), "VectorField::applyAffineTransformation: Provided value for 'm_12' must be numeric"
		assert isNumeric(m_21, include_numpy_flag = True), "VectorField::applyAffineTransformation: Provided value for 'm_21' must be numeric"
		assert isNumeric(m_22, include_numpy_flag = True), "VectorField::applyAffineTransformation: Provided value for 'm_22' must be numeric"
		assert isNumeric(b_1, include_numpy_flag = True), "VectorField::applyAffineTransformation: Provided value for 'b_1' must be numeric"
		assert isNumeric(b_2, include_numpy_flag = True), "VectorField::applyAffineTransformation: Provided value for 'b_2' must be numeric"
		# Bounds
		assert -float("inf") < m_11 and m_11 < float("inf"), "VectorField::applyAffineTransformation: Provided value for 'm_11' must be finite"
		assert -float("inf") < m_12 and m_12 < float("inf"), "VectorField::applyAffineTransformation: Provided value for 'm_12' must be finite"
		assert -float("inf") < m_21 and m_21 < float("inf"), "VectorField::applyAffineTransformation: Provided value for 'm_21' must be finite"
		assert -float("inf") < m_22 and m_22 < float("inf"), "VectorField::applyAffineTransformation: Provided value for 'm_22' must be finite"
		assert -float("inf") < b_1 and b_1 < float("inf"), "VectorField::applyAffineTransformation: Provided value for 'b_1' must be finite"
		assert -float("inf") < b_2 and b_2 < float("inf"), "VectorField::applyAffineTransformation: Provided value for 'b_2' must be finite"

		# Loop over the vectors in the field and apply the needed transformation
		for row_index in range(self._n_rows):
			for col_index in range(self._n_cols):
				# Get the old values at this location
				old_vector_x = self._all_vectors_x[row_index, col_index]
				old_vector_y = self._all_vectors_y[row_index, col_index]

				# Compute and store the new values
				self._all_vectors_x[row_index, col_index] = m_11 * old_vector_x + m_12 * old_vector_y + b_1
				self._all_vectors_y[row_index, col_index] = m_21 * old_vector_x + m_22 * old_vector_y + b_2

		# Recompute the derivative information (if needed)
		if self._derivatives_computed_flag == True:
			self.computeDerivativeInfo()

	### Define a function for plotting the vector field ###
	def plotVectorField(self, gap_size:int, plot_type:str, show_flag:bool = True, save_flag:bool = False):
		# Create a plot of the vector field using matplotlib
		# Verify the inputs
		# Gap size
		assert type(gap_size) == int, "VectorField::plotVectorField: Provided value for 'gap_size' must be an int object"
		assert gap_size >= 10, "VectorField::plotVectorField: Provided value for 'gap_size' must be >= 10"
		assert gap_size <= self._n_rows / 4, "VectorField::plotVectorField: Provided value for 'gap_size' must be <= 25% of the number of rows"
		assert gap_size <= self._n_cols / 4, "VectorField::plotVectorField: Provided value for 'gap_size' must be <= 25% of the number of columns"
		# Plot type
		assert plot_type in ["quiver", "streamplot"], "VectorField::plotVectorField: Provided value for 'plot_type' must be 'quiver' or 'streamplot'"
		# Show/save flags
		assert type(show_flag) == bool, "VectorField::plotVectorField: Provided value for 'show_flag' must be a bool object"
		assert type(save_flag) == bool, "VectorField::plotVectorField: Provided value for 'save_flag' must be a bool object"
		assert show_flag == True or save_flag == True, "VectorField::plotVectorField: At least of the provided values for 'show_flag' and 'save_flag' must be True"

		# Create 1D arrays used to sample the rows and columns
		# Get the numbers x and y samples
		n_samples_x = int((self._n_cols - 1) / gap_size)
		n_samples_y = int((self._n_rows - 1) / gap_size)
		# Compute the shifts needed to center the grid
		shift_x = int(((self._n_cols - 1) - (n_samples_x * gap_size)) / 2)
		shift_y = int(((self._n_rows - 1) - (n_samples_y * gap_size)) / 2)
		# Generate the lists of x-value and y-value locations
		x_loc_list = [shift_x + gap_size * index for index in range(n_samples_x)]
		y_loc_list = [shift_y + gap_size * index for index in range(n_samples_y)]

		# Convert these lists to arrays using meshgrid
		x_loc_array, y_loc_array = meshgrid(x_loc_list, y_loc_list)

		# Extract the directions at these points
		# Initialize the arrays
		x_dir_array = zeros((n_samples_y, n_samples_x), dtype = float)
		y_dir_array = zeros((n_samples_y, n_samples_x), dtype = float)
		# Get the directions
		for y_index in range(n_samples_y):
			for x_index in range(n_samples_x):
				# Get the row and column indices
				row_index = y_loc_array[y_index, x_index]
				col_index = x_loc_array[y_index, x_index]
				# Store the vector
				x_dir_array[y_index, x_index] = self._all_vectors_x[row_index, col_index]
				y_dir_array[y_index, x_index] = self._all_vectors_y[row_index, col_index]

		# Create the figure representing the vector field
		# Create the figure
		plt.figure(figsize = (10, 8), layout = "constrained")
		# Add the needed traces and set the title
		if plot_type == "quiver":
			plt.quiver(x_loc_array, y_loc_array, x_dir_array, y_dir_array)
			plt.title("Quiver Representation Of The Generated Vector Field")
		else:
			plt.streamplot(x_loc_array, y_loc_array, x_dir_array, y_dir_array)
			plt.title("Streamplot Representation Of The Generated Vector Field")
		# Format the figure
		plt.xlabel("columm index")
		plt.ylabel("row index")

		# Show the figure (if needed)
		if show_flag == True:
			plt.show()

		# Save the figure (if needed)
		if save_flag == True:
			# Get a path to which the figure should be saved and make sure cancel wasn't clicked
			figure_path = askSaveFilename(allowed_extensions = ["png"])
			assert figure_path is not None, "VectorField::plotVectorField: Unable to save figure because cancel button was clicked"
			# Save the figure to this location
			plt.savefig(figure_path)

	### Define functions for computing and displaying curl and divergence of the vector field ###
	def computeDerivativeInfo(self):
		# Compute all derivative-related values for the vector field
		# Mark that needed information has not been handled
		self._derivatives_computed_flag = False

		# Define the tuple of shared values needed by all workers
		shared_info = ((self._all_vectors_x,
						self._all_vectors_y,
						self._n_cols,
						self._n_rows),)

		# Create the list of inputs used for each step in the process
		all_inputs = []
		for row_index in range(self._n_rows):
			for col_index in range(self._n_cols):
				all_inputs.append((row_index, col_index))

		# Initialize a pool with the needed number of processes, run the computation in parallel, and end by closing the pool
		pool = Pool(processes = max(cpu_count() - 1, 1), initializer = _initializeComputeDerivatives, initargs = shared_info)
		all_outputs = pool.map(_proxyComputeDerivatives, all_inputs)
		pool.close()

		# Compute the vector associated with each point
		self._all_curls = zeros((self._n_rows, self._n_cols), dtype = float)
		self._all_divergences = zeros((self._n_rows, self._n_cols), dtype = float)
		self._all_jacobians = zeros((self._n_rows, self._n_cols), dtype = float)
		for index in range(self._n_rows * self._n_cols):
			# Get the corresponding inputs and outputs
			input_pair = all_inputs[index]
			output_tuple = all_outputs[index]
			# Store the results
			self._all_curls[input_pair[0], input_pair[1]] = output_tuple[0]
			self._all_divergences[input_pair[0], input_pair[1]] = output_tuple[1]
			self._all_jacobians[input_pair[0], input_pair[1]] = output_tuple[2]

		# Mark that the curl and divergence have been computed
		self._derivatives_computed_flag = True

	def renderDerivativeInfo(self, curl_flag:bool, divergence_flag:bool, jacobian_flag:bool, circle_flag:bool = False, show_flag:bool = True, save_flag:bool = False) -> Image.Image:
		# Render the needed combination of derivative-related values of the vector field, return the resulting PIL image object
		# Only proceed if derivative-related values have been computed
		assert self._derivatives_computed_flag == True, "VectorField::renderDerivativeInfo: Only able to plot derivative-related values once they have been generated"

		# Verify the inputs
		# Derivative types
		assert type(curl_flag) == bool, "VectorField::renderDerivativeInfo: Provided value for 'curl_flag' must be a bool object"
		assert type(divergence_flag) == bool, "VectorField::rednerDerivativeInfo: Provided value for 'divergence_flag' must be a bool object"
		assert type(jacobian_flag) == bool, "VectorField::renderDerivativeInfo: Provided value for 'jacobian_flag' must be a bool object"
		assert (curl_flag == True) or (divergence_flag == True) or (jacobian_flag == True), "VectorField::renderDerivativeInfo: At least one of the provided values for 'curl_flag', 'divergence_flag' and 'jacobian_flag' must be True"
		# Circle flag
		assert type(circle_flag) == bool, "VectorField::renderDerivativeInfo: Provided value for 'circle_flag' must be a bool object"
		# Show/save flags
		assert type(show_flag) == bool, "VectorField::renderDerivativeInfo: Provided value for 'show_flag' must be a bool object"
		assert type(save_flag) == bool, "VectorField::renderDerivativeInfo: Provided value for 'save_flag' must be a bool object"
		assert show_flag == True or save_flag == True, "VectorField::renderDerivativeInfo: At least of the provided values for 'show_flag' and 'save_flag' must be True"

		# Process the needed derivative-related values for the needed types
		# Initialize the storage
		rgb_array_per_type = {}
		# Loop over the derivative types
		for derivative_type in ["curl", "divergence", "jacobian"]:
			# Determine if this analysis should proceed
			if derivative_type == "curl" and curl_flag == True:
				proceed_flag = True
			elif derivative_type == "divergence" and divergence_flag == True:
				proceed_flag = True
			elif derivative_type == "jacobian" and jacobian_flag == True:
				proceed_flag = True
			else:
				proceed_flag = False

			# Proceed with the analysis (if needed)
			if proceed_flag == True:
				# Extract the currently needed derivative values to use
				if derivative_type == "curl":
					current_derivative_values = self._all_curls
				elif derivative_type == "divergence":
					current_derivative_values = self._all_divergences
				else:
					current_derivative_values = self._all_jacobians

				# Compute the maximum and minimum values of this derivative type
				max_derivative = np_max(current_derivative_values)
				min_derivative = np_min(current_derivative_values)

				# Compute the maximum magnitude value of this derivative type
				max_magnitude_derivative = max(abs(max_derivative), abs(min_derivative))

				# Initialize the RGB array used for the image
				rgb_array_per_type[derivative_type] = zeros((self._n_rows, self._n_cols, 3), dtype = float)

				# Compute the raw RGB values associated with each pixel
				for row_index in range(self._n_rows):
					for col_index in range(self._n_cols):
						# Get the derivative-related value for this point
						derivative_value = current_derivative_values[row_index, col_index]

						# Compute the RGB values for the image
						if derivative_value > 0:
							if max_magnitude_derivative > 0:
								rgb_array_per_type[derivative_type][row_index, col_index, 2] = int(255 * derivative_value / max_magnitude_derivative)
						elif derivative_value < 0:
							rgb_array_per_type[derivative_type][row_index, col_index, 0] = int(-255 * derivative_value / max_magnitude_derivative)

		# Process the above into a single RGB array depending on how many were selected
		if len(rgb_array_per_type) == 1:
			# Only a single derivative type was used, set that as the RGB array
			derivative_rgb_array = list(rgb_array_per_type.values())[0]
		else:
			# Multiple derivative types were used, extract the needed values and combine them using PCA
			# Initialize the raw data array needed for PCA
			raw_data_array = zeros((self._n_rows * self._n_cols, 2 * len(rgb_array_per_type)), dtype = float)

			# Extract the needed values from the RGB arrays
			counter = 0
			for derivative_type in rgb_array_per_type:
				# Store the RGB values in the needed channels
				raw_data_array[:, counter] = rgb_array_per_type[derivative_type][:, :, 2].reshape(self._n_rows * self._n_cols)
				raw_data_array[:, counter + 1] = rgb_array_per_type[derivative_type][:, :, 0].reshape(self._n_rows * self._n_cols)

				# Iterate to the next counter
				counter += 2

			# Perform PCA on the raw data and extract the needed results
			pca_results = performPCA(raw_data_array = raw_data_array)
			projected_data_array = pca_results["outputs"]["projected_data_array"]

			# Reshape the projected array to match the original shape
			reshaped_projected_data_array = zeros((self._n_rows, self._n_cols, 3), dtype = float)
			reshaped_projected_data_array[:, :, 0] = projected_data_array[:, 0].reshape(self._n_rows, self._n_cols)
			reshaped_projected_data_array[:, :, 1] = projected_data_array[:, 1].reshape(self._n_rows, self._n_cols)
			reshaped_projected_data_array[:, :, 2] = projected_data_array[:, 2].reshape(self._n_rows, self._n_cols)

			# Compute the maximum and minimum values of the projected data
			max_projected = np_max(reshaped_projected_data_array)
			min_projected = np_min(reshaped_projected_data_array)

			# Initialize the RGB array representing the normalized projected data
			derivative_rgb_array = zeros((self._n_rows, self._n_cols, 3), dtype = float)

			# Normalize the projected data as needed
			for row_index in range(self._n_rows):
				for col_index in range(self._n_cols):
					# Get the raw red, green and blue values
					raw_red_value = reshaped_projected_data_array[row_index, col_index, 0]
					raw_green_value = reshaped_projected_data_array[row_index, col_index, 1]
					raw_blue_value = reshaped_projected_data_array[row_index, col_index, 2]

					# Handle the various information channels
					if min_projected < max_projected:
						derivative_rgb_array[row_index, col_index, 0] = int(255 * (raw_red_value - min_projected) / (max_projected - min_projected))
						derivative_rgb_array[row_index, col_index, 1] = int(255 * (raw_green_value - min_projected) / (max_projected - min_projected))
						derivative_rgb_array[row_index, col_index, 2] = int(255 * (raw_blue_value - min_projected) / (max_projected - min_projected))
					else:
						derivative_rgb_array[row_index, col_index, 0] = 127
						derivative_rgb_array[row_index, col_index, 1] = 127
						derivative_rgb_array[row_index, col_index, 2] = 127

		# Draw white circles around the base vector locations (if needed)
		if circle_flag == True:
			# Set the circle's radius
			circle_radius = int(0.01 * min(self._n_rows, self._n_cols))

			# Loop over the base points and draw the circle at each
			for base_vector_index in range(self._n_base_vectors):
				# Get the row and column indices of the base vector
				base_row_index = self._all_base_vector_locations_row[base_vector_index]
				base_col_index = self._all_base_vector_locations_col[base_vector_index]

				# Loop over nearby indices and set to white (if needed)
				for row_index in range(base_row_index - circle_radius, base_row_index + circle_radius):
					if 0 <= row_index and row_index < self._n_rows:
						for col_index in range(base_col_index - circle_radius, base_col_index + circle_radius):
							if 0 <= col_index and col_index < self._n_cols:
								# Compute the current distance
								current_distance = sqrt((base_row_index - row_index)**2 + (base_col_index - col_index)**2)

								# Set to white or black if sufficiently close
								if current_distance <= circle_radius / 2:
									curl_rgb_array[row_index, col_index, :] = 255
								elif current_distance <= circle_radius:
									curl_rgb_array[row_index, col_index, :] = 0

		# Create the image from the RGB array
		derivative_image = Image.fromarray(derivative_rgb_array.astype(uint8), "RGB")

		# Show the image (if needed)
		if show_flag == True:
			derivative_image.show()

		# Save the image (if needed)
		if save_flag == True:
			# Get a path to which the image should be saved and make sure cancel wasn't clicked
			image_path = askSaveFilename(allowed_extensions = ["png"])
			assert image_path is not None, "VectorField::renderDerivativeInfo: Unable to save image because cancel button was clicked"
			# Save the image to this location
			derivative_image.save(image_path, "PNG")

		# Return the result
		return derivative_image


##################################################################
### Define a class which generates 2-dimensional vector fields ###
##################################################################
# Create the decorator needed for making the attributes private
vector_field_generator_decorator = privacyDecorator(["_all_base_vector_locations_col",		# class variables
											  		 "_all_base_vector_locations_row",
											  		 "_all_base_vectors_x",
											  		 "_all_base_vectors_y",
											  		 "_all_vectors_x",
											  		 "_all_vectors_y",
											  		 "_n_base_vectors",
											  		 "_n_cols",
											  		 "_n_rows",
											  		 "_overwrites",
											  		 "_seed",
											  		 "_softmax_normalizer",
											  		 "_computeRemainingVectors",			# internal functions
											  		 "_generateBaseVectors",
											  		 "_getValue"])

# Define the class with private attributes
@vector_field_generator_decorator
class VectorFieldGenerator:
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
		assert type(n_rows) == int, "VectorFieldGenerator::__init__: Provided value for 'n_rows' must be an int object"
		assert 100 <= n_rows and n_rows <= 5000, "VectorFieldGenerator::__init__: Provided value for 'n_rows' must be >= 100 and <= 5000"
		assert type(n_cols) == int, "VectorFieldGenerator::__init__: Provided value for 'n_cols' must be an int object"
		assert 100 <= n_cols and n_cols <= 5000, "VectorFieldGenerator::__init__: Provided value for 'n_cols' must be >= 100 and <= 5000"
		for key in kwargs:
			assert key in self._DEFAULT_VALUES, "VectorFieldGenerator::__init__: Provided keyword arguments must be keys in '_DEFAULT_VALUES'"

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
		assert type(n_base_vectors_min) in [float, int], "VectorFieldGenerator::__init__: Provided value for 'n_base_vectors_min' must be a float or int object"
		assert type(n_base_vectors_mean) in [float, int], "VectorFieldGenerator::__init__: Provided value for 'n_base_vectors_mean' must be a float or int object"
		assert 1 <= n_base_vectors_min and n_base_vectors_min < float("inf"), "VectorFieldGenerator::__init__: Provided value for 'n_base_vectors_min' must be >= 1 and finite"
		assert n_base_vectors_min <= n_base_vectors_mean and n_base_vectors_mean < float("inf"), "VectorFieldGenerator::__init__: Provided value for 'n_base_vectors_mean' must be >= 'n_base_vectors_min' and finite"
		# Minimum and maximum base vector lengths
		assert type(base_vector_length_min) in [float, int], "VectorFieldGenerator::__init__: Provided value for 'base_vector_length_min' must be a float or int object"
		assert type(base_vector_length_max) in [float, int], "VectorFieldGenerator::__init__: Provided value for 'base_vector_length_max' must be a float or int object"
		assert 0 < base_vector_length_min and base_vector_length_min < float("inf"), "VectorFieldGenerator::__init__: Provided value for 'base_vector_length_min' must be positive and finite"
		assert base_vector_length_min <= base_vector_length_max and base_vector_length_max < float("inf"), "VectorFieldGenerator::__init__: Provided value for 'base_vector_length_max' must be >= 'base_vector_length_min' and finite"

		# Initialize additional needed storage
		self._all_base_vector_locations_col = None
		self._all_base_vector_locations_row = None
		self._all_base_vectors_x = None
		self._all_base_vectors_y = None
		self._all_vectors_x = None
		self._all_vectors_y = None
		self._n_base_vectors = None
		self._seed = None
		self._softmax_normalizer = None

	### Define internal function for combining default and overwrite values ###
	def _getValue(self, key:str) -> Any:
		# Combine the default and overwrite values
		# Verify the inputs
		assert type(key) == str, "VectorFieldGenerator::_getValue: Provided value for 'key' must be a str object"
		assert key in self._DEFAULT_VALUES, "VectorFieldGenerator::_getValue: Provided value for 'key' must appear in '_DEFAULT_VALUES'"

		# Return the result
		return self._overwrites.get(key, self._DEFAULT_VALUES[key])

	### Define internal functions needed for generating components of the vector field ###
	def _generateBaseVectors(self, seed:int = None):
		# Generate the base vectors of the vector field (using the given seed if needed)
		# Set the random seed (if needed)
		self._seed = seed
		if seed is not None:
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

	def _computeRemainingVectors(self, softmax_normalizer:Any):
		# Generate all other vectors in the vector field
		# Store the provided softmax normalizer value
		self._softmax_normalizer = softmax_normalizer

		# Define the tuple of shared values needed by all workers
		shared_info = ((self._all_base_vector_locations_col,
						self._all_base_vector_locations_row,
						self._all_base_vectors_x,
						self._all_base_vectors_y,
						self._n_base_vectors,
						self._n_cols,
						self._n_rows,
						self._softmax_normalizer),)

		# Create the list of inputs used for each step in the process
		all_inputs = []
		for row_index in range(self._n_rows):
			for col_index in range(self._n_cols):
				all_inputs.append((row_index, col_index))

		# Initialize a pool with the needed number of processes, run the computation in parallel, and end by closing the pool
		pool = Pool(processes = max(cpu_count() - 1, 1), initializer = _initializeComputeVector, initargs = shared_info)
		all_outputs = pool.map(_proxyComputeVector, all_inputs)
		pool.close()

		# Compute the vector associated with each point
		self._all_vectors_x = zeros((self._n_rows, self._n_cols), dtype = float)
		self._all_vectors_y = zeros((self._n_rows, self._n_cols), dtype = float)
		for index in range(self._n_rows * self._n_cols):
			input_pair = all_inputs[index]
			output_pair = all_outputs[index]
			self._all_vectors_x[input_pair[0], input_pair[1]] = output_pair[0]
			self._all_vectors_y[input_pair[0], input_pair[1]] = output_pair[1]

	### Define the external function needed for generating the vector field object ###
	def generate(self, softmax_normalizer:Any, seed:int = None) -> VectorField:
		# Return a VectorField object generated by randomly creating base vectors and interpolating as needed
		# Verify the inputs
		# Seed value
		if seed is not None:
			assert type(seed) == int, "VectorFieldGenerator::generate: If provided, value for 'seed' must be an int object"
			assert 0 <= seed and seed < 2**32, "VectorFieldGenerator::generate: If provided, value for 'seed' must be >= 0 and < 2^32"
		# Softmax normalizer
		assert isNumeric(softmax_normalizer, include_numpy_flag = False) == True, "VectorFieldGenerator::generate: Provided value for 'softmax_normalizer' must be a float or int object"
		assert 0 < softmax_normalizer and softmax_normalizer < float("inf"), "VectorFieldGenerator::generate: Provided value for 'softmax_normalizer' must be positive and finite"

		# Regenerate the base vectors and recompute the interpolated vectors (if needed)
		if seed is None or seed != self._seed:
			# Random seed differs from the last one, regenerate base vectors and mark that the remaining vectors must be computed
			self._generateBaseVectors(seed = seed)
			self._computeRemainingVectors(softmax_normalizer = softmax_normalizer)
		elif softmax_normalizer != self._softmax_normalizer:
			# Random seed matches the last one but softmax normalizer is different, mark that the remaining vectors must be computed
			self._computeRemainingVectors(softmax_normalizer = softmax_normalizer)

		# Return the vector field which uses the generated values
		return VectorField(n_rows = self._n_rows, n_cols = self._n_cols, all_vectors_x = self._all_vectors_x, all_vectors_y = self._all_vectors_y)