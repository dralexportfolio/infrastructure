##########################################
### Import needed general dependencies ###
##########################################
# Internal modules
from tkinter_helper import askSaveFilename
from type_helper import isListWithNumericEntries, isNumeric

# External modules
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from numpy import matmul, zeros
from numpy.linalg import inv
import plotly.graph_objects as go
from PrivateAttributesDecorator import private_attributes_dec
from typing import Any, Tuple


###########################################################
### Define the private attributes needed for each class ###
###########################################################
# Abstract spline class
SPLINE_PRIVATE_ATTRIBUTES = ["_n_points",								# class variables
							 "_x_values",
							 "_y_values",
							 "_findIndex"]								# private functions
							 
# Additional for linear spline class
LINEAR_SPLINE_PRIVATE_ATTRIBUTES = ["_base_x_value_per_index",			# class variables
									"_base_y_value_per_index",
									"_slope_per_index"]
									
# Additional for natural cubic spline class
NATURAL_CUBIC_SPLINE_PRIVATE_ATTRIBUTES = ["_base_x_value_per_index",	# class variables
									       "_base_y_value_per_index",
									       "_linear_coefficient_per_index",
									       "_quadratic_coefficient_per_index",
									       "_cubic_coefficient_per_index"]


########################################
### Define the abstract spline class ###
########################################
# Create the decorator needed for making the attributes private
spline_decorator = private_attributes_dec(*SPLINE_PRIVATE_ATTRIBUTES)	

# Define the class with private attributes
@spline_decorator
class Spline(ABC):
	### Initialize the class ###
	def __init__(self, x_values:list, y_values:list):
		# Verify the inputs
		assert isListWithNumericEntries(x_values, include_numpy_flag = False) == True, "Spline::__init__: Provided value for 'x_values' must be a list of numeric entries"
		assert isListWithNumericEntries(y_values, include_numpy_flag = False) == True, "Spline::__init__: Provided value for 'y_values' must be a list of numeric entries"
		assert len(x_values) > 0, "Spline::__init__: Provided value for 'x_values' must be a non-empty list"
		assert len(y_values) > 0, "Spline::__init__: Provided value for 'y_values' must be a non-empty list"
		assert len(x_values) == len(y_values), "Spline::__init__: Provided values for 'x_values' and 'y_values' must be of the same length"
		
		# Extract the number of points defining the spline
		self._n_points = len(x_values)
		
		# Verify that all entries in the list of x-values are in strictly increasing order
		for index in range(self._n_points - 1):
			assert x_values[index] < x_values[index + 1], "Spline::__init__: Provided value for 'x_values' must be a list in strictly increasing order"
			
		# Verify that the x-values are in the needed range
		assert x_values[0] > -float("inf"), "Spline::__init__: Provided value for 'x_values' must have first (i.e. smallest) value greater than negative infinity"
		assert x_values[-1] < float("inf"), "Spline::__init__: Provided value for 'x_values' must have last (i.e. largest) value less than infinity"
		
		# Verify that the y-values are in the needed range
		for index in range(self._n_points):
			assert -float("inf") < y_values[index] and y_values[index] < float("inf"), "Spline::__init__: Provided value for 'y_values' must have all entries finite"
		
		# Store the provided values
		self._x_values = x_values
		self._y_values = y_values
		
	### Define needed concrete methods ###
	def _findIndex(self, x_value:Any) -> int:
		# Return the index of the stored point to the immediate right of the provided x-value
		# Verify the inputs
		assert isNumeric(x_value, include_numpy_flag = False) == True, "Spline::_findIndex: Provided value for 'x_value' must be numeric"
		assert -float("inf") < x_value and x_value < float("inf"), "Spline::_findIndex: Provided value for 'x_value' must be finite"
		
		# Handle the various cases
		if x_value <= self._x_values[0]:
			# To the left of the left-most point, return 0
			return 0
		elif x_value > self._x_values[-1]:
			# To the right of the right-most point, return 1 plus the index of the right-most point
			return self._n_points
		else:
			# Between the left-most and right-most points, search for the needed index and return it
			for region_index in range(1, self._n_points):
				if self._x_values[region_index - 1] < x_value and x_value <= self._x_values[region_index]:
					return region_index
					
	def getPlotInfo(self, x_lower:Any = None, x_upper:Any = None) -> dict:
		# Return a dictionary of information required for plotting this spline
		# Verify the inputs
		assert x_lower is None or isNumeric(x_lower, include_numpy_flag = False) == True, "Spline::getPlotInfo: If provided, value for 'x_lower' must be a numeric"
		assert x_upper is None or isNumeric(x_upper, include_numpy_flag = False) == True, "Spline::getPlotInfo: If provided, value for 'x_upper' must be a numeric"
		
		# Set lower and upper x-values (if needed)
		if x_lower is None:
			x_lower = self._x_values[0]
		if x_upper is None:
			x_upper = self._x_values[-1]
			
		# Make sure the lower bound is less than the upper bound
		assert x_lower < x_upper, "Spline::getPlotInfo: Provided values must result in a lower x-value which is less than the upper x-value"
		
		# Generate the x-values and y-values to plot
		x_values_interpolated = [x_lower + (x_upper - x_lower) * index / 1000 for index in range(1001)]
		y_values_interpolated = [self.evaluate(x_value = x_value) for x_value in x_values_interpolated]
		
		# Create the dictionary of results to return
		plot_results = {}
		plot_results["x_values_original"] = self._x_values
		plot_results["y_values_original"] = self._y_values
		plot_results["x_values_interpolated"] = x_values_interpolated
		plot_results["y_values_interpolated"] = y_values_interpolated
		plot_results["x_lower"] = x_lower
		plot_results["x_upper"] = x_upper
		
		# Return the results
		return plot_results
					
	def plot(self, x_lower:Any = None, x_upper:Any = None, save_flag:bool = False, show_flag:bool = True, used_engine:str = "matplotlib"):
		# Save and/or show a plot representing the current spline information
		# Verify the inputs
		assert type(save_flag) == bool, "Spline::plot: Provided value for 'save_flag' must be bool object"
		assert type(show_flag) == bool, "Spline::plot: Provided value for 'show_flag' must be bool object"
		assert save_flag == True or show_flag == True, "Spline::plot: At least one of the provided values for 'save_flag' and 'show_flag' must be True"
		assert used_engine in ["matplotlib", "plotly"], "Spline::plot: Provided value for 'used_engine' must be 'matplotlib' or 'plotly'"
		
		# Get the filename to save to, raise error if not selected (if needed)
		if save_flag == True:
			# Get the needed path
			if used_engine == "matplotlib":
				filename_path = askSaveFilename(allowed_extensions = ["png"])
			else:
				filename_path = askSaveFilename(allowed_extensions = ["html"])
				
			# End if not selected
			assert filename_path is not None, "Spline::plot: Unable to proceed with saving plot because no filename was seleceted"
		
		# Generate the needed plot information and extract the values
		plot_results = self.getPlotInfo(x_lower = x_lower, x_upper = x_upper)
		x_values_original = plot_results["x_values_original"]
		y_values_original = plot_results["y_values_original"]
		x_values_interpolated = plot_results["x_values_interpolated"]
		y_values_interpolated = plot_results["y_values_interpolated"]
		x_lower = plot_results["x_lower"]
		x_upper = plot_results["x_upper"]
		
		# Set the plot title and axis labels
		plot_title = "Visualization Of Spline From X = " + str(x_lower) + " To " + str(x_upper)
		x_label = "x-value"
		y_label = "y-value"
		
		# Save the needed plot
		if used_engine == "matplotlib":
			# Create the figure
			plt.figure()
			# Add the needed traces
			plt.plot(x_values_interpolated, y_values_interpolated, "b-")
			plt.scatter(x_values_original, y_values_original, None, "b")
			# Format the figure
			plt.title(plot_title)
			plt.xlim(left = x_lower, right = x_upper)
			plt.xlabel(x_label)
			plt.ylabel(y_label)
			# Save the figure (if needed)
			if save_flag == True:
				plt.savefig(filename_path)
			# Show the figure
			if show_flag == True:
				plt.show()
		else:
			# Create the figure
			fig = go.Figure()
			# Add the needed traces
			fig.add_trace(go.Scatter(x = x_values_interpolated, y = y_values_interpolated, showlegend = False, mode = "lines", line = {"color": "blue"}))
			fig.add_trace(go.Scatter(x = x_values_original, y = y_values_original, showlegend = False, mode = "markers", line = {"color": "blue"}))
			# Format the figure
			fig.update_layout(title = plot_title, xaxis_range = [x_lower, x_upper])
			fig.update_xaxes(title = x_label)
			fig.update_yaxes(title = y_label)
			# Save the figure (if needed)
			if save_flag == True:
				fig.write_html(filename_path)
			# Show the figure
			if show_flag == True:
				fig.show()
		
	### Define needed abstract methods ###
	@abstractmethod
	def evaluate(self, x_value:Any) -> Any:
		# Determine the y-value of the spline at the provided x-value (note: abstract version verifies inputs and computes region index)
		# Verify the inputs
		assert isNumeric(x_value, include_numpy_flag = False) == True, "Spline::evaluate: Provided value for 'x_value' must be numeric"
		assert -float("inf") < x_value and x_value < float("inf"), "Spline::evaluate: Provided value for 'x_value' must be finite"
		
		# Determine the index needed for the given point
		region_index = self._findIndex(x_value = x_value)
		
		# Return the result
		return region_index

	### Define a deepcopy function compatible with PrivateAttributesDecorator ###
	@abstractmethod
	def deepcopy(self):
		# Create a copy of this object and return it
		pass
		
		
###############################################
### Define the concrete linear spline class ###
###############################################
# Create the decorator needed for making the attributes private
linear_spline_decorator = private_attributes_dec(*(SPLINE_PRIVATE_ATTRIBUTES + LINEAR_SPLINE_PRIVATE_ATTRIBUTES))

# Define the class with private attributes
@linear_spline_decorator
class LinearSpline(Spline):
	### Initialize the class ###
	def __init__(self, x_values:list, y_values:list):
		# Initialize via the super-class
		super().__init__(x_values = x_values, y_values = y_values)
		
		# Make sure there are at least 2 points defining this spline
		assert self._n_points >= 2, "LinearSpline::__init__: At least 2 points must be provided in order to define a linear spline"
		
		# Pre-compute the base values and slope needed for each index section
		self._base_x_value_per_index = {}
		self._base_y_value_per_index = {}
		self._slope_per_index = {}
		for region_index in range(self._n_points + 1):
			if region_index <= 1:
				self._base_x_value_per_index[region_index] = self._x_values[0]
				self._base_y_value_per_index[region_index] = self._y_values[0]
				self._slope_per_index[region_index] = (self._y_values[1] - self._y_values[0]) / (self._x_values[1] - self._x_values[0])
			elif region_index >= self._n_points - 1:
				self._base_x_value_per_index[region_index] = self._x_values[-2]
				self._base_y_value_per_index[region_index] = self._y_values[-2]
				self._slope_per_index[region_index] = (self._y_values[-1] - self._y_values[-2]) / (self._x_values[-1] - self._x_values[-2])
			else:
				self._base_x_value_per_index[region_index] = self._x_values[region_index - 1]
				self._base_y_value_per_index[region_index] = self._y_values[region_index - 1]
				self._slope_per_index[region_index] = (self._y_values[region_index] - self._y_values[region_index - 1]) / (self._x_values[region_index] - self._x_values[region_index - 1])

	### Define concrete versions of the abstract super-class methods ###
	def evaluate(self, x_value:Any) -> Any:
		# Determine the y-value of the spline at the provided x-value
		# Get the needed index via the super-class
		region_index = super().evaluate(x_value = x_value)
		
		# Compute the interpolated y-value
		y_value = self._base_y_value_per_index[region_index] + self._slope_per_index[region_index] * (x_value - self._base_x_value_per_index[region_index])
		
		# Return the result
		return y_value

	### Define a deepcopy function compatible with PrivateAttributesDecorator ###
	def deepcopy(self):
		# Create a copy of this object and return it
		# Initialize a Spline object with the same inputs
		copy_of_self = type(self)(x_values = deepcopy(self._x_values),
								  y_values = deepcopy(self._y_values))

		# Return the copied object
		return copy_of_self
		
		
######################################################
### Define the concrete natural cubic spline class ###
######################################################
# Create the decorator needed for making the attributes private
natural_cubic_spline_decorator = private_attributes_dec(*(SPLINE_PRIVATE_ATTRIBUTES + NATURAL_CUBIC_SPLINE_PRIVATE_ATTRIBUTES))

# Define the class with private attributes
@natural_cubic_spline_decorator
class NaturalCubicSpline(Spline):
	### Initialize the class ###
	def __init__(self, x_values:list, y_values:list):
		# Initialize via the super-class
		super().__init__(x_values = x_values, y_values = y_values)

		# Make sure there are at least 3 points defining this spline
		assert self._n_points >= 3, "NaturalCubicSpline::__init__: At least 3 points must be provided in order to define a natural cubic spline"

		# Compute the successive differences of the x-values and y-values
		x_differences = [self._x_values[index + 1] - self._x_values[index] for index in range(self._n_points - 1)]
		y_differences = [self._y_values[index + 1] - self._y_values[index] for index in range(self._n_points - 1)]

		# Define the augmented matrix and vector needed for computing the augmented quadratic coefficients
		# Initialize the needed storage
		needed_matrix = zeros((self._n_points, self._n_points), dtype = float)
		needed_vector = zeros((self._n_points, 1), dtype = float)
		# Set the top and bottom rows of the matrix
		needed_matrix[0, 0] = 1
		needed_matrix[self._n_points - 1, self._n_points - 1] = 1
		# Set all entries in the other rows
		for index in range(1, self._n_points - 1):
			# Handle the 3 entries of the matrix
			needed_matrix[index, index - 1] = x_differences[index - 1]
			needed_matrix[index, index] = 2 * (x_differences[index - 1] + x_differences[index])
			needed_matrix[index, index + 1] = x_differences[index]
			# Handle the only entry of the vector
			needed_vector[index, 0] += 3 * y_differences[index] / x_differences[index]
			needed_vector[index, 0] -= 3 * y_differences[index - 1] / x_differences[index - 1]

		# Solve this system to get the augmented quadratic coefficients
		needed_solution = matmul(inv(needed_matrix), needed_vector)
		augmented_quadratic_coefficients = []
		for index in range(self._n_points):
			augmented_quadratic_coefficients.append(float(needed_solution[index, 0]))

		# Compute the corresponding cubic coefficients
		cubic_coefficients = []
		for index in range(self._n_points - 1):
			numerator = augmented_quadratic_coefficients[index + 1] - augmented_quadratic_coefficients[index]
			denominator = 3 * x_differences[index]
			cubic_coefficients.append(numerator / denominator)

		# Compute the corresponding linear coefficients
		linear_coefficients = []
		for index in range(self._n_points - 1):
			term_1 = y_differences[index] / x_differences[index]
			term_2 = (2 * augmented_quadratic_coefficients[index] + augmented_quadratic_coefficients[index + 1]) * x_differences[index] / 3
			linear_coefficients.append(term_1 - term_2)

		# Pre-compute the base values and slope needed for each index section
		self._base_x_value_per_index = {}
		self._base_y_value_per_index = {}
		self._linear_coefficient_per_index = {}
		self._quadratic_coefficient_per_index = {}
		self._cubic_coefficient_per_index = {}
		for region_index in range(self._n_points + 1):
			if region_index <= 1:
				self._base_x_value_per_index[region_index] = self._x_values[0]
				self._base_y_value_per_index[region_index] = self._y_values[0]
				self._linear_coefficient_per_index[region_index] = linear_coefficients[0]
				self._quadratic_coefficient_per_index[region_index] = augmented_quadratic_coefficients[0]
				self._cubic_coefficient_per_index[region_index] = cubic_coefficients[0]
			elif region_index >= self._n_points - 1:
				self._base_x_value_per_index[region_index] = self._x_values[-2]
				self._base_y_value_per_index[region_index] = self._y_values[-2]
				self._linear_coefficient_per_index[region_index] = linear_coefficients[-1]
				self._quadratic_coefficient_per_index[region_index] = augmented_quadratic_coefficients[-2]
				self._cubic_coefficient_per_index[region_index] = cubic_coefficients[-1]
			else:
				self._base_x_value_per_index[region_index] = self._x_values[region_index - 1]
				self._base_y_value_per_index[region_index] = self._y_values[region_index - 1]
				self._linear_coefficient_per_index[region_index] = linear_coefficients[region_index - 1]
				self._quadratic_coefficient_per_index[region_index] = augmented_quadratic_coefficients[region_index - 1]
				self._cubic_coefficient_per_index[region_index] = cubic_coefficients[region_index - 1]

	### Define concrete versions of the abstract super-class methods ###
	def evaluate(self, x_value:Any) -> Any:
		# Determine the y-value of the spline at the provided x-value
		# Get the needed index via the super-class
		region_index = super().evaluate(x_value = x_value)

		# Compute the interpolated y-value
		y_value = self._base_y_value_per_index[region_index]
		y_value += self._linear_coefficient_per_index[region_index] * (x_value - self._base_x_value_per_index[region_index])
		y_value += self._quadratic_coefficient_per_index[region_index] * (x_value - self._base_x_value_per_index[region_index])**2
		y_value += self._cubic_coefficient_per_index[region_index] * (x_value - self._base_x_value_per_index[region_index])**3

		# Return the result
		return y_value

	### Define a deepcopy function compatible with PrivateAttributesDecorator ###
	def deepcopy(self):
		# Create a copy of this object and return it
		# Initialize a Spline object with the same inputs
		copy_of_self = type(self)(x_values = deepcopy(self._x_values),
								  y_values = deepcopy(self._y_values))

		# Return the copied object
		return copy_of_self