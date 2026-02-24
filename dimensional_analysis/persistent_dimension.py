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

# Internal modules
from color_helper import customSpectrum
from dimension_reduction import performPCA
from sqlite3_helper import addTable, appendRow, getColumnNames, getColumnTypes, getExistingTables, getRowCount, readColumn, readRow, replaceRow
from tkinter_helper import askSaveFilename
from type_helper import isNumeric

# External modules
from os import remove
from os.path import exists
from math import sqrt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numpy import array, cumsum, mean, ndarray, zeros
from numpy.linalg import norm
from pathlib import PosixPath, WindowsPath
import plotly.graph_objects as go
from scipy.special import softmax
from typing import Any, Union


##################################################################################################
### Define all constants and functions needed for generating and verifying dimension databases ###
##################################################################################################
# Define the table names needed for a dimension database along with the relevant column names and types
# Input settings table
TABLE_NAME_INPUT_SETTINGS = "input_settings"
COLUMN_NAMES_INPUT_SETTINGS = ["n_rows", "n_cols", "min_softmax_distance", "max_softmax_distance", "n_distances"]
COLUMN_TYPES_INPUT_SETTINGS = ["BIGINT", "BIGINT", "REAL", "REAL", "BIGINT"]
# Distances used table
TABLE_NAME_DISTANCES_USED = "distances_used"
COLUMN_NAMES_DISTANCES_USED = ["table_name", "softmax_distance"]
COLUMN_TYPES_DISTANCES_USED = ["TEXT", "REAL"]
# Raw data array table
TABLE_NAME_RAW_DATA_ARRAY = "raw_data_array"
COLUMN_NAMES_FUNCTION_RAW_DATA_ARRAY = lambda n_cols: ["parameter_" + str(index + 1) for index in range(n_cols)]
COLUMN_TYPES_FUNCTION_RAW_DATA_ARRAY = lambda n_cols: ["REAL" for _ in range(n_cols)]
# Projected data array table
TABLE_NAME_PROJECTED_DATA_ARRAY = "projected_data_array"
COLUMN_NAMES_FUNCTION_PROJECTED_DATA_ARRAY = lambda n_cols: ["parameter_" + str(index + 1) for index in range(n_cols)]
COLUMN_TYPES_FUNCTION_PROJECTED_DATA_ARRAY = lambda n_cols: ["REAL" for _ in range(n_cols)]
# Cumulative percent variances tables
TABLE_NAME_FUNCTION_CUMULATIVE_PERCENT_VARIANCES = lambda distance_index: "cumulative_percent_variances_" + str(distance_index)
COLUMN_NAMES_FUNCTION_CUMULATIVE_PERCENT_VARIANCES = lambda n_cols: ["cumulative_" + str(index) for index in range(n_cols + 1)]
COLUMN_TYPES_FUNCTION_CUMULATIVE_PERCENT_VARIANCES = lambda n_cols: ["REAL" for _ in range(n_cols + 1)]

# Define the function for generating a dimension database (i.e. dimension estimates for each point as a function of percent variance and softmax distance
def generateDimensionDatabase(raw_data_array:ndarray, min_softmax_distance:Any, max_softmax_distance:Any, n_distances:int = 1) -> Union[PosixPath, WindowsPath]:
	# Use PCA to compute the information for estimating pointwise dimension and write it to a db file, return the used db file path
	# Verify the inputs
	assert type(raw_data_array) == ndarray, "generateDimensionDatabase: Provided value for 'raw_data_array' must be a numpy.ndarray object"
	assert len(raw_data_array.shape) == 2, "generateDimensionDatabase: Provided value for 'raw_data_array' must be a 2-dimensional numpy array"
	assert raw_data_array.shape[0] > 0, "generateDimensionDatabase: Provided value for 'raw_data_array' must have a non-zero number of points, i.e. at least 1 row"
	assert raw_data_array.shape[1] > 0, "generateDimensionDatabase: Provided value for 'raw_data_array' must have a non-zero number of features, i.e. at least 1 column"
	assert isNumeric(min_softmax_distance, include_numpy_flag = True) == True, "generateDimensionDatabase: Provided value for 'min_softmax_distance' must be numeric"
	assert isNumeric(max_softmax_distance, include_numpy_flag = True) == True, "generateDimensionDatabase: Provided value for 'max_softmax_distance' must be numeric"
	assert 0 < min_softmax_distance and min_softmax_distance < float("inf"), "generateDimensionDatabase: Provided value for 'min_softmax_distance' must be positive and finite"
	assert 0 < max_softmax_distance and max_softmax_distance < float("inf"), "generateDimensionDatabase: Provided value for 'max_softmax_distance' must be positive and finite"
	assert type(n_distances) == int, "generateDimensionDatabase: Provided value for 'n_distances' must be an int object"
	assert n_distances > 0, "generateDimensionDatabase: Provided value for 'n_distances' must be positive"

	# Verify that the minimum and maximum softmax distances are valid given the choice for number of softmax distances
	if n_distances == 1:
		assert min_softmax_distance == max_softmax_distance, "generateDimensionDatabase: If provided value for 'n_distances' is equal to 1, value for 'min_softmax_distance' must be equal to value for 'max_softmax_distance'"
	else:
		assert min_softmax_distance < max_softmax_distance, "generateDimensionDatabase: If provided value for 'n_distances' is greater than 1, value for 'min_softmax_distance' must be less than value for 'max_softmax_distance'"

	# Compute the softmax distances to use and get the associated table names
	all_softmax_distances = [min_softmax_distance]
	all_cumulative_table_names = [TABLE_NAME_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(0)]
	for distance_index in range(1, n_distances):
		all_softmax_distances.append(min_softmax_distance + (max_softmax_distance - min_softmax_distance) * distance_index / (n_distances - 1))
		all_cumulative_table_names.append(TABLE_NAME_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(distance_index))

	# Get the db filename to save to and delete any existing file (if needed), raise error if not selected
	db_path = askSaveFilename(allowed_extensions = ["db"])
	if exists(db_path):
		remove(db_path)
	
	# Extract the number of rows and columns in the data
	n_rows = raw_data_array.shape[0]
	n_cols = raw_data_array.shape[1]
	
	# Create the needed db file and tables
	addTable(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, column_names = COLUMN_NAMES_INPUT_SETTINGS, column_types = COLUMN_TYPES_INPUT_SETTINGS)
	addTable(db_path = db_path, table_name = TABLE_NAME_DISTANCES_USED, column_names = COLUMN_NAMES_DISTANCES_USED, column_types = COLUMN_TYPES_DISTANCES_USED)
	addTable(db_path = db_path, table_name = TABLE_NAME_RAW_DATA_ARRAY, column_names = COLUMN_NAMES_FUNCTION_RAW_DATA_ARRAY(n_cols), column_types = COLUMN_TYPES_FUNCTION_RAW_DATA_ARRAY(n_cols))
	addTable(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY, column_names = COLUMN_NAMES_FUNCTION_PROJECTED_DATA_ARRAY(n_cols), column_types = COLUMN_TYPES_FUNCTION_PROJECTED_DATA_ARRAY(n_cols))
	for distance_index in range(n_distances):
		addTable(db_path = db_path, table_name = all_cumulative_table_names[distance_index], column_names = COLUMN_NAMES_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(n_cols), column_types = COLUMN_TYPES_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(n_cols))

	# Write the settings to the db file
	appendRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS)
	replaceRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0, new_row = [n_rows, n_cols, float(min_softmax_distance), float(max_softmax_distance), n_distances])

	# Write the softmax distances and associated table names to the db file
	for distance_index in range(n_distances):
		appendRow(db_path = db_path, table_name = TABLE_NAME_DISTANCES_USED)
		replaceRow(db_path = db_path, table_name = TABLE_NAME_DISTANCES_USED, row_index = distance_index, new_row = [all_cumulative_table_names[distance_index], float(all_softmax_distances[distance_index])])

	# Write the raw data array to the db file
	for row_index in range(n_rows):
		# Get the new row as float values
		new_row = [float(value) for value in raw_data_array[row_index, :]]
		# Write to the db file
		appendRow(db_path = db_path, table_name = TABLE_NAME_RAW_DATA_ARRAY)
		replaceRow(db_path = db_path, table_name = TABLE_NAME_RAW_DATA_ARRAY, row_index = row_index, new_row = new_row)

	# Perform PCA on the raw data array to get the projected data array used for plotting
	pca_results = performPCA(raw_data_array = raw_data_array, normalize_flag = False)
	projected_data_array = pca_results["outputs"]["projected_data_array"]

	# Write the projected data array to the db file
	for row_index in range(n_rows):
		# Get the new row as float values
		new_row = [float(value) for value in projected_data_array[row_index, :]]
		# Write to the db file
		appendRow(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY)
		replaceRow(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY, row_index = row_index, new_row = new_row)
	
	# Loop over the data points and compute the needed information
	for row_index in range(n_rows):
		# Set the center vector to be the current row
		center_vector = raw_data_array[row_index, :]

		# Compute the distances from the points to the center vector
		distance_array = zeros(n_rows, dtype = float)
		for other_row_index in range(n_rows):
			distance_array[other_row_index] = norm(raw_data_array[other_row_index, :] - center_vector)

		# Loop over the needed softmax distances
		for distance_index in range(n_distances):
			# Compute the weight vector using softmax on the distances
			weight_vector = softmax(-(distance_array / all_softmax_distances[distance_index])**2)

			# Compute the needed PCA results
			pca_results = performPCA(raw_data_array = raw_data_array, normalize_flag = False, center_vector = center_vector, weight_vector = weight_vector)

			# Compute the cumulative percent variances from these results (note: force first and last values to be 0 and 100 respectively)
			cumulative_percent_variances = [0.0] + [float(value) for value in cumsum(pca_results["outputs"]["ordered_percent_variances"])]
			cumulative_percent_variances[-1] = 100.0

			# Write this information to the db file
			appendRow(db_path = db_path, table_name = all_cumulative_table_names[distance_index])
			replaceRow(db_path = db_path, table_name = all_cumulative_table_names[distance_index], row_index = row_index, new_row = cumulative_percent_variances)
		
	# Return the path of the db file
	return db_path

# Define a function for verifying that a provided db file path represents a valid dimension database
def verifyDimensionDatabase(db_path:Union[PosixPath, WindowsPath]):
	# Verify that the provided db file path represents a valid dimension database (at least as far as table names, column names, column types and row counts)
	# Verify the inputs
	assert type(db_path) in [PosixPath, WindowsPath], "verifyDimensionDatabase: Provided value for 'db_path' must be a PosixPath or WindowsPath object"

	# Load the list of tables from the db file
	table_names = getExistingTables(db_path = db_path)

	# Make sure the input settings table exists and has the correct column names, column types and row count
	assert TABLE_NAME_INPUT_SETTINGS in table_names, "verifyDimensionDatabase: Provided value for 'db_path' must refer to a database with " + TABLE_NAME_INPUT_SETTINGS + " as a table name"
	assert getColumnNames(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS) == COLUMN_NAMES_INPUT_SETTINGS, "verifyDimensionDatabase: Table of name " + TABLE_NAME_INPUT_SETTINGS + " has the incorrect column names"
	assert getColumnTypes(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS) == COLUMN_TYPES_INPUT_SETTINGS, "verifyDimensionDatabase: Table of name " + TABLE_NAME_INPUT_SETTINGS + " has the incorrect column types"
	assert getRowCount(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS) == 1, "verifyDimensionDatabase: Table of name " + TABLE_NAME_INPUT_SETTINGS + " has the incorrect number or rows"

	# Load needed values from the input settings table
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]
	n_distances = read_row[4]

	# Make sure all other tables exist and have the correct column names, column types and row counts
	# Distances used table
	assert TABLE_NAME_DISTANCES_USED in table_names, "verifyDimensionDatabase: Provided value for 'db_path' must refer to a database with " + TABLE_NAME_DISTANCES_USED + " as a table name"
	assert getColumnNames(db_path = db_path, table_name = TABLE_NAME_DISTANCES_USED) == COLUMN_NAMES_DISTANCES_USED, "verifyDimensionDatabase: Table of name " + TABLE_NAME_DISTANCES_USED + " has the incorrect column names"
	assert getColumnTypes(db_path = db_path, table_name = TABLE_NAME_DISTANCES_USED) == COLUMN_TYPES_DISTANCES_USED, "verifyDimensionDatabase: Table of name " + TABLE_NAME_DISTANCES_USED + " has the incorrect column types"
	assert getRowCount(db_path = db_path, table_name = TABLE_NAME_DISTANCES_USED) == n_distances, "verifyDimensionDatabase: Table of name " + TABLE_NAME_DISTANCES_USED + " has the incorrect number or rows"
	# Raw data array table
	assert TABLE_NAME_RAW_DATA_ARRAY in table_names, "verifyDimensionDatabase: Provided value for 'db_path' must refer to a database with " + TABLE_NAME_RAW_DATA_ARRAY + " as a table name"
	assert getColumnNames(db_path = db_path, table_name = TABLE_NAME_RAW_DATA_ARRAY) == COLUMN_NAMES_FUNCTION_RAW_DATA_ARRAY(n_cols), "verifyDimensionDatabase: Table of name " + TABLE_NAME_RAW_DATA_ARRAY + " has the incorrect column names"
	assert getColumnTypes(db_path = db_path, table_name = TABLE_NAME_RAW_DATA_ARRAY) == COLUMN_TYPES_FUNCTION_RAW_DATA_ARRAY(n_cols), "verifyDimensionDatabase: Table of name " + TABLE_NAME_RAW_DATA_ARRAY + " has the incorrect column types"
	assert getRowCount(db_path = db_path, table_name = TABLE_NAME_RAW_DATA_ARRAY) == n_rows, "verifyDimensionDatabase: Table of name " + TABLE_NAME_RAW_DATA_ARRAY + " has the incorrect number or rows"
	# Projected data array table
	assert TABLE_NAME_PROJECTED_DATA_ARRAY in table_names, "verifyDimensionDatabase: Provided value for 'db_path' must refer to a database with " + TABLE_NAME_PROJECTED_DATA_ARRAY + " as a table name"
	assert getColumnNames(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY) == COLUMN_NAMES_FUNCTION_PROJECTED_DATA_ARRAY(n_cols), "verifyDimensionDatabase: Table of name " + TABLE_NAME_PROJECTED_DATA_ARRAY + " has the incorrect column names"
	assert getColumnTypes(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY) == COLUMN_TYPES_FUNCTION_PROJECTED_DATA_ARRAY(n_cols), "verifyDimensionDatabase: Table of name " + TABLE_NAME_PROJECTED_DATA_ARRAY + " has the incorrect column types"
	assert getRowCount(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY) == n_rows, "verifyDimensionDatabase: Table of name " + TABLE_NAME_PROJECTED_DATA_ARRAY + " has the incorrect number or rows"
	# Cumulative percent variances tables
	for distance_index in range(n_distances):
		current_table_name = TABLE_NAME_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(distance_index)
		assert current_table_name in table_names, "verifyDimensionDatabase: Provided value for 'db_path' must refer to a database with " + current_table_name + " as a table name"
		assert getColumnNames(db_path = db_path, table_name = current_table_name) == COLUMN_NAMES_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(n_cols), "verifyDimensionDatabase: Table of name " + current_table_name + " has the incorrect column names"
		assert getColumnTypes(db_path = db_path, table_name = current_table_name) == COLUMN_TYPES_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(n_cols), "verifyDimensionDatabase: Table of name " + current_table_name + " has the incorrect column types"
		assert getRowCount(db_path = db_path, table_name = current_table_name) == n_rows, "verifyDimensionDatabase: Table of name " + current_table_name + " has the incorrect number or rows"


#########################################################################################
### Define a function which estimates the local dimension of each point in a data set ###
#########################################################################################
def estimatePointwiseDimension(db_path:Union[PosixPath, WindowsPath], softmax_distance:Any, percent_variance:Any, needed_indices:list = None) -> dict:
	# Compute the pointwise dimension for each requested point the data stored in the pre-computed db file
	# Verify that the provided db file is a valid dimension database
	verifyDimensionDatabase(db_path = db_path)

	# Verify the other inputs
	assert isNumeric(softmax_distance, include_numpy_flag = True) == True, "estimatePointwiseDimension: Provided value for 'softmax_distance' must be numeric"
	assert isNumeric(percent_variance, include_numpy_flag = True) == True, "estimatePointwiseDimension: Provided value for 'percent_variance' must be numeric"
	assert 0 <= percent_variance and percent_variance <= 100, "estimatePointwiseDimension: Provided value for 'percent_variance' must be >= 0 and <= 100"
	if needed_indices is not None:
		assert type(needed_indices) == list, "estimatePointwiseDimension: If provided, value for 'needed_indices' must be a list object"
		assert len(needed_indices) > 0, "estimatePointwiseDimension: If provided, value for 'needed_indices' must be a non-empty list"
		for row_index in needed_indices:
			assert type(row_index) == int, "estimatePointwiseDimension: If provided, value for 'needed_indices' must be a list of int objects"

	# Get the relevant input settings from the db file
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]
	min_softmax_distance = read_row[2]
	max_softmax_distance = read_row[3]
	n_distances = read_row[4]

	# Handle additional verification of setting of the needed indices
	if needed_indices is None:
		# Needed indices not provided, compute for all points in the database
		needed_indices = range(n_rows)
	else:
		# Specific indices requested, make sure they are valid
		assert len(set(needed_indices)) == len(needed_indices), "estimatePointwiseDimension: If provided, value for 'needed_indices' must be a list of distinct entries"
		for row_index in needed_indices:
			assert 0 <= row_index and row_index < n_rows, "estimatePointwiseDimension: If provided, value for 'needed_indices' must be a list of non-negative integers less the number of rows from the database (in this case " + str(n_rows) + ")"

	# Verify that the softmax is valid
	assert min_softmax_distance <= softmax_distance and softmax_distance <= max_softmax_distance, "estimatePointwiseDimension: Provided value for 'softmax_distance' must be between minimum and maximum softmax distances stored in this db file"

	# Define an internal helper function for getting a dimension estimate at a given softmax distance index
	def getDimensionEstimate(softmax_index:int, row_index:int) -> float:
		# Load the cumulative percent variances for this data point
		all_percent_variances = readRow(db_path = db_path, table_name = TABLE_NAME_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(softmax_index), row_index = row_index)

		# Compute the estimated dimension by linearly interpolating
		for percent_index in range(n_cols):
			# Get the current percent variance bounds
			lower_percent_variance = all_percent_variances[percent_index]
			upper_percent_variance = all_percent_variances[percent_index + 1]

			# Proceed if the percent variance is in this range
			if lower_percent_variance <= percent_variance and percent_variance <= upper_percent_variance:
				# Compute the lower and upper percent variance weights
				lower_percent_weight = (upper_percent_variance - percent_variance) / (upper_percent_variance - lower_percent_variance)
				upper_percent_weight = (percent_variance - lower_percent_variance) / (upper_percent_variance - lower_percent_variance)

				# Return the needed dimension estimate and break
				return lower_percent_weight * percent_index + upper_percent_weight * (percent_index + 1)

	# Initialize the dictionary of results
	dimension_results = {}

	# Compute the dimension estimates for each point
	for row_index in needed_indices:
		if n_distances == 1:
			# Only a single softmax distance was used, do a single variable linear interpolation
			dimension_results[row_index] = getDimensionEstimate(softmax_index = 0, row_index = row_index)
		else:
			# Multiple softmax distances were used, do a double variable linear interpolation
			# Load the list of softmax distances used
			all_softmax_distances = readColumn(db_path = db_path, table_name = TABLE_NAME_DISTANCES_USED, column_name = "softmax_distance")

			# Search for the correct range of softmax
			for softmax_index in range(n_distances - 1):
				# Get the current softmax distance bounds
				lower_softmax_distance = all_softmax_distances[softmax_index]
				upper_softmax_distance = all_softmax_distances[softmax_index + 1]

				# Proceed if the softmax distance is in this range
				if lower_softmax_distance <= softmax_distance and softmax_distance <= upper_softmax_distance:
					# Compute the lower and upper softmax weights
					lower_softmax_weight = (upper_softmax_distance - softmax_distance) / (upper_softmax_distance - lower_softmax_distance)
					upper_softmax_weight = (softmax_distance - lower_softmax_distance) / (upper_softmax_distance - lower_softmax_distance)

					# Compute the dimension estimate using the lower and upper softmax distances
					lower_dimension_estimate = getDimensionEstimate(softmax_index = softmax_index, row_index = row_index)
					upper_dimension_estimate = getDimensionEstimate(softmax_index = softmax_index + 1, row_index = row_index)

					# Combine to get the needed dimension estimate
					dimension_results[row_index] = lower_softmax_weight * lower_dimension_estimate + upper_softmax_weight * upper_dimension_estimate
	
	# Return the results
	return dimension_results


##########################################################################
### Define functions for visualizing the pointwise dimension estimates ###
##########################################################################
def plotDimensionEstimateOfPoint(db_path:Union[PosixPath, WindowsPath], row_index:int, min_softmax_distance:Any, max_softmax_distance:Any, min_percent_variance:Any = 0,
								 max_percent_variance:Any = 100, n_samples:int = 100, used_engine:str = "matplotlib", round_flag:bool = False, show_flag:bool = True, save_flag:bool = False):
	# Generate a plot of the estimated dimension for the given point
	# Verify that the provided db file is a valid dimension database
	verifyDimensionDatabase(db_path = db_path)

	# Get the relevant input settings from the db file
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]
	n_distances = read_row[4]

	# Verify the other inputs
	# Row index (i.e. point to plot)
	assert type(row_index) == int, "plotDimensionEstimateOfPoint: Provided value for 'row_index' must be an int object"
	assert 0 <= row_index and row_index < n_rows, "plotDimensionEstimateOfPoint: Provided value for 'row_index' must be non-negative and less the number of rows from the database (in this case " + str(n_rows) + ")"
	# Softmax distance and percent variance bounds
	assert isNumeric(min_softmax_distance, include_numpy_flag = True) == True, "plotDimensionEstimateOfPoint: Provided value for 'min_softmax_distance' must be numeric"
	assert isNumeric(max_softmax_distance, include_numpy_flag = True) == True, "plotDimensionEstimateOfPoint: Provided value for 'max_softmax_distance' must be numeric"
	assert 0 < min_softmax_distance and min_softmax_distance < float("inf"), "plotDimensionEstimateOfPoint: Provided value for 'min_softmax_distance' must be positive and finite"
	assert 0 < max_softmax_distance and max_softmax_distance < float("inf"), "plotDimensionEstimateOfPoint: Provided value for 'max_softmax_distance' must be positive and finite"
	assert min_softmax_distance <= max_softmax_distance, "plotDimensionEstimateOfPoint: Provided value for 'min_softmax_distance' must be less than or equal to value for 'max_softmax_distance'"
	assert isNumeric(min_softmax_distance, include_numpy_flag = True) == True, "plotDimensionEstimateOfPoint: Provided value for 'min_softmax_distance' must be numeric"
	assert isNumeric(max_softmax_distance, include_numpy_flag = True) == True, "plotDimensionEstimateOfPoint: Provided value for 'max_softmax_distance' must be numeric"
	assert 0 <= min_percent_variance and min_percent_variance <= 100, "plotDimensionEstimateOfPoint: Provided value for 'min_percent_variance' must be >= 0 and <= 100"
	assert 0 <= max_percent_variance and max_percent_variance <= 100, "plotDimensionEstimateOfPoint: Provided value for 'max_percent_variance' must be >= 0 and <= 100"
	assert min_percent_variance <= max_percent_variance, "plotDimensionEstimateOfPoint: Provided value for 'min_percent_variance' must be less than or equal to value for 'max_percent_variance'"
	# Plot options
	assert type(n_samples) == int, "plotDimensionEstimateOfPoint: Provided value for 'n_samples' must be a int object"
	assert 10 <= n_samples and n_samples <= 1000, "plotDimensionEstimateOfPoint: Provided value for 'n_samples' must be >= 10 and <= 1000"
	assert used_engine in ["matplotlib", "plotly"], "plotDimensionEstimateOfPoint: Provided value for 'used_engine' must be 'matplotlib' or 'plotly'"
	assert type(round_flag) == bool, "plotDimensionEstimateOfPoint: Provided value for 'round_flag' must be a bool object"
	assert type(show_flag) == bool, "plotDimensionEstimateOfPoint: Provided value for 'show_flag' must be a bool object"
	assert type(save_flag) == bool, "plotDimensionEstimateOfPoint: Provided value for 'save_flag' must be a bool object"

	# Make sure the softmax distance bounds are valid given the contents of the db file
	assert read_row[2] <= min_softmax_distance and min_softmax_distance <= read_row[3], "plotDimensionEstimateOfPoint: Provided value for 'min_softmax_distance' must fall in the range given by the db file (in this case " + str(read_row[2]) + " to " + str(read_row[3]) + ")"
	assert read_row[2] <= max_softmax_distance and max_softmax_distance <= read_row[3], "plotDimensionEstimateOfPoint: Provided value for 'max_softmax_distance' must fall in the range given by the db file (in this case " + str(read_row[2]) + " to " + str(read_row[3]) + ")"

	# Make sure at least one of the bounds is non-trivial
	assert min_softmax_distance < max_softmax_distance or min_percent_variance < max_percent_variance, "plotDimensionEstimateOfPoint: At least one of 'min_softmax_distance' < 'max_softmax_distance' or 'min_percent_variance' < 'max_percent_variance' must be True"

	# Get a path to which the image should be saved and make sure cancel wasn't clicked (if needed)
	if save_flag == True:
		# Get the needed image path
		if used_engine == "matplotlib":
			image_path = askSaveFilename(allowed_extensions = ["png"])
			assert image_path is not None, "plotDimensionEstimateOfPoint: Unable to save matplotlib figure because cancel button was clicked"
		else:
			image_path = askSaveFilename(allowed_extensions = ["html"])
			assert image_path is not None, "plotDimensionEstimateOfPoint: Unable to save plotly figure because cancel button was clicked"
	else:
		# Set the image path to None because it is not needed
		image_path = None

	# Define an internal helper function for 2D plots
	def plot2D(x_values:list, y_values:list, plot_title:str, x_label:str, y_label:str, used_engine:str, image_path:Union[PosixPath, WindowsPath]):
		# Create, show and save the needed figure of the provided data
		# Plot the needed information
		if used_engine == "matplotlib":
			# Handle the case of using matplotlib
			# Create the figure
			plt.figure(figsize = (10, 8), layout = "constrained")

			# Add the needed traces
			plt.plot(x_values,y_values)

			# Format the figure
			plt.title(plot_title)
			plt.xlabel(x_label)
			plt.ylabel(y_label)
			plt.grid()

			# Show the figure (if needed)
			if show_flag == True:
				plt.show()

			# Save the figure (if needed)
			if save_flag == True:
				plt.savefig(image_path)
		else:
			# Handle the case of using plotly
			# Create the figure
			fig = go.Figure()

			# Add the needed traces
			fig.add_trace(go.Scatter(x = x_values, y = y_values, showlegend = False))

			# Format the figure
			fig.update_layout(title = plot_title)
			fig.update_xaxes(title = x_label)
			fig.update_yaxes(title = y_label)

			# Show the figure (if needed)
			if show_flag == True:
				fig.show()

			# Save the figure (if needed)
			if save_flag == True:
				fig.write_html(image_path)

	# Handle the various cases
	if min_softmax_distance == max_softmax_distance:
		# Create a 2D plot with fixed softmax distance
		# Generate the x-values and y-values for this plot
		x_values = []
		y_values = []
		for sample_index in range(n_samples):
			# Compute the percent variance and corresponding dimension estimate
			percent_variance = min_percent_variance + (max_percent_variance - min_percent_variance) * sample_index / (n_samples - 1)
			dimension_estimate = estimatePointwiseDimension(db_path = db_path,
															softmax_distance = min_softmax_distance,
															percent_variance = percent_variance,
															needed_indices = [row_index])[row_index]
			# Append to the needed lists
			x_values.append(percent_variance)
			y_values.append(round(dimension_estimate) if round_flag == True else dimension_estimate)

		# Define shared plot information
		plot_title = ("(Rounded) " if round_flag == True else "") + "Estimated Dimension Of Point " + str(row_index) + " (As Function Of Explained Variance, Softmax Distance Of " + str(min_softmax_distance) + ")"
		x_label = "explained variance"
		y_label = "estimated dimension"

		# Handle the needed plotting of data
		plot2D(x_values = x_values, y_values = y_values, plot_title = plot_title, x_label = x_label, y_label = y_label, used_engine = used_engine, image_path = image_path)
	elif min_percent_variance == max_percent_variance:
		# Create a 2D plot with fixed percent variance
		# Generate the x-values and y-values for this plot
		x_values = []
		y_values = []
		for sample_index in range(n_samples):
			# Compute the percent variance and corresponding dimension estimate
			softmax_distance = min_softmax_distance + (max_softmax_distance - min_softmax_distance) * sample_index / (n_samples - 1)
			dimension_estimate = estimatePointwiseDimension(db_path = db_path,
															softmax_distance = softmax_distance,
															percent_variance = min_percent_variance,
															needed_indices = [row_index])[row_index]
			# Append to the needed lists
			x_values.append(softmax_distance)
			y_values.append(round(dimension_estimate) if round_flag == True else dimension_estimate)

		# Define shared plot information
		plot_title = ("(Rounded) " if round_flag == True else "") + "Estimated Dimension Of Point " + str(row_index) + " (As Function Of Softmax Distance, Explained Variance Of " + str(min_percent_variance) + "%)"
		x_label = "softmax distance"
		y_label = "estimated dimension"

		# Handle the needed plotting of data
		plot2D(x_values = x_values, y_values = y_values, plot_title = plot_title, x_label = x_label, y_label = y_label, used_engine = used_engine, image_path = image_path)
	else:
		# Compute the coverage ratios for softmax distance and percent variance
		softmax_ratio = (max_softmax_distance - min_softmax_distance) / (read_row[3] - read_row[2])
		percent_ratio = (max_percent_variance - min_percent_variance) / 100

		# Compute the corresponding numbers of softmax distance and percent variance samples
		n_samples_softmax = 1 + int(sqrt(n_samples * softmax_ratio / percent_ratio))
		n_samples_percent = 1 + int(sqrt(n_samples * percent_ratio / softmax_ratio))

		# Adjust the number of samples to make sure they are both at least 2
		if n_samples_softmax == 1:
			n_samples_softmax = 2
		if n_samples_percent == 1:
			n_samples_percent = 2

		# Create arrays containing the needed values depending on the render engine used
		if used_engine == "matplotlib":
			# Get the RGB spectrum as hex codes
			rgb_hex_spectrum = [customSpectrum(parameter = index / 100).asStringHex() for index in range(101)]
			# Convert to a color map usable by matplotlib
			color_map = mcolors.LinearSegmentedColormap.from_list("my_custom_scale", rgb_hex_spectrum)
			# Initialize the lists of needed x-values, y-values and z-values
			x_values = []
			y_values = []
			z_values = []
			# Create arrays for x-values, y-values and z-values
			for percent_index in range(n_samples_percent):
				# Create the new rows for the lists of lists
				new_row_x = []
				new_row_y = []
				new_row_z = []
				# Fill in the values for the new rows
				for softmax_index in range(n_samples_softmax):
					# Compute the new x-value, y-value and z-value
					softmax_distance = min_softmax_distance + (max_softmax_distance - min_softmax_distance) * softmax_index / (n_samples_softmax - 1)
					percent_variance = min_percent_variance + (max_percent_variance - min_percent_variance) * percent_index / (n_samples_percent - 1)
					dimension_estimate = estimatePointwiseDimension(db_path = db_path,
																	softmax_distance = softmax_distance,
																	percent_variance = percent_variance,
																	needed_indices = [row_index])[row_index]
					# Append to the needed lists
					new_row_x.append(softmax_distance)
					new_row_y.append(percent_variance)
					new_row_z.append(round(dimension_estimate) if round_flag == True else dimension_estimate)
				# Add the new rows to the arrays
				x_values.append(new_row_x)
				y_values.append(new_row_y)
				z_values.append(new_row_z)
			# Convert the z-values to a numpy array
			z_values = array(z_values, dtype = float)
		else:
			# Get a color scale usable by plotly
			color_scale = [[index / 100, customSpectrum(parameter = index / 100).asStringTuple()] for index in range(101)]
			# Create an array for the x-values
			x_values = zeros(n_samples_softmax, dtype = float)
			for softmax_index in range(n_samples_softmax):
				x_values[softmax_index] = min_softmax_distance + (max_softmax_distance - min_softmax_distance) * softmax_index / (n_samples_softmax - 1)
			# Create an array for the y-values
			y_values = zeros(n_samples_percent, dtype = float)
			for percent_index in range(n_samples_percent):
				y_values[percent_index] = min_percent_variance + (max_percent_variance - min_percent_variance) * percent_index / (n_samples_percent - 1)
			# Create an array for the z-values as well as associated point labels
			z_values = zeros((n_samples_percent, n_samples_softmax), dtype = float)
			point_labels = zeros((n_samples_softmax, n_samples_percent, 3), dtype = float)
			for percent_index in range(n_samples_percent):
				for softmax_index in range(n_samples_softmax):
					# Fill in the information for the z-value
					dimension_estimate = estimatePointwiseDimension(db_path = db_path,
																	softmax_distance = x_values[softmax_index],
																	percent_variance = y_values[percent_index],
																	needed_indices = [row_index])[row_index]
					z_values[percent_index, softmax_index] = round(dimension_estimate) if round_flag == True else dimension_estimate
					# Add in the information for the point labels
					point_labels[softmax_index, percent_index, 0] = round(x_values[softmax_index], 3)
					point_labels[softmax_index, percent_index, 1] = round(y_values[percent_index], 3)
					point_labels[softmax_index, percent_index, 2] = round(z_values[percent_index, softmax_index], 3)

		# Define shared plot information
		plot_title = ("(Rounded) " if round_flag == True else "") + "Estimated Dimension Of Point " + str(row_index) + " (As Function Of Softmax Distance And Explained Variance)"
		x_label = "softmax distance"
		y_label = "explained variance"
		z_label = "estimated dimension"

		# Plot the needed information
		if used_engine == "matplotlib":
			# Handle the case of using matplotlib
			# Create the figure
			fig = plt.figure(figsize = (10, 8), layout = "constrained")
			ax = fig.add_subplot(111, projection = "3d")
			# Add the needed traces
			surface_plot = ax.plot_surface(x_values, y_values, z_values, cmap = color_map)
			# Format the figure
			plt.title(plot_title)
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.set_zlabel(z_label)
			fig.colorbar(surface_plot, ax = ax, pad = 0.1)
			surface_plot.set_clim(0, n_cols)
			# Show the figure (if needed)
			if show_flag == True:
				plt.show()
			# Save the figure (if needed)
			if save_flag == True:
				plt.savefig(image_path)
		else:
			# Handle the case of using plotly
			# Create the figure
			fig = go.Figure()
			# Add the needed traces
			fig.add_trace(go.Surface(x = x_values,
									 y = y_values,
									 z = z_values,
									 showlegend = False,
									 customdata = point_labels,
									 hovertemplate = ("<b>Softmax Distance:</b> %{customdata[0]}<br>" +
									 				  "<b>Percent Variance:</b> %{customdata[1]}<br>" +
									   			      "<b>Estimated Dimension:</b> %{customdata[2]}<br>" +
									   			   	  "<extra></extra>"),
									 colorscale = color_scale,
									 cmin = 0,
									 cmax = n_cols))
			# Format the figure
			fig.update_layout(title = plot_title,
							  scene = {"xaxis_title": x_label,
							  		   "yaxis_title": y_label,
							  		   "zaxis_title": z_label})
			# Show the figure (if needed)
			if show_flag == True:
				fig.show()
			# Save the figure (if needed)
			if save_flag == True:
				fig.write_html(image_path)

def plotDimensionEstimateOfSet(db_path:Union[PosixPath, WindowsPath], softmax_distance:Any, percent_variance:Any, plot_type:str,
							   used_engine:str = "matplotlib", round_flag:bool = False, show_flag:bool = True, save_flag:bool = False):
	# Generate a scatter plot representing the pointwise dimension estimates
	# Verify that the provided db file is a valid dimension database
	verifyDimensionDatabase(db_path = db_path)

	# Get the relevant input settings from the db file
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]
	min_softmax_distance = read_row[2]
	max_softmax_distance = read_row[3]

	# Verify the other inputs
	assert isNumeric(softmax_distance, include_numpy_flag = True) == True, "plotDimensionEstimateOfSet: Provided value for 'softmax_distance' must be numeric"
	assert 0 < softmax_distance and softmax_distance < float("inf"), "plotDimensionEstimateOfSet: Provided value for 'softmax_distance' must be positive and finite"
	assert isNumeric(percent_variance, include_numpy_flag = True) == True, "plotDimensionEstimateOfSet: Provided value for 'percent_variance' must be numeric"
	assert 0 <= percent_variance and percent_variance <= 100, "plotDimensionEstimateOfSet: Provided value for 'percent_variance' must be >= 0 and <= 100"
	assert plot_type in ["bar", "scatter2D", "scatter3D"], "plotDimensionEstimateOfSet: Provided value for 'plot_type' must be 'bar', 'scatter2D' or 'scatter3D'"
	assert used_engine in ["matplotlib", "plotly"], "plotDimensionEstimateOfSet: Provided value for 'used_engine' must be 'matplotlib' or 'plotly'"
	assert type(round_flag) == bool, "plotDimensionEstimateOfSet: Provided value for 'round_flag' must be a bool object"
	assert type(show_flag) == bool, "plotDimensionEstimateOfSet: Provided value for 'show_flag' must be a bool object"
	assert type(save_flag) == bool, "plotDimensionEstimateOfSet: Provided value for 'save_flag' must be a bool object"
	assert show_flag == True or save_flag == True, "plotDimensionEstimateOfSet: At least of the provided values for 'show_flag' and 'save_flag' must be True"

	# Make sure the softmax distance are valid given the contents of the db file
	assert min_softmax_distance <= softmax_distance and softmax_distance <= max_softmax_distance, "plotDimensionEstimateOfSet: Provided value for 'softmax_distance' must fall in the range given by the db file (in this case " + str(min_softmax_distance) + " to " + str(max_softmax_distance) + ")"

	# Make sure the number of columns is sufficiently large
	if plot_type == "scatter3D":
		assert n_cols >= 3, "plotDimensionEstimateOfSet: Number of columns in raw data set must be at least 3 when value for 'plot_type' is 'scatter3D'"
	else:
		assert n_cols >= 2, "plotDimensionEstimateOfSet: Number of columns in raw data set must be at least 2 when value for 'plot_type' is 'bar' or 'scatter2D'"

	# Load the projected data array from the db file
	projected_data_array = zeros((n_rows, n_cols), dtype = float)
	for row_index in range(n_rows):
		projected_data_array[row_index, :] = readRow(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY, row_index = row_index)

	# Estimate the pointwise dimension for each point at the needed percent variance (converted to a list)
	dimension_results = list(estimatePointwiseDimension(db_path = db_path, softmax_distance = softmax_distance, percent_variance = percent_variance).values())

	# Round the dimension estimates to the nearest integer (if needed)
	if round_flag == True:
		dimension_results = [round(value) for value in dimension_results]

	# Get all information needed showing the dimension estimates
	if used_engine == "matplotlib":
		# Get the RGB spectrum as hex codes
		rgb_hex_spectrum = [customSpectrum(parameter = index / 100).asStringHex() for index in range(101)]
		# Convert to a color map usable by matplotlib
		color_map = mcolors.LinearSegmentedColormap.from_list("my_custom_scale", rgb_hex_spectrum)
	else:
		# Get a color scale usable by plotly
		color_scale = [[index / 100, customSpectrum(parameter = index / 100).asStringTuple()] for index in range(101)]
		# Get the labels needed for each point
		point_labels = []
		for row_index in range(n_rows):
			if round_flag == True:
				point_labels.append((row_index, dimension_results[row_index]))
			else:
				point_labels.append((row_index, round(dimension_results[row_index], 3)))

	# Define shared plot information
	plot_title = ("(Rounded) " if round_flag == True else "") + "Estimated Pointwise Dimension Of Set (Softmax Distance Of " + str(softmax_distance) + ", Explained Variance Of " + str(percent_variance) + "%)"
	if plot_type == "bar":
		x_label = "point index"
		y_label = "estimated dimension"
	else:
		x_label = "1st principal direction"
		y_label = "2nd principal direction"
		z_label = "3rd principal direction"

	# Get a path to which the image should be saved and make sure cancel wasn't clicked (if needed)
	if save_flag == True:
		if used_engine == "matplotlib":
			image_path = askSaveFilename(allowed_extensions = ["png"])
			assert image_path is not None, "plotDimensionEstimateOfPoint: Unable to save matplotlib figure because cancel button was clicked"
		else:
			image_path = askSaveFilename(allowed_extensions = ["html"])
			assert image_path is not None, "plotDimensionEstimateOfPoint: Unable to save plotly figure because cancel button was clicked"

	# Create a scatter plot to visualize the pointwise dimension
	if used_engine == "matplotlib":
		# Create the needed matplotlib figure
		# Create the figure (and axis if needed)
		fig = plt.figure(figsize = (10, 8), layout = "constrained")
		if plot_type == "bar":
			ax = fig.add_subplot()
		elif plot_type == "scatter3D":
			ax = fig.add_subplot(projection = "3d")
		# Handle the various cases
		if plot_type == "bar":
			# Set the color normalizer and get the colors by height
			normalizer = plt.Normalize(0, n_cols)
			bar_colors = color_map(normalizer(dimension_results))
			# Add the needed bars
			bar_plot = ax.bar([str(row_index) for row_index in range(n_rows)], dimension_results, color = bar_colors)
			# Create the needed colorbar
			scalar_mappable = ScalarMappable(cmap = color_map, norm = normalizer)
			scalar_mappable.set_array([])
			fig.colorbar(scalar_mappable, ax = ax, pad = 0.1)
		elif plot_type == "scatter2D":
			# Scatter in two dimensions
			plt.scatter(projected_data_array[:, 0], projected_data_array[:, 1], c = dimension_results, cmap = color_map)
			# Create the needed colorbar
			plt.colorbar()
			plt.clim(0, n_cols)
			# Turn on the grid
			plt.grid()
		else:
			# Scatter in three dimensions
			scatter_plot = ax.scatter(projected_data_array[:, 0], projected_data_array[:, 1], projected_data_array[:, 2], c = dimension_results, cmap = color_map)
			# Create the needed colorbar
			fig.colorbar(scatter_plot, ax = ax, pad = 0.1)
			scatter_plot.set_clim(0, n_cols)
		# Perform additional formating for the figure
		plt.title(plot_title)
		if plot_type != "scatter3D":
			plt.xlabel(x_label)
			plt.ylabel(y_label)
			plt.axis("equal")
		else:
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.set_zlabel(z_label)
			ax.set_aspect("equal")
		# Show the figure (if needed)
		if show_flag == True:
			plt.show()
		# Save the figure (if needed)
		if save_flag == True:
			plt.savefig(image_path)
	else:
		# Create the needed plotly figure
		# Create the figure
		fig = go.Figure()
		# Handle the various cases
		if plot_type == "bar":
			# Add the needed bars
			fig.add_trace(go.Bar(x = [str(row_index) for row_index in range(n_rows)],
								 y =  dimension_results,
								 showlegend = False,
								 customdata = point_labels,
								 hovertemplate = ("<b>Index Of Point:</b> %{customdata[0]}<br>"
												  "<b>Estimated Dimension:</b> %{customdata[1]}<br>"
												  "<extra></extra>"),
								 marker = {"color": dimension_results,
								 		   "colorscale": color_scale,
								 		   "showscale": True,
								 		   "cmin": 0,
									       "cmax": n_cols}))
		elif plot_type == "scatter2D":
			# Scatter in two dimensions
			fig.add_trace(go.Scatter(x = projected_data_array[:, 0],
			                         y = projected_data_array[:, 1],
			                         showlegend = False,
									 customdata = point_labels,
								   	 hovertemplate = ("<b>Index Of Point:</b> %{customdata[0]}<br>"
													  "<b>Estimated Dimension:</b> %{customdata[1]}<br>"
													  "<extra></extra>"),
									 mode = "markers",
									 marker = {"color": dimension_results,
									           "colorscale": color_scale,
									           "showscale": True,
									           "cmin": 0,
									           "cmax": n_cols}))
		else:
			# Scatter in three dimensions
			fig.add_trace(go.Scatter3d(x = projected_data_array[:, 0],
			                           y = projected_data_array[:, 1],
									   z = projected_data_array[:, 2],
									   showlegend = False,
									   customdata = point_labels,
									   hovertemplate = ("<b>Index Of Point:</b> %{customdata[0]}<br>" +
									   					"<b>Estimated Dimension:</b> %{customdata[1]}<br>" +
									   					"<extra></extra>"),
									   mode = "markers",
									   marker = {"color": dimension_results,
									             "colorscale": color_scale,
									             "showscale": True,
									             "cmin": 0,
									             "cmax": n_cols,
									             "size": 3}))
		# Perform additional formating for the figure
		if plot_type != "scatter3D":
			fig.update_layout(title = plot_title)
			fig.update_xaxes(title = x_label)
			fig.update_yaxes(title = y_label)
		else:
			fig.update_layout(title = plot_title,
							  scene = {"xaxis_title": x_label,
							           "yaxis_title": y_label,
							           "zaxis_title": z_label})
		# Show the figure (if needed)
		if show_flag == True:
			fig.show()
		# Save the figure (if needed)
		if save_flag == True:
			fig.write_html(image_path)