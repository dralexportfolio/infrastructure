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
from spline_helper import LinearSpline
from sqlite3_helper import addTable, appendRow, getColumnNames, getColumnTypes, getExistingTables, getRowCount, readColumn, readRow, replaceRow
from tkinter_helper import askSaveFilename
from type_helper import isNumeric

# External modules
from os import remove
from os.path import exists
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numpy import cumsum, mean, ndarray, zeros
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
BASE_TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES = "cumulative_percent_variances"
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
		assert min_softmax_distance == max_softmax_distance, "generateDimensionDatabase: If provided value for 'n_distances' is equal to 1 then value for 'min_softmax_distance' must be equal to value for 'max_softmax_distance'"
	else:
		assert min_softmax_distance < max_softmax_distance, "generateDimensionDatabase: If provided value for 'n_distances' is greater than 1 then value for 'min_softmax_distance' must be less than value for 'max_softmax_distance'"

	# Compute the softmax distances to use and get the associated table names
	all_softmax_distances = [min_softmax_distance]
	all_cumulative_table_names = [BASE_TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES + "_0"]
	for distance_index in range(1, n_distances):
		all_softmax_distances.append(min_softmax_distance + (max_softmax_distance - min_softmax_distance) * distance_index / (n_distances - 1))
		all_cumulative_table_names.append(BASE_TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES + "_" + str(distance_index))

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
		current_table_name = BASE_TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES + "_" + str(distance_index)
		assert current_table_name in table_names, "verifyDimensionDatabase: Provided value for 'db_path' must refer to a database with " + current_table_name + " as a table name"
		assert getColumnNames(db_path = db_path, table_name = current_table_name) == COLUMN_NAMES_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(n_cols), "verifyDimensionDatabase: Table of name " + current_table_name + " has the incorrect column names"
		assert getColumnTypes(db_path = db_path, table_name = current_table_name) == COLUMN_TYPES_FUNCTION_CUMULATIVE_PERCENT_VARIANCES(n_cols), "verifyDimensionDatabase: Table of name " + current_table_name + " has the incorrect column types"
		assert getRowCount(db_path = db_path, table_name = current_table_name) == n_rows, "verifyDimensionDatabase: Table of name " + current_table_name + " has the incorrect number or rows"


#########################################################################################
### Define a function which estimates the local dimension of each point in a data set ###
#########################################################################################
def estimatePointwiseDimension(db_path:Union[PosixPath, WindowsPath], percent_variance:Any, softmax_distance:Any) -> list:
	# Compute the pointwise dimension for each point the data stored in the pre-computed db file
	# Verify that the provided db file is a valid dimension database
	verifyDimensionDatabase(db_path = db_path)

	# Verify the other inputs
	assert isNumeric(percent_variance, include_numpy_flag = True) == True, "estimatePointwiseDimension: Provided value for 'percent_variance' must be numeric"
	assert 0 <= percent_variance and percent_variance <= 100, "estimatePointwiseDimension: Provided value for 'percent_variance' must be >= 0 and <= 100"
	assert isNumeric(softmax_distance, include_numpy_flag = True) == True, "estimatePointwiseDimension: Provided value for 'softmax_distance' must be numeric"

	# Get the relevant input settings from the db file
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]
	min_softmax_distance = read_row[2]
	max_softmax_distance = read_row[3]
	n_distances = read_row[4]

	# Verify that the softmax is valid
	assert min_softmax_distance <= softmax_distance and softmax_distance <= max_softmax_distance, "estimatePointwiseDimension: Provided value for 'softmax_distance' must be between minimum and maximum softmax distances stored in this db file"

	# Get the softmax distances and associated table names from the db file
	all_softmax_distances = readColumn(db_path = db_path, table_name = TABLE_NAME_DISTANCES_USED, column_name = "softmax_distance")
	all_cumulative_table_names = readColumn(db_path = db_path, table_name = TABLE_NAME_DISTANCES_USED, column_name = "table_name")

	# Determine which softmax distances to use from the table and associated weights
	if n_distances == 1:
		n_tables_used = 1
		needed_weights = [1]
		needed_cumulative_table_names = all_cumulative_table_names
	else:
		n_tables_used = 2
		for distance_index in range(n_distances - 1):
			softmax_distance_1 = all_softmax_distances[distance_index]
			softmax_distance_2 = all_softmax_distances[distance_index + 1]
			if softmax_distance_1 <= softmax_distance and softmax_distance <= softmax_distance_2:
				denominator = softmax_distance_2 - softmax_distance_1
				needed_weights = [(softmax_distance_2 - softmax_distance) / denominator, (softmax_distance - softmax_distance_1) / denominator]
				needed_cumulative_table_names = [all_cumulative_table_names[distance_index], all_cumulative_table_names[distance_index + 1]]
				break

	# Create the y-values for the linear splines
	y_values = [index for index in range(n_cols + 1)]
	
	# Initialize the list of results
	dimension_results = []
	
	# Compute the dimension estimates for each point
	for row_index in range(n_rows):
		# Initialize the current dimension estimate
		dimension_estimate = 0

		# Take the weighted mean of values from the needed tables
		for table_index in range(n_tables_used):
			# Read the needed row from the db file as the x-values
			x_values = readRow(db_path = db_path, table_name = needed_cumulative_table_names[table_index], row_index = row_index)
		
			# Create a linear spline using these x-values and y-values
			linear_spline = LinearSpline(x_values = x_values, y_values = y_values)

			# Update the dimension estimate according to the given weight
			dimension_estimate += needed_weights[table_index] * linear_spline.evaluate(x_value = percent_variance)

		# Append the current dimension estimate
		dimension_results.append(dimension_estimate)
	
	# Return the results
	return dimension_results
	
'''
###########################################################################################
### Define functions for generating visualizations of the pointwise dimension estimates ###
###########################################################################################
def plotCumulativeVariances(db_path:Union[PosixPath, WindowsPath], used_engine:str = "matplotlib", mean_only_flag:bool = False, show_flag:bool = True, save_flag:bool = False):
	# Generate a plot of cumulative percent variances for each point
	# Verify the inputs
	assert type(db_path) in [PosixPath, WindowsPath], "plotCumulativeVariances: Provided value for 'db_path' must be a PosixPath or WindowsPath object"
	assert used_engine in ["matplotlib", "plotly"], "plotCumulativeVariances: Provided value for 'used_engine' must be 'matplotlib' or 'plotly'"
	assert type(mean_only_flag) == bool, "plotCumulativeVariances: Provided value for 'mean_only_flag' must be a bool object"
	assert type(show_flag) == bool, "plotCumulativeVariances: Provided value for 'show_flag' must be a bool object"
	assert type(save_flag) == bool, "plotCumulativeVariances: Provided value for 'save_flag' must be a bool object"
	assert show_flag == True or save_flag == True, "plotCumulativeVariances: At least of the provided values for 'show_flag' and 'save_flag' must be True"

	# Get the number of rows and columns in the raw data (as well as the softmax distance)
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]
	softmax_distance = read_row[2]

	# Load the cumulative percent variances from the db file
	cumulative_percent_variances = zeros((n_rows, n_cols + 1), dtype = float)
	for row_index in range(n_rows):
		cumulative_percent_variances[row_index, :] = readRow(db_path = db_path, table_name = TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES, row_index = row_index)

	# Get the x-values and title shared between the traces
	x_values = [index for index in range(n_cols + 1)]
	plot_title = "Cumulative Percent Variances From Dimension Estimation (Softmax Distance Of " + str(softmax_distance) + ")"

	# Plot the needed information
	if used_engine == "matplotlib":
		# Handle the case of using matplotlib
		# Create the figure
		plt.figure(figsize = (10, 8))
		# Add the needed traces
		if mean_only_flag == False:
			for row_index in range(n_rows):
				plt.plot(x_values, cumulative_percent_variances[row_index, :], color = customSpectrum(parameter = row_index / (n_rows - 1)).asStringHex(), zorder = 0)
			plt.plot(x_values, mean(cumulative_percent_variances, axis = 0), label = "Mean", color = "black", zorder = 10)
		else:
			plt.plot(x_values, mean(cumulative_percent_variances, axis = 0), color = "black")
		# Format the figure
		if mean_only_flag == False:
			plt.title(plot_title)
			plt.legend()
		else:
			plt.title("Mean " + plot_title)
		plt.xlabel("number of principal directions")
		plt.ylabel("cumulative percent variance")
		plt.grid()
		# Show the figure (if needed)
		if show_flag == True:
			plt.show()
		# Save the figure (if needed)
		if save_flag == True:
			# Get a path to which the image should be saved and make sure cancel wasn't clicked
			image_path = askSaveFilename(allowed_extensions = ["png"])
			assert image_path is not None, "plotCumulativeVariances: Unable to save matplotlib figure because cancel button was clicked"
			# Save the image to this location
			plt.savefig(image_path)
	else:
		# Handle the case of using plotly
		# Create the figure
		fig = go.Figure()
		# Add the needed traces
		if mean_only_flag == False:
			for row_index in range(n_rows):
				fig.add_trace(go.Scatter(x = x_values, y = cumulative_percent_variances[row_index, :], name = "Point #" + str(row_index + 1),
										 marker = {"color": customSpectrum(parameter = row_index / (n_rows - 1)).asStringTuple()}))
			fig.add_trace(go.Scatter(x = x_values, y = mean(cumulative_percent_variances, axis = 0), name = "Mean", marker = {"color": "black"}))
		else:
			fig.add_trace(go.Scatter(x = x_values, y = mean(cumulative_percent_variances, axis = 0), showlegend = False, marker = {"color": "black"}))
		# Format the figure
		if mean_only_flag == False:
			fig.update_layout(title = plot_title)
		else:
			fig.update_layout(title = "Mean " + plot_title)
		fig.update_xaxes(title = "number of principal directions")
		fig.update_yaxes(title = "cumulative percent variance")
		# Show the figure (if needed)
		if show_flag == True:
			fig.show()
		# Save the figure (if needed)
		if save_flag == True:
			# Get a path to which the image should be saved and make sure cancel wasn't clicked
			image_path = askSaveFilename(allowed_extensions = ["html"])
			assert image_path is not None, "plotCumulativeVariances: Unable to save plotly figure because cancel button was clicked"
			# Save the image to this location
			fig.write_html(image_path)

def plotMarginalVariances(db_path:Union[PosixPath, WindowsPath], used_engine:str = "matplotlib", mean_only_flag:bool = False, show_flag:bool = True, save_flag:bool = False):
	# Generate a plot of marginal percent variances for each point
	# Verify the inputs
	assert type(db_path) in [PosixPath, WindowsPath], "plotMarginalVariances: Provided value for 'db_path' must be a PosixPath or WindowsPath object"
	assert used_engine in ["matplotlib", "plotly"], "plotMarginalVariances: Provided value for 'used_engine' must be 'matplotlib' or 'plotly'"
	assert type(mean_only_flag) == bool, "plotMarginalVariances: Provided value for 'mean_only_flag' must be a bool object"
	assert type(show_flag) == bool, "plotMarginalVariances: Provided value for 'show_flag' must be a bool object"
	assert type(save_flag) == bool, "plotMarginalVariances: Provided value for 'save_flag' must be a bool object"
	assert show_flag == True or save_flag == True, "plotMarginalVariances: At least of the provided values for 'show_flag' and 'save_flag' must be True"

	# Get the number of rows and columns in the raw data (as well as the softmax distance)
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]
	softmax_distance = read_row[2]

	# Load the cumulative percent variances from the db file
	cumulative_percent_variances = zeros((n_rows, n_cols + 1), dtype = float)
	for row_index in range(n_rows):
		cumulative_percent_variances[row_index, :] = readRow(db_path = db_path, table_name = TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES, row_index = row_index)

	# Compute the marginal percent variances from the cumulative
	marginal_percent_variances = zeros((n_rows, n_cols), dtype = float)
	for col_index in range(n_cols):
		marginal_percent_variances[:, col_index] = cumulative_percent_variances[:, col_index + 1] - cumulative_percent_variances[:, col_index]

	# Get the x-values and title shared between the traces
	x_values = [index + 1 for index in range(n_cols)]
	plot_title = "Marginal Percent Variances From Dimension Estimation (Softmax Distance Of " + str(softmax_distance) + ")"

	# Plot the needed information
	if used_engine == "matplotlib":
		# Handle the case of using matplotlib
		# Create the figure
		plt.figure(figsize = (10, 8))
		# Add the needed traces
		if mean_only_flag == False:
			for row_index in range(n_rows):
				plt.plot(x_values, marginal_percent_variances[row_index, :], color = customSpectrum(parameter = row_index / (n_rows - 1)).asStringHex(), zorder = 0)
			plt.plot(x_values, mean(marginal_percent_variances, axis = 0), label = "Mean", color = "black", zorder = 10)
		else:
			plt.plot(x_values, mean(marginal_percent_variances, axis = 0), color = "black")
		# Format the figure
		if mean_only_flag == False:
			plt.title(plot_title)
			plt.legend()
		else:
			plt.title("Mean " + plot_title)
		plt.xlabel("number of principal directions")
		plt.ylabel("marginal percent variance")
		plt.grid()
		# Show the figure (if needed)
		if show_flag == True:
			plt.show()
		# Save the figure (if needed)
		if save_flag == True:
			# Get a path to which the image should be saved and make sure cancel wasn't clicked
			image_path = askSaveFilename(allowed_extensions = ["png"])
			assert image_path is not None, "plotMarginalVariances: Unable to save matplotlib figure because cancel button was clicked"
			# Save the image to this location
			plt.savefig(image_path)
	else:
		# Handle the case of using plotly
		# Create the figure
		fig = go.Figure()
		# Add the needed traces
		if mean_only_flag == False:
			for row_index in range(n_rows):
				fig.add_trace(go.Scatter(x = x_values, y = marginal_percent_variances[row_index, :], name = "Point #" + str(row_index + 1),
										 marker = {"color": customSpectrum(parameter = row_index / (n_rows - 1)).asStringTuple()}))
			fig.add_trace(go.Scatter(x = x_values, y = mean(marginal_percent_variances, axis = 0), name = "Mean", marker = {"color": "black"}))
		else:
			fig.add_trace(go.Scatter(x = x_values, y = mean(marginal_percent_variances, axis = 0), showlegend = False, marker = {"color": "black"}))
		# Format the figure
		if mean_only_flag == False:
			fig.update_layout(title = plot_title)
		else:
			fig.update_layout(title = "Mean " + plot_title)
		fig.update_xaxes(title = "number of principal directions")
		fig.update_yaxes(title = "marginal percent variance")
		# Show the figure (if needed)
		if show_flag == True:
			fig.show()
		# Save the figure (if needed)
		if save_flag == True:
			# Get a path to which the image should be saved and make sure cancel wasn't clicked
			image_path = askSaveFilename(allowed_extensions = ["html"])
			assert image_path is not None, "plotMarginalVariances: Unable to save plotly figure because cancel button was clicked"
			# Save the image to this location
			fig.write_html(image_path)

def visualizePointwiseEstimate(db_path:Union[PosixPath, WindowsPath], percent_variance:Any, used_engine:str = "matplotlib", use_3d_flag:bool = False, show_flag:bool = True, save_flag:bool = False):
	# Generate a scatter plot representing the pointwise dimension estimates
	# Verify the inputs
	assert type(db_path) in [PosixPath, WindowsPath], "visualizePointwiseEstimate: Provided value for 'db_path' must be a PosixPath or WindowsPath object"
	assert isNumeric(percent_variance, include_numpy_flag = True) == True, "visualizePointwiseEstimate: Provided value for 'percent_variance' must be numeric"
	assert 0 <= percent_variance and percent_variance <= 100, "visualizePointwiseEstimate: Provided value for 'percent_variance' must be >= 0 and <= 100"
	assert used_engine in ["matplotlib", "plotly"], "visualizePointwiseEstimate: Provided value for 'used_engine' must be 'matplotlib' or 'plotly'"
	assert type(use_3d_flag) == bool, "visualizePointwiseEstimate: Provided value for 'use_3d_flag' must be a bool object"
	assert type(show_flag) == bool, "visualizePointwiseEstimate: Provided value for 'show_flag' must be a bool object"
	assert type(save_flag) == bool, "visualizePointwiseEstimate: Provided value for 'save_flag' must be a bool object"
	assert show_flag == True or save_flag == True, "visualizePointwiseEstimate: At least of the provided values for 'show_flag' and 'save_flag' must be True"

	# Get the number of rows and columns in the raw data (as well as the softmax distance)
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]
	softmax_distance = read_row[2]

	# Make sure the number of columns is sufficiently large
	if use_3d_flag == False:
		assert n_cols >= 2, "visualizePointwiseEstimate: Number of columns in raw data set must be at least 2 when visualizing in 2D"
	else:
		assert n_cols >= 3, "visualizePointwiseEstimate: Number of columns in raw data set must be at least 2 when visualizing in 3D"

	# Load the projected data array from the db file
	projected_data_array = zeros((n_rows, n_cols), dtype = float)
	for row_index in range(n_rows):
		projected_data_array[row_index, :] = readRow(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY, row_index = row_index)

	# Estimate the pointwise dimension for each point at the needed percent variance
	dimension_results = estimatePointwiseDimension(db_path = db_path, percent_variance = percent_variance)

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
		point_labels = [round(value, 3) for value in dimension_results]

	# Set the plot title and axis labels
	plot_title = "Estimated Pointwise Dimension Of Data (Softmax Distance Of " + str(softmax_distance) + ", Explained Variance Of " + str(percent_variance) + "%)"
	x_label = "1st principal direction"
	y_label = "2nd principal direction"
	z_label = "3rd principal direction"

	# Create a scatter plot to visualize the pointwise dimension
	if used_engine == "matplotlib":
		# Create the needed matplotlib figure
		if use_3d_flag == False:
			# Handle the 2-dimensional case
			# Create the figure
			plt.figure(figsize = (10, 8))
			# Add the needed traces
			plt.scatter(projected_data_array[:, 0], projected_data_array[:, 1], c = dimension_results, cmap = color_map)
			# Format the figure
			plt.title(plot_title)
			plt.xlabel(x_label)
			plt.ylabel(y_label)
			plt.grid()
			plt.colorbar()
			plt.clim(0, n_cols)
		else:
			# Handle the 3-dimensional case
			# Create the figure
			fig = plt.figure(figsize = (10, 8))
			ax = fig.add_subplot(projection = "3d")
			# Add the needed traces
			scatter_plot = ax.scatter(projected_data_array[:, 0], projected_data_array[:, 1], projected_data_array[:, 2], c = dimension_results, cmap = color_map)
			# Format the figure
			plt.title(plot_title)
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.set_zlabel(z_label)
			fig.colorbar(scatter_plot, ax = ax)
			scatter_plot.set_clim(0, n_cols)
		# Show the figure (if needed)
		if show_flag == True:
			plt.show()
		# Save the figure (if needed)
		if save_flag == True:
			# Get a path to which the image should be saved and make sure cancel wasn't clicked
			image_path = askSaveFilename(allowed_extensions = ["png"])
			assert image_path is not None, "visualizePointwiseEstimate: Unable to save matplotlib figure because cancel button was clicked"
			# Save the image to this location
			plt.savefig(image_path)
	else:
		# Create the needed plotly figure
		if use_3d_flag == False:
			# Handle the 2-dimensional case
			# Create the figure
			fig = go.Figure()
			# Add the needed traces
			fig.add_trace(go.Scatter(x = projected_data_array[:, 0],
			                         y = projected_data_array[:, 1],
			                         showlegend = False,
									 text = point_labels,
									 hovertemplate = "<b>Estimated Dimension:</b> %{text}<br>",
									 mode = "markers",
									 marker = {"color": dimension_results,
									           "colorscale": color_scale,
									           "showscale": True,
									           "cmin": 0,
									           "cmax": n_cols}))
			# Format the figure
			fig.update_layout(title = plot_title)
			fig.update_xaxes(title = x_label)
			fig.update_yaxes(title = y_label)
		else:
			# Handle the 3-dimensional case
			# Create the figure
			fig = go.Figure()
			# Add the needed traces
			fig.add_trace(go.Scatter3d(x = projected_data_array[:, 0],
			                           y = projected_data_array[:, 1],
									   z = projected_data_array[:, 2],
									   showlegend = False,
									   text = point_labels,
									   hovertemplate = "<b>Estimated Dimension:</b> %{text}<br>",
									   mode = "markers",
									   marker = {"color": dimension_results,
									             "colorscale": color_scale,
									             "showscale": True,
									             "cmin": 0,
									             "cmax": n_cols,
									             "size": 3}))
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
			# Get a path to which the image should be saved and make sure cancel wasn't clicked
			image_path = askSaveFilename(allowed_extensions = ["html"])
			assert image_path is not None, "visualizePointwiseEstimate: Unable to save plotly figure because cancel button was clicked"
			# Save the image to this location
			fig.write_html(image_path)
'''