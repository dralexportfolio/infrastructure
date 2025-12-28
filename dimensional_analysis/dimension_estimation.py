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
from dimension_reduction import performPCA
from spline_helper import LinearSpline
from sqlite3_helper import addTable, appendRow, getColumnNames, getRowCount, readRow, replaceRow
from tkinter_helper import askSaveFilename
from type_helper import isNumeric

# External modules
import matplotlib.pyplot as plt
from numpy import cumsum, ndarray, zeros
from numpy.linalg import norm
from pathlib import PosixPath, WindowsPath
import plotly.graph_objects as go
from scipy.special import softmax
from typing import Any, Union


##################################
### Set the needed table names ###
##################################
TABLE_NAME_INPUT_SETTINGS = "input_settings"
TABLE_NAME_RAW_DATA_ARRAY = "raw_data_array"
TABLE_NAME_PROJECTED_DATA_ARRAY = "projected_data_array"
TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES = "cumulative_percent_variances"


####################################################################################################################
### Define a function generates a db file needed for estimateing the local dimension of each point in a data set ###
####################################################################################################################
def generateDimensionDatabase(raw_data_array:ndarray, softmax_distance:Any) -> Union[PosixPath, WindowsPath]:
	# Use PCA to ompute the information for estimating pointwise dimension and write it to a db file, return the path written to
	# Verify the inputs
	assert type(raw_data_array) == ndarray, "generateDimensionDatabase: Provided value for 'raw_data_array' must be a numpy.ndarry object"
	assert len(raw_data_array.shape) == 2, "generateDimensionDatabase: Provided value for 'raw_data_array' must be a 2-dimensional numpy array"
	assert raw_data_array.shape[0] > 0, "generateDimensionDatabase: Provided value for 'raw_data_array' must have a non-zero number of points, i.e. at least 1 row"
	assert raw_data_array.shape[1] > 0, "generateDimensionDatabase: Provided value for 'raw_data_array' must have a non-zero number of features, i.e. at least 1 column"
	assert isNumeric(softmax_distance, include_numpy_flag = True) == True, "generateDimensionDatabase: Provided value for 'softmax_distance' must be numeric"
	assert 0 < softmax_distance and softmax_distance < float("inf"), "generateDimensionDatabase: Provided value for 'softmax_distance' must be positive and finite"
	
	# Get the db filename to save to, raise error if not selected
	db_path = askSaveFilename(allowed_extensions = ["db"])
	
	# Extract the number of rows and columns in the data
	n_rows = raw_data_array.shape[0]
	n_cols = raw_data_array.shape[1]
	
	# Create the column names and types needed for the needed tables
	# Input settings table
	input_settings_column_names = ["n_rows", "n_cols", "softmax_distance"]
	input_settings_column_types = ["BIGINT", "BIGINT", "REAL"]
	# Raw data array table
	raw_data_array_column_names = ["parameter_" + str(index + 1) for index in range(n_cols)]
	raw_data_array_column_types = ["REAL" for _ in range(n_cols)]
	# Projected data array table
	projected_data_array_column_names = ["parameter_" + str(index + 1) for index in range(n_cols)]
	projected_data_array_column_types = ["REAL" for _ in range(n_cols)]
	# Cumulative percent variances table
	cumulative_percent_variances_column_names = ["cumulative_" + str(index) for index in range(n_cols + 1)]
	cumulative_percent_variances_column_types = ["REAL" for _ in range(n_cols + 1)]
	
	# Create the needed db file and tables
	addTable(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, column_names = input_settings_column_names, column_types = input_settings_column_types)
	addTable(db_path = db_path, table_name = TABLE_NAME_RAW_DATA_ARRAY, column_names = raw_data_array_column_names, column_types = raw_data_array_column_types)
	addTable(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY, column_names = projected_data_array_column_names, column_types = projected_data_array_column_types)
	addTable(db_path = db_path, table_name = TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES, column_names = cumulative_percent_variances_column_names, column_types = cumulative_percent_variances_column_types)

	# Write the settings to the db file
	appendRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS)
	replaceRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0, new_row = [n_rows, n_cols, float(softmax_distance)])
	
	# Write the raw data array to the db file
	for row_index in range(n_rows):
		# Get the new row as float values
		new_row = [float(value) for value in raw_data_array[row_index, :]]
		# Write to the db file
		appendRow(db_path = db_path, table_name = TABLE_NAME_RAW_DATA_ARRAY)
		replaceRow(db_path = db_path, table_name = TABLE_NAME_RAW_DATA_ARRAY, row_index = row_index, new_row = new_row)

	# Perform PCA on the raw data array to get the projected data array used for plotting
	pca_results = performPCA(raw_data_array = raw_data_array)
	projected_data_array = pca_results["outputs"]["projected_data_array"]

	# Write the projected data array to the db file
	for row_index in range(n_rows):
		# Get the new row as float values
		new_row = [float(value) for value in projected_data_array[row_index, :]]
		# Write to the db file
		appendRow(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY)
		replaceRow(db_path = db_path, table_name = TABLE_NAME_PROJECTED_DATA_ARRAY, row_index = row_index, new_row = new_row)
	
	# Compute the pairwise distances between each data point
	distance_array = zeros((n_rows, n_rows), dtype = float)
	for index_1 in range(n_rows - 1):
		for index_2 in range(index_1 + 1, n_rows):
			current_distance = norm(raw_data_array[index_1, :] - raw_data_array[index_2, :])
			distance_array[index_1, index_2] = current_distance
			distance_array[index_2, index_1] = current_distance
	
	# Loop over the data points and compute the needed information
	for row_index in range(n_rows):
		# Compute the weight vector using softmax on the distances
		weight_vector = softmax(-(distance_array[row_index, :] / softmax_distance)**2)
		
		# Compute the needed PCA results
		pca_results = performPCA(raw_data_array = raw_data_array, weight_vector = weight_vector)
		
		# Compute the cumulative percent variances from these results (note: force first and last values to be 0 and 100 respectively)
		cumulative_percent_variances = [0.0] + [float(value) for value in cumsum(pca_results["outputs"]["ordered_percent_variances"])]
		cumulative_percent_variances[-1] = 100.0
		
		# Write this information to the db file
		appendRow(db_path = db_path, table_name = TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES)
		replaceRow(db_path = db_path, table_name = TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES, row_index = row_index, new_row = cumulative_percent_variances)
		
	# Return the path of the db file
	return db_path
		

#########################################################################################
### Define a function which estimates the local dimension of each point in a data set ###
#########################################################################################
def estimatePointwiseDimension(db_path:Union[PosixPath, WindowsPath], percent_variance:Any) -> list:
	# Compute the pointwise dimension for each point the data set using the pre-computed db file
	# Verify the inputs
	assert type(db_path) in [PosixPath, WindowsPath], "estimatePointwiseDimension: Provided value for 'db_path' must be a PosixPath or WindowsPath object"
	assert isNumeric(percent_variance, include_numpy_flag = True) == True, "estimatePointwiseDimension: Provided value for 'percent_variance' must be numeric"
	assert 0 <= percent_variance and percent_variance <= 100, "estimatePointwiseDimension: Provided value for 'percent_variance' must be >= 0 and <= 100"

	# Get the number of rows and columns in the raw data
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]
	
	# Create the y-values for the linear splines
	y_values = [index for index in range(n_cols + 1)]
	
	# Initialize the list of results
	dimension_results = []
	
	# Compute the dimension estimates for each point
	for row_index in range(n_rows):
		# Read the needed row from the db file as the x-values
		x_values = readRow(db_path = db_path, table_name = TABLE_NAME_CUMULATIVE_PERCENT_VARIANCES, row_index = row_index)
		
		# Create a linear spline using these x-values and y-values
		linear_spline = LinearSpline(x_values = x_values, y_values = y_values)
		
		# Append the current dimension estimate
		dimension_results.append(linear_spline.evaluate(x_value = percent_variance))
	
	# Return the results
	return dimension_results
	
	
#############################################################################################
### Define a function for generating a visualization of the pointwise dimension estimates ###
#############################################################################################
def visualizePointwiseEstimate(db_path:Union[PosixPath, WindowsPath], percent_variance:Any, used_engine:str = "matplotlib"):
	# Generate a scatter plot representing the pointwise dimension estimates
	# Verify the inputs
	assert type(db_path) in [PosixPath, WindowsPath], "visualizePointwiseEstimate: Provided value for 'db_path' must be a PosixPath or WindowsPath object"
	assert isNumeric(percent_variance, include_numpy_flag = True) == True, "visualizePointwiseEstimate: Provided value for 'percent_variance' must be numeric"
	assert 0 <= percent_variance and percent_variance <= 100, "visualizePointwiseEstimate: Provided value for 'percent_variance' must be >= 0 and <= 100"
	assert used_engine in ["matplotlib", "plotly"], "visualizePointwiseEstimate: Provided value for 'used_engine' must be 'matplotlib' or 'plotly'"

	# Get the number of rows and columns in the raw data
	read_row = readRow(db_path = db_path, table_name = TABLE_NAME_INPUT_SETTINGS, row_index = 0)
	n_rows = read_row[0]
	n_cols = read_row[1]

	# Estimate the pointwise dimension for each point at the needed percent variance
	dimension_results = estimatePointwiseDimension(db_path = db_path, percent_variance = percent_variance)

	# Load the projected data array from the db file

	# Create a scatter plot to visualize the pointwise dimension
	if used_engine == "matplotlib":
		pass
	else:
		pass


from numpy import random
raw_data_array = random.rand(50, 10)
db_path = generateDimensionDatabase(raw_data_array = raw_data_array, softmax_distance = 1)
print(estimatePointwiseDimension(db_path = db_path, percent_variance = 0))
print(estimatePointwiseDimension(db_path = db_path, percent_variance = 25))
print(estimatePointwiseDimension(db_path = db_path, percent_variance = 50))
print(estimatePointwiseDimension(db_path = db_path, percent_variance = 75))
print(estimatePointwiseDimension(db_path = db_path, percent_variance = 100))
visualizePointwiseEstimate(db_path = db_path, percent_variance = 50)

