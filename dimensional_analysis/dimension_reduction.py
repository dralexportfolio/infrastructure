##########################################
### Import needed general dependencies ###
##########################################
# External modules
from numpy import argsort, matmul, mean, ndarray, ones, std, zeros
from numpy import min as np_max
from numpy import min as np_min
from numpy.linalg import svd
from scipy.sparse import diags


#####################################################################
### Define a function for performing principal component analysis ###
#####################################################################
def performPCA(raw_data_array:ndarray, normalize_flag:bool = False, center_vector:ndarray = None, weight_vector:ndarray = None) -> dict:
	# Compute information relevant to PCA analysis on the given raw data
	# Verify the inputs
	assert type(raw_data_array) == ndarray, "performPCA: Provided value for 'raw_data_array' must be a numpy.ndarry object"
	assert len(raw_data_array.shape) == 2, "performPCA: Provided value for 'raw_data_array' must be a 2-dimensional numpy array"
	assert raw_data_array.shape[0] > 0, "performPCA: Provided value for 'raw_data_array' must have a non-zero number of points, i.e. at least 1 row"
	assert raw_data_array.shape[1] > 0, "performPCA: Provided value for 'raw_data_array' must have a non-zero number of features, i.e. at least 1 column"
	assert type(normalize_flag) == bool, "performPCA: Provided value for 'normalize_flag' must be a bool object"
	if center_vector is not None:
		assert type(center_vector) == ndarray, "performPCA: If provided, value for 'center_vector' must be a numpy.ndarry object"
		assert len(center_vector.shape) == 1, "performPCA: If provided, value for 'center_vector' must be a 1-dimensional numpy array"
		assert center_vector.shape[1] == raw_data_array.shape[1], "performPCA: If provided, value for 'center_vector' must be have length equal to number of columns in 'raw_data_array'"
		assert  -float("inf") < np_min(center_vector) and np_max(center_vector) < float("inf"), "performPCA: If provided, value for 'center_vector' must have all finite entries"
	if weight_vector is not None:
		assert type(weight_vector) == ndarray, "performPCA: If provided, value for 'weight_vector' must be a numpy.ndarry object"
		assert len(weight_vector.shape) == 1, "performPCA: If provided, value for 'weight_vector' must be a 1-dimensional numpy array"
		assert weight_vector.shape[0] == raw_data_array.shape[0], "performPCA: If provided, value for 'weight_vector' must be have length equal to number of rows in 'raw_data_array'"
		assert np_min(weight_vector) >= 0, "performPCA: If provided, value for 'weight_vector' must have all non-negative entries"
		assert np_max(weight_vector) > 0, "performPCA: If provided, value for 'weight_vector' must have at least one positive entry"
	
	# Extract the numbers of rows and columns in the data
	n_rows = raw_data_array.shape[0]
	n_cols = raw_data_array.shape[1]

	# Compute the center vector as the center of mass of the data (if needed)
	if center_vector is None:
		center_vector = mean(raw_data_array, axis = 0)

	# Set the weight vector to be an array of all 1's (if needed)
	if weight_vector is None:
		weight_vector = ones(n_rows, dtype = float)

	# Create the square-root of the weight matrix as a sparse array (will be n_rows x n_rows)
	sqrt_weight_array = diags(weight_vector**0.5)

	# Shift the raw data so that the center is at the origin (will be n_rows x n_cols)
	shifted_data_array = zeros((n_rows, n_cols), dtype = float)
	for row_index in range(n_rows):
		shifted_data_array[row_index, :] = raw_data_array[row_index, :] - center_vector

	# Normalize the columns of the shifted data to have the same deviations (if needed) (will be n_rows x n_cols)
	if normalize_flag == False:
		# Don't normalize the array
		normalized_data_array = shifted_data_array
	else:
		# Divide each column by the deviation of that column (if it is non-zero, otherwise will all be 0 because of shift)
		normalized_data_array = zeros((n_rows, n_cols), dtype = float)
		all_deviations = std(shifted_data_array, axis = 0)
		for col_index in range(n_cols):
			if all_deviations[col_index] > 0:
				normalized_data_array[:, col_index] = shifted_data_array[:, col_index] / all_deviations[col_index]

	# Weight the normalized data points according to the square-root weight array (will be n_rows x n_cols)
	# Note: numpy.matmul doesn't work with sparse matrices so use @ instead
	weighted_data_array = sqrt_weight_array @ normalized_data_array

	# Perform Thin SVD (not Full SVD) on the weighted data array
	u_array, s_array, v_array_transpose = svd(weighted_data_array, full_matrices = False)

	# Get the indices in the order for which the singular values will decrease
	decreasing_index_order = list(argsort(s_array))
	decreasing_index_order.reverse()

	# Reorder the singular values and principal component vectors to match this decreasing order
	# Initialize the needed storage
	ordered_singular_values = zeros(n_cols, dtype = float)
	ordered_principal_components = zeros((n_cols, n_cols), dtype = float)
	# Insert the values in the needed order
	for col_index in range(n_cols):
		ordered_singular_values[col_index] = s_array[decreasing_index_order[col_index]]
		ordered_principal_components[:, col_index] = v_array_transpose[decreasing_index_order[col_index], :]

	# Convert the ordered singular values stored to percent variances
	ordered_percent_variances = 100 * ordered_singular_values**2 / sum(ordered_singular_values**2)
	
	# Project the normalized data onto the principal directions in decreasing variance order (will be n_rows x n_cols)
	projected_data_array = matmul(normalized_data_array, ordered_principal_components)
	
	# Construct a dictionary of relevant results
	pca_results = {"inputs": {}, "outputs": {}}
	pca_results["inputs"]["raw_data_array"] = raw_data_array
	pca_results["inputs"]["normalize_flag"] = normalize_flag
	pca_results["inputs"]["center_vector"] = center_vector
	pca_results["inputs"]["weight_vector"] = weight_vector
	pca_results["outputs"]["normalized_data_array"] = normalized_data_array
	pca_results["outputs"]["ordered_singular_values"] = ordered_singular_values
	pca_results["outputs"]["ordered_principal_components"] = ordered_principal_components
	pca_results["outputs"]["ordered_percent_variances"] = ordered_percent_variances
	pca_results["outputs"]["projected_data_array"] = projected_data_array
	
	# Return the results
	return pca_results
