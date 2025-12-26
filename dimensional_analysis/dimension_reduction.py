##########################################
### Import needed general dependencies ###
##########################################
# External modules
from numpy import argsort, matmul, mean, ndarray, ones, std, zeros
from numpy import min as np_max
from numpy import min as np_min
from numpy.linalg import eig


#####################################################################
### Define a function for performing principal component analysis ###
#####################################################################
def performPCA(raw_data_array:ndarray, weight_vector:ndarray = None) -> dict:
	# Compute information relevant to PCA analysis on the given raw data
	# Verify the inputs
	assert type(raw_data_array) == ndarray, "performPCA: Provided value for 'raw_data_array' must be a numpy.ndarry object"
	assert len(raw_data_array.shape) == 2, "performPCA: Provided value for 'raw_data_array' must be a 2-dimensional numpy array"
	assert raw_data_array.shape[0] > 0, "performPCA: Provided value for 'raw_data_array' must have a non-zero number of points, i.e. at least 1 row"
	assert raw_data_array.shape[1] > 0, "performPCA: Provided value for 'raw_data_array' must have a non-zero number of features, i.e. at least 1 column"
	if weight_vector is not None:
		assert type(weight_vector) == ndarray, "performPCA: If provided, value for 'weight_vector' must be a numpy.ndarry object"
		assert len(weight_vector.shape) == 1, "performPCA: If provided, value for 'weight_vector' must be a 1-dimensional numpy array"
		assert weight_vector.shape[0] == raw_data_array.shape[0], "performPCA: If provided, value for 'weight_vector' must be have length equal to number of rows in 'raw_data_array'"
		assert np_min(weight_vector) >= 0, "performPCA: If provided, value for 'weight_vector' must have all non-negative entry"
		assert np_max(weight_vector) > 0, "performPCA: If provided, value for 'weight_vector' must have at least one positive entry"
	
	# Extract the numbers of rows and columns in the data
	n_rows = raw_data_array.shape[0]
	n_cols = raw_data_array.shape[1]
	
	# Set the weight vector to be all ones if not provided
	if weight_vector is None:
		weight_vector = ones(n_rows, dtype = float)
	
	# Construct the weight array to use moving forward
	weight_array = zeros((n_rows, n_rows), dtype = float)
	for index in range(n_rows):
		weight_array[index, index] = weight_vector[index]
			
	# Weight each row of the data by the needed amount and compute the weighted means and deviations
	weighted_data_array = matmul(weight_array, raw_data_array)
	weighted_data_means = mean(weighted_data_array, axis = 0)
	weighted_data_deviations = std(weighted_data_array, axis = 0)
	
	# Center the data, compute the weighted deviations and normalize (will be n_rows x n_cols)
	normalized_data_array = (weighted_data_array - weighted_data_means) / weighted_data_deviations
	
	# Compute the covariance matrix of the normalized data (will be n_cols x n_cols)
	covariance_array = matmul(normalized_data_array.T, matmul(weight_array, normalized_data_array)) / sum(weight_vector)
	
	# Compute the eigenvalues and eigenvectors of the covariance matrix (will be n_cols and n_cols x n_cols respectively)
	covariance_eigenvalues, covariance_eigenvectors = eig(covariance_array)
	
	# Get the indices in the order for which the covariance eigenvalues will decrease
	decreasing_index_order = list(argsort(covariance_eigenvalues))
	decreasing_index_order.reverse()
	
	# Reorder the eigenvalue and eigenvectors to be in this decreasing order
	# Initialize the needed storage
	ordered_covariance_eigenvalues = zeros(covariance_eigenvalues.shape, dtype = covariance_eigenvalues.dtype)
	ordered_covariance_eigenvectors = zeros(covariance_eigenvectors.shape, dtype = covariance_eigenvectors.dtype)
	# Insert the values in the needed order
	for col_index in range(n_cols):
		ordered_covariance_eigenvalues[col_index] = covariance_eigenvalues[decreasing_index_order[col_index]]
		ordered_covariance_eigenvectors[:, col_index] = covariance_eigenvectors[:, decreasing_index_order[col_index]]
	
	# Nomalize the eigenvalues to get the percent variances in decreasing variance order
	ordered_percent_variances = 100 * ordered_covariance_eigenvalues / sum(ordered_covariance_eigenvalues)
	
	# Project the normalized data onto the principal directions in decreasing variance order  (will be n_rows x n_cols)
	projected_data_array = matmul(normalized_data_array, ordered_covariance_eigenvectors)
	
	# Construct a dictionary of relevant results
	pca_results = {"inputs": {}, "outputs": {}}
	pca_results["inputs"]["raw_data_array"] = raw_data_array
	pca_results["inputs"]["weight_vector"] = weight_vector
	pca_results["outputs"]["weighted_data_means"] = weighted_data_means
	pca_results["outputs"]["weighted_data_deviations"] = weighted_data_deviations
	pca_results["outputs"]["normalized_data_array"] = normalized_data_array
	pca_results["outputs"]["covariance_array"] = covariance_array
	pca_results["outputs"]["ordered_covariance_eigenvalues"] = ordered_covariance_eigenvalues
	pca_results["outputs"]["ordered_covariance_eigenvectors"] = ordered_covariance_eigenvectors
	pca_results["outputs"]["ordered_percent_variances"] = ordered_percent_variances
	pca_results["outputs"]["projected_data_array"] = projected_data_array
	
	# Return the results
	return pca_results
