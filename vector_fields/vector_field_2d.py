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


###################################################
### Define the 2-dimensional vector field class ###
###################################################
# Create the decorator needed for making the attributes private
vector_field_2d_decorator = private_attributes_dec("_n_cols",			# class variables
												   "_n_rows")

# Define the class with private attributes
class VectorField2D:
	### Initialize the class ###
	def __init__(self, n_rows:int, n_cols:int):
		# Verify the inputs
		assert type(n_rows) == int, "VectorField2D::__init__: Provided value for 'n_rows' must be an int object"
		assert type(n_cols) == int, "VectorField2D::__init__: Provided value for 'n_cols' must be an int object"
		
		# Store the provided values
		self._n_rows = n_rows
		self._n_cols = n_cols
