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
from color_helper import RGB
from Polygon import Polygon
from type_helper import isListWithNumericEntries, isNumeric

# External modules
from PrivateAttributesDecorator import private_attributes_dec
from typing import Any


############################################################
### Define the board class to render a group of polygons ###
############################################################
# Create the decorator needed for making the attributes private
board_decorator = private_attributes_dec("_all_polygons",				# class variables
										 "_bevel_attitude",
										 "_bevel_size",
										 "_n_polygons",
										 "_sun_angle",
										 "_sun_attitude",
										 "_x_shift_per_polygon",
										 "_y_shift_per_polygon",
										 "_processInputs")				# internal functions

# Define the class with private attributes
@board_decorator
class Board:
	### Initialize the class ###
	def __init__(self, n_polygons:int, all_polygons:list, x_shift_per_polygon:list, y_shift_per_polygon:list):
		# Verify the inputs
		assert type(n_polygons) == int, "Board::__init__: Provided value for 'n_polygons' must be an int object"
		assert n_polygons > 0, "Board::__init__: Provided value for 'n_polygons' must be positive"
		assert type(all_polygons) == list, "Board::__init__: Provided value for 'all_polygons' must be a list object"
		assert isListWithNumericEntries(x_shift_per_polygon, include_numpy_flag = True) == True, "Board::__init__: Provided value for 'x_shift_per_polygon' must be a list of numeric objects"
		assert isListWithNumericEntries(y_shift_per_polygon, include_numpy_flag = True) == True, "Board::__init__: Provided value for 'y_shift_per_polygon' must be a list of numeric objects"
		assert len(all_polygons) == n_polygons, "Board::__init__: Provided value for 'all_polygons' must be a list of length 'n_polygons'"
		assert len(x_shift_per_polygon) == n_polygons, "Board::__init__: Provided value for 'x_shift_per_polygon' must be a list of length 'n_polygons'"
		assert len(y_shift_per_polygon) == n_polygons, "Board::__init__: Provided value for 'y_shift_per_polygon' must be a list of length 'n_polygons'"
		for index in range(n_polygons):
			assert type(all_polygons[index]) == Polygon, "Board::__init__: Provided value for 'all_polygons' must be a list of Polygon objects"
			assert -float("inf") < x_shift_per_polygon[index] and x_shift_per_polygon[index] < float("inf"), "Board::__init__: Entries in provided value for 'x_shift_per_polygon' must be finite"
			assert -float("inf") < y_shift_per_polygon[index] and y_shift_per_polygon[index] < float("inf"), "Board::__init__: Entries in provided value for 'y_shift_per_polygon' must be finite"

		# Store the provided values
		self._n_polygons = n_polygons
		self._all_polygons = all_polygons
		self._x_shift_per_polygon = x_shift_per_polygon
		self._y_shift_per_polygon = y_shift_per_polygon

from Polygon import SQUARE_1x1
n_polygons = 4
all_polygons = [SQUARE_1x1, SQUARE_1x1, SQUARE_1x1, SQUARE_1x1]
x_shift_per_polygon = [0, 1, 0, 1]
y_shift_per_polygon = [0, 0, 1, 1]
board = Board(n_polygons = n_polygons, all_polygons = all_polygons, x_shift_per_polygon = x_shift_per_polygon, y_shift_per_polygon = y_shift_per_polygon)