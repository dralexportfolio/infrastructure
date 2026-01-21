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
board_decorator = private_attributes_dec("_all_bevel_info_flag",		# class variables
										 "_all_deepcopy_polygons",
										 "_all_sun_info_flag",
										 "_n_polygons",
										 "_render_axis",
										 "_render_figure",
										 "_x_lower",
										 "_x_shift_per_polygon",
										 "_x_upper",
										 "_y_lower",
										 "_y_shift_per_polygon",
										 "_y_upper",
										 "_checkFlags",					# internal functions
										 "_computeBounds")

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

		# Create a list of deep copies of the provided polygons
		all_deepcopy_polygons = [polygon.deepcopy() for polygon in all_polygons]

		# Store the provided values
		self._n_polygons = n_polygons
		self._all_deepcopy_polygons = all_deepcopy_polygons
		self._x_shift_per_polygon = x_shift_per_polygon
		self._y_shift_per_polygon = y_shift_per_polygon

		# Determine if all bevel and sun information of all polygons have already been preprocessed
		self._checkFlags()

		# Compute the lower and upper x-values and y-values for the board
		self._computeBounds()

	### Define internal function for checking polygon flags and computing board bounds ###
	def _checkFlags(self):
		# Check the preprocess flags of each polygon and determine if they are all True
		# Initialize the flags to True
		self._all_bevel_info_flag = True
		self._all_sun_info_flag = True

		# Loop over the polygons and update to False if not preprocessed
		for polygon in self._all_deepcopy_polygons:
			polygon_info = polygon.getInfo()
			if polygon_info["preprocess_bevel_flag"] == False:
				self._all_bevel_info_flag = False
			if polygon_info["preprocess_sun_flag"] == False:
				self._all_sun_info_flag = False

	def _computeBounds(self):
		# Compute the render bounds for the board by looping over the polygons
		# Initialize the bounds
		self._x_lower = float("inf")
		self._x_upper = -float("inf")
		self._y_lower = float("inf")
		self._y_upper = float("inf")

		# Loop over the polygons and extract the needed information
		for polygon in self._all_deepcopy_polygons:
			polygon_info = polygon.getInfo()
			self._x_lower = min(self._x_lower, polygon_info["x_lower"])
			self._x_upper = max(self._x_upper, polygon_info["x_upper"])
			self._y_lower = min(self._y_lower, polygon_info["y_lower"])
			self._y_upper = max(self._y_upper, polygon_info["y_upper"])

	### Define functions for preprocessing bevel and sun information for all polygons ###
	def preprocessBevelInfo(self, bevel_attitude:Any, bevel_size:Any, polygon_index:int = None):
		# Preprocess all information related to the bevel for all polygons (or a specific one if index is provided)
		# Verify the polygon index (and leave the other inputs to be verified by the polygons themselves)
		if polygon_index is not None:
			assert type(polygon_index) == int, "Board::preprocessBevelInfo: If provided, value for 'polygon_index' must be an int object"
			assert 0 <= polygon_index and polygon_index < self._n_polygons, "Board::preprocessBevelInfo: If provided, value for 'polygon_index' must be non-negative and less than the number of polygons on the board (i.e. " + str(self._n_polygons) + ")"

		# Handle the various cases
		if polygon_index is not None:
			# Update the information for the given polygon
			self._all_deepcopy_polygons[polygon_index].preprocessBevelInfo(bevel_attitude = bevel_attitude, bevel_size = bevel_size)
		else:
			# Update the information for all polygons
			for polygon_index in range(self._n_polygons):
				self._all_deepcopy_polygons[polygon_index].preprocessBevelInfo(bevel_attitude = bevel_attitude, bevel_size = bevel_size)

		# Determine if all bevel and sun information of all polygons have already been preprocessed
		self._checkFlags()

	def preprocessAllSunInfo(self):
	# Preprocess all information related to the sun for all polygons (or a specific one if index is provided)
		pass

from Polygon import SQUARE_1x1

n_polygons = 4
all_polygons = [SQUARE_1x1, SQUARE_1x1, SQUARE_1x1, SQUARE_1x1]

bevel_attitude = 25
bevel_size = 0.1
all_polygons[0].preprocessBevelInfo(bevel_attitude = bevel_attitude, bevel_size = bevel_size)
print(all_polygons[1].getInfo())

x_shift_per_polygon = [0, 1, 0, 1]
y_shift_per_polygon = [0, 0, 1, 1]
board = Board(n_polygons = n_polygons, all_polygons = all_polygons, x_shift_per_polygon = x_shift_per_polygon, y_shift_per_polygon = y_shift_per_polygon)