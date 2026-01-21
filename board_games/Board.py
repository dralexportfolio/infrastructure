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
from io import BytesIO
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from numpy import zeros
from PIL import Image
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

	### Define internal functions for checking polygon flags and computing board bounds ###
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
		self._y_upper = -float("inf")

		# Loop over the polygons and extract the needed information
		for polygon_index in range(self._n_polygons):
			polygon_info = self._all_deepcopy_polygons[polygon_index].getInfo()
			self._x_lower = min(self._x_lower, polygon_info["x_lower"] + self._x_shift_per_polygon[polygon_index])
			self._x_upper = max(self._x_upper, polygon_info["x_upper"] + self._x_shift_per_polygon[polygon_index])
			self._y_lower = min(self._y_lower, polygon_info["y_lower"] + self._y_shift_per_polygon[polygon_index])
			self._y_upper = max(self._y_upper, polygon_info["y_upper"] + self._y_shift_per_polygon[polygon_index])

	### Define external functions for preprocessing bevel and sun information for all polygons ###
	def preprocessBevelInfo(self, bevel_attitude:Any, bevel_size:Any, polygon_index:int = None):
		# Preprocess all information related to the bevel for all polygons (or a specific one if index is provided)
		# Verify the polygon index (and leave the other inputs to be verified by the polygons themselves)
		if polygon_index is not None:
			assert type(polygon_index) == int, "Board::preprocessBevelInfo: If provided, value for 'polygon_index' must be an int object"
			assert 0 <= polygon_index and polygon_index < self._n_polygons, "Board::preprocessBevelInfo: If provided, value for 'polygon_index' must be non-negative and less than the number of polygons on the board (i.e. " + str(self._n_polygons) + ")"

		# Set the indices to loop over
		if polygon_index is not None:
			needed_indices = range(polygon_index, polygon_index + 1)
		else:
			needed_indices = range(self._n_polygons)

		# Update the information for the needed polygons
		for index in needed_indices:
			self._all_deepcopy_polygons[index].preprocessBevelInfo(bevel_attitude = bevel_attitude, bevel_size = bevel_size)

		# Determine if all bevel and sun information of all polygons have already been preprocessed
		self._checkFlags()

	def preprocessAllSunInfo(self, sun_angle:Any, sun_attitude:Any, polygon_index:int = None):
		# Preprocess all information related to the sun for all polygons (or a specific one if index is provided)
		# Verify the polygon index (and leave the other inputs to be verified by the polygons themselves)
		if polygon_index is not None:
			assert type(polygon_index) == int, "Board::preprocessSunInfo: If provided, value for 'polygon_index' must be an int object"
			assert 0 <= polygon_index and polygon_index < self._n_polygons, "Board::preprocessSunInfo: If provided, value for 'polygon_index' must be non-negative and less than the number of polygons on the board (i.e. " + str(self._n_polygons) + ")"

		# Set the indices to loop over
		if polygon_index is not None:
			needed_indices = range(polygon_index, polygon_index + 1)
		else:
			needed_indices = range(self._n_polygons)

		# Update the information for the needed polygons
		for index in needed_indices:
			self._all_deepcopy_polygons[index].preprocessSunInfo(sun_angle = sun_angle, sun_attitude = sun_attitude)

		# Determine if all bevel and sun information of all polygons have already been preprocessed
		self._checkFlags()

	### Define an external function for rendering the board as a single image ###
	def render(self, dpi:int, min_brightness:Any = 0, max_brightness:Any = 1, tint_shade:RGB = RGB((255, 255, 255))) -> Image.Image:
		# Return a PIL image render of the board with the preprocessed settings
		# Only proceed if all bevel and sun information has been preprocessed
		assert self._all_bevel_info_flag == True, "Board::render: Only able to render board image once all bevel information has been preprocessed"
		assert self._all_sun_info_flag == True, "Board::render: Only able to render board image once all sun information has been preprocessed"

		# Verify the inputs which won't be handled by the polygon's computeRenderInfo method
		assert type(dpi) == int, "Board::render: Provided value for 'dpi' must be an int object"
		assert 72 <= dpi and dpi <= 900, "Board::render: Provided value for 'dpi' must be >= 72 and <= 900"

		# Create the figure and axis to which to render, crop it to the needed size (in normalized figure coordinates), and adjust the axis as needed
		self._render_figure, self._render_axis = plt.subplots(figsize = (self._x_upper - self._x_lower, self._y_upper - self._y_lower))
		self._render_figure.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, wspace = 0, hspace = 0)
		self._render_axis.set_xlim(left = self._x_lower, right = self._x_upper)
		self._render_axis.set_ylim(bottom = self._y_lower, top = self._y_upper)
		self._render_axis.axis("off")

		# Loop over the polygons and render each associated patch
		for polygon_index in range(self._n_polygons):
			# Perform additional input verification and compute the render information using computeRenderInfo with the needed shift
			render_info = self._all_deepcopy_polygons[polygon_index].computeRenderInfo(min_brightness = min_brightness,
																					   max_brightness = max_brightness,
																					   tint_shade = tint_shade,
																					   x_shift = self._x_shift_per_polygon[polygon_index],
																					   y_shift = self._y_shift_per_polygon[polygon_index])

			# Extract all needed values
			n_faces = render_info["n_faces"]
			n_edges_per_face = render_info["n_edges_per_face"]
			x_values_per_face = render_info["x_values_per_face"]
			y_values_per_face = render_info["y_values_per_face"]
			rgb_values_per_face = render_info["rgb_values_per_face"]

			# Loop over the faces of this polygon and create the needed patches
			for face_index in range(n_faces):
				# Create an array containing the vertices of this face
				vertex_array = zeros((n_edges_per_face[face_index], 2), dtype = float)
				for row_index in range(n_edges_per_face[face_index]):
					vertex_array[row_index, 0] = x_values_per_face[face_index][row_index]
					vertex_array[row_index, 1] = y_values_per_face[face_index][row_index]

				# Create the needed patch in the given color and add it to the plot
				self._render_axis.add_patch(patches.Polygon(vertex_array, closed = True, edgecolor = None, facecolor = rgb_values_per_face[face_index], linewidth = 0))

		# Redraw the canvas now that all faces have been updated
		self._render_figure.canvas.draw_idle()

		# Create a buffer to which the image will be saved
		image_buffer = BytesIO()

		# Save the figure to the buffer
		self._render_figure.savefig(image_buffer, dpi = dpi, format = "png", transparent = True)

		# Rewind to the beginning of the buffer and load the render as a PIL image
		image_buffer.seek(0)
		board_render = Image.open(image_buffer)

		# Return the result
		return board_render



from Polygon import SQUARE_1x1

n_rows = 8
n_cols = 8

n_polygons = n_rows * n_cols
all_polygons = [SQUARE_1x1 for _ in range(n_polygons)]

x_shift_per_polygon = []
y_shift_per_polygon = []
x_shift = 0
y_shift = 0
while True:
	x_shift_per_polygon.append(x_shift)
	y_shift_per_polygon.append(y_shift)

	x_shift += 1
	if x_shift == n_cols:
		x_shift = 0
		y_shift += 1
		if y_shift == n_rows:
			break

bevel_attitude = 25
bevel_size = 0.1
all_polygons[0].preprocessBevelInfo(bevel_attitude = bevel_attitude, bevel_size = bevel_size)
print(all_polygons[1].getInfo())

board = Board(n_polygons = n_polygons, all_polygons = all_polygons, x_shift_per_polygon = x_shift_per_polygon, y_shift_per_polygon = y_shift_per_polygon)

board.preprocessBevelInfo(bevel_attitude = 25, bevel_size = 0.1)
board.preprocessAllSunInfo(sun_angle = 120, sun_attitude = 35)
board.render(dpi = 600, tint_shade = RGB((255, 0, 255))).show()