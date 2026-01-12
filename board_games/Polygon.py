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
from type_helper import isListWithNumericEntries, isNumeric, tolerantlyCompare

# External modules
from math import acos, pi
from numpy import array, matmul
from numpy.linalg import det, inv, norm
from PrivateAttributesDecorator import private_attributes_dec
from typing import Any


####################################################################
### Define the polygon class as a basis for commonly used shapes ###
####################################################################
# Create the decorator needed for making the attributes private
polygon_decorator = private_attributes_dec("_bevel_attitude"				# class variables
										   "_bevel_size",
										   "_n_edges_per_face",
										   "_n_vertices",
										   "_normal_vector_per_edge",
										   "_normal_vector_per_face",
										   "_patch_per_face",
										   "_preprocess_bevel_flag",
										   "_preprocess_sun_flag",
										   "_raw_brightness_per_face",
										   "_render_axis",
										   "_render_figure",
										   "_sun_angle",
										   "_sun_attitude",
										   "_x_midpoint_per_edge",
										   "_x_value_per_vertex",
										   "_x_values_per_face",
										   "_y_midpoint_per_edge",
										   "_y_value_per_vertex",
										   "_y_values_per_face",
										   "_processInputs")				# internal functions

# Define the class with private attributes
@polygon_decorator
class Polygon:
	### Initialize the class ###
	def __init__(self, n_vertices:int, x_value_per_vertex:list, y_value_per_vertex:list):
		# Verify and store the inputs
		self._processInputs(n_vertices = n_vertices, x_value_per_vertex = x_value_per_vertex, y_value_per_vertex = y_value_per_vertex)

		# Compute the midpoints and externally facing unit normal vectors of each edge
		# Initialize the needed lists
		self._x_midpoint_per_edge = []
		self._y_midpoint_per_edge = []
		self._normal_vector_per_edge = []
		# Compute the midpoint values
		for index in range(self._n_vertices):
			# Compute the midpoint coordinates
			self._x_midpoint_per_edge.append((self._x_value_per_vertex[index] + self._x_value_per_vertex[(index + 1) % self._n_vertices]) / 2)
			self._y_midpoint_per_edge.append((self._y_value_per_vertex[index] + self._y_value_per_vertex[(index + 1) % self._n_vertices]) / 2)
			# Compute the externally facing unit normal vector
			x_component = self._y_value_per_vertex[(index + 1) % self._n_vertices] - self._y_value_per_vertex[index]
			y_component = self._x_value_per_vertex[index] - self._x_value_per_vertex[(index + 1) % self._n_vertices]
			normal_vector = array([[x_component], [y_component]], dtype = float64)
			self._normal_vector_per_edge.append(normal_vector / norm(normal_vector))

		# Initialize flags indicating the preprocess state of the class
		self._preprocess_bevel_flag = False
		self._preprocess_sun_flag = False

		# Initialize all other variables which will be used later in the class
		self._bevel_attitude = None
		self._bevel_size = None
		self._n_edges_per_face = None
		self._normal_vector_per_face = None
		self._patch_per_face = None
		self._raw_brightness_per_face = None
		self._render_axis = None
		self._render_figure = None
		self._sun_angle = None
		self._sun_attitude = None
		self._x_values_per_face = None
		self._y_values_per_face = None

	### Define an internal function to verify and store that the provided inputs values are valid ###
	def _processInputs(self, n_vertices:int, x_value_per_vertex:list, y_value_per_vertex:list):
		# Verify the inputs provided upon class initialization
		# Check the types and list lengths
		assert type(n_vertices) == int, "Polygon::_processInputs: Provided value for 'n_vertices' must be an int object"
		assert n_vertices >= 3, "Polygon::_processInputs: Provided value for 'n_vertices' must be >= 3"
		assert isListWithNumericEntries(x_value_per_vertex, include_numpy_flag = True) == True, "Polygon::_processInputs: Provided value for 'x_value_per_vertex' must be a list object containing numeric entries"
		assert len(x_value_per_vertex) == n_vertices, "Polygon::_processInputs: Provided value for 'x_value_per_vertex' must be a list of length equal to 'n_vertices'"
		assert -float("inf") < min(x_value_per_vertex) and max(x_value_per_vertex) < float("inf"), "Polygon::_processInputs: Provided value for 'x_value_per_vertex' must be a list of all finite entries"
		assert isListWithNumericEntries(y_value_per_vertex, include_numpy_flag = True) == True, "Polygon::_processInputs: Provided value for 'y_value_per_vertex' must be a list object containing numeric entries"
		assert len(y_value_per_vertex) == n_vertices, "Polygon::_processInputs: Provided value for 'y_value_per_vertex' must be a list of length equal to 'n_vertices'"
		assert -float("inf") < min(y_value_per_vertex) and max(y_value_per_vertex) < float("inf"), "Polygon::_processInputs: Provided value for 'y_value_per_vertex' must be a list of all finite entries"

		# Make sure that all vertex value pairs are distinct
		for index_1 in range(n_vertices - 1):
			for index_2 in range(index_1 + 1, n_vertices):
				delta_x = x_value_per_vertex[index_2] - x_value_per_vertex[index_1]
				delta_y = y_value_per_vertex[index_2] - y_value_per_vertex[index_1]
				assert delta_x**2 + delta_y**2 > 0, "Polygon::_processInputs: Provided values for 'x_value_per_vertex' and 'y_value_per_vertex' must represent distinct points in the plane"

		# Make sure that none of the line segments defining the edges intersect each other
		for index_1 in range(n_vertices - 1):
			# Get the vector defining the 1st edge
			delta_x_1 = x_value_per_vertex[index_1 + 1] - x_value_per_vertex[index_1]
			delta_y_1 = y_value_per_vertex[index_1 + 1] - y_value_per_vertex[index_1]
			direction_1 = array([[delta_x_1], [delta_y_1]], dtype = float)

			# Loop over the other edges
			for index_2 in range(index_1 + 1, n_vertices):
				# Get the vector defining the 2nd edge
				delta_x_2 = x_value_per_vertex[(index_2 + 1) % n_vertices] - x_value_per_vertex[index_2]
				delta_y_2 = y_value_per_vertex[(index_2 + 1) % n_vertices] - y_value_per_vertex[index_2]
				direction_2 = array([[delta_x_2], [delta_y_2]], dtype = float)

				# Create the matrix and vector needed to solve the relevant system
				needed_matrix = array([[direction_1[0, 0], -direction_2[0, 0]], [direction_1[1, 0], -direction_2[1, 0]]], dtype = float)
				needed_vector = array([[x_value_per_vertex[index_2] - x_value_per_vertex[index_1]], [y_value_per_vertex[index_2] - y_value_per_vertex[index_1]]], dtype = float)

				# Handle the various cases
				if tolerantlyCompare(det(needed_matrix), "!=", 0):
					# Solve the needed system and determine if error needs to be raised
					needed_solution = matmul(inv(needed_matrix), needed_vector)
					condition_1 = tolerantlyCompare(needed_solution[0, 0], "<=", 0)
					condition_2 = tolerantlyCompare(1, "<=", needed_solution[0, 0])
					condition_3 = tolerantlyCompare(needed_solution[1, 0], "<=", 0)
					condition_4 = tolerantlyCompare(1, "<=", needed_solution[1, 0])
					assert condition_1 or condition_2 or condition_3 or condition_4, "Polygon::_processInputs: Provided values for 'x_value_per_vertex' and 'y_value_per_vertex' must represent a set of line segments which do not intersect each other"
				else:
					# Either or 0 or infinitely many solutions because directions are parallel, make sure there are 0 solutions
					# Create the matrix used to determine if the two edges lie on the same line
					parallelogram_matrix = array([[direction_1[0, 0], needed_vector[0, 0]], [direction_1[1, 0], needed_vector[1, 0]]], dtype = float)
					# Lie on the same line if determinant is 0, if so make sure there is no overlap
					if tolerantlyCompare(det(parallelogram_matrix), "==", 0):
						# Parametrize the line so that the 1st edge is from 0 to 1, then find the bounds for the 2nd edge
						if direction_1[0, 0] != 0:
							start_2 = (x_value_per_vertex[index_2] - x_value_per_vertex[index_1]) / direction_1[0, 0]
							end_2 = start_2 + direction_2[0, 0] / direction_1[0, 0]
						else:
							start_2 = (y_value_per_vertex[index_2] - y_value_per_vertex[index_1]) / direction_1[1, 0]
							end_2 = start_2 + direction_2[1, 0] / direction_1[1, 0]
						# Determine if the line segments overlap and raise error if so
						condition_1 = tolerantlyCompare(start_2, "<=", 0)
						condition_2 = tolerantlyCompare(1, "<=", start_2)
						condition_3 = tolerantlyCompare(end_2, "<=", 0)
						condition_4 = tolerantlyCompare(1, "<=", end_2)
						assert (condition_1 or condition_2) and (condition_3 or condition_4), "Polygon::_processInputs: Provided values for 'x_value_per_vertex' and 'y_value_per_vertex' must represent a set of line segments which do not overlap when they lie on the same line"

		# Cycle through the vertices on the boundary of the polygon and determine the total angle turned, keep order if CCW and reverse if CW
		# Initialize the degree counter
		degree_counter = 0
		# Loop over the vertices and compute the signed degrees turned
		for index in range(n_vertices):
			# Get the vector defining the 1st edge
			delta_x_1 = x_value_per_vertex[index_1 + 1] - x_value_per_vertex[index_1]
			delta_y_1 = y_value_per_vertex[index_1 + 1] - y_value_per_vertex[index_1]
			direction_1 = array([[delta_x_1], [delta_y_1]], dtype = float)
			# Get the vector defining the 2nd edge
			delta_x_2 = x_value_per_vertex[(index_2 + 1) % n_vertices] - x_value_per_vertex[index_2]
			delta_y_2 = y_value_per_vertex[(index_2 + 1) % n_vertices] - y_value_per_vertex[index_2]
			direction_2 = array([[delta_x_2], [delta_y_2]], dtype = float)
			# Normalize the direction vectors
			direction_1 /= norm(direction_1)
			direction_2 /= norm(direction_2)
			# Compute the cosine of the angle using the dot product
			cos_angle = direction_1[0, 0] * direction_2[0, 0] + direction_1[1, 0] * direction_2[1, 0]
			# Create the matrix used to determine the sign of the resulting angle
			sign_matrix = array([[direction_1[0, 0], direction_2[0, 0]], [direction_1[1, 0], direction_2[1, 0]]], dtype = float)
			# Update the degree counter accordingly
			if det(sign_matrix) >= 0:
				degree_counter += acos(cos_angle) * 180 / pi
			else:
				degree_counter -= acos(cos_angle) * 180 / pi
		# Reverse the order of the vertices (if needed)
		if degree_counter < 0:
			x_value_per_vertex.reverse()
			y_value_per_vertex.reverse()

		# Store the provided values
		self._n_vertices = n_vertices
		self._x_value_per_vertex = x_value_per_vertex
		self._y_value_per_vertex = y_value_per_vertex

	### Define external functions which preprocess information for rendering ###
	def preprocessBevelInfo(self, bevel_attitude:Any, bevel_size:Any):
		# Preprocess all information related to the bevel
		# Verify the inputs
		assert isNumeric(bevel_attitude, include_numpy_flag = True) == True, "Polygon::preprocessBevelInfo: Provided value for 'bevel_attitude' must be numeric"
		assert 0 <= bevel_attitude and bevel_attitude <= 75, "Polygon::preprocessBevelInfo: Provided value for 'bevel_attitude' must be >= 0 and <= 75"
		assert isNumeric(bevel_size, include_numpy_flag = True) == True, "Polygon::preprocessBevelInfo: Provided value for 'bevel_size' must be numeric"
		assert bevel_size > 0, "Polygon::preprocessBevelInfo: Provided value for 'bevel_size' must be positive"

		# Store the provided values
		self._bevel_attitude = bevel_attitude
		self._bevel_size = bevel_size

		# Set the preprocess flags to False and forget the previous sun information
		self._preprocess_bevel_flag = False
		self._preprocess_sun_flag = False
		self._sun_angle = None
		self._sun_attitude = None

		# Compute the vertex locations for the interior shape created by the bevel
		# Initialize the needed lists
		x_interior_per_vertex = []
		y_interior_per_vertex = []
		# Compute the needed values
		for index in range(self._n_vertices):
			# Compute the shifted base points for the needed lines
			shifted_x_1 = self._x_value_per_vertex[index - 1] - self._bevel_size * self._normal_vector_per_edge[index - 1][0, 0]
			shifted_y_1 = self._y_value_per_vertex[index - 1] - self._bevel_size * self._normal_vector_per_edge[index - 1][1, 0]
			shifted_x_2 = self._x_value_per_vertex[index] - self._bevel_size * self._normal_vector_per_edge[index][0, 0]
			shifted_y_2 = self._y_value_per_vertex[index] - self._bevel_size * self._normal_vector_per_edge[index][1, 0]
			# Get the vector defining the 1st edge
			delta_x_1 = x_value_per_vertex[index] - x_value_per_vertex[index - 1]
			delta_y_1 = y_value_per_vertex[index] - y_value_per_vertex[index - 1]
			direction_1 = array([[delta_x_1], [delta_y_1]], dtype = float)
			# Get the vector defining the 2nd edge
			delta_x_2 = x_value_per_vertex[(index + 1) % n_vertices] - x_value_per_vertex[index]
			delta_y_2 = y_value_per_vertex[(index + 1) % n_vertices] - y_value_per_vertex[index]
			direction_2 = array([[delta_x_2], [delta_y_2]], dtype = float)
			# Create the matrix and vector needed to solve the system
			needed_matrix = array([[direction_1[0, 0], -direction_2[0, 0]], [direction_1[1, 0], -direction_2[1, 0]]], dtype = float)
			needed_vector = array([[shifted_x_2 - shifted_x_1], [shifted_y_2 - shifted_y_1]], dtype = float)
			# Handle the various cases
			if tolerantlyCompare(det(needed_vector), "!=", 0):
				# Not parallel lines so solve the needed system to get the intersection of the lines
				needed_solution = matmul(inv(needed_matrix), needed_vector)
				x_intersect = shifted_x_2 + direction_2[0, 0] * needed_solution[1, 0]
				y_intersect = shifted_y_2 + direction_2[1, 0] * needed_solution[1, 0]
			else:
				# Parallel lines so just use the shifted base point
				x_intersect = shifted_x_2
				y_intersect = shifted_y_2
			# Append intersections to the lists
			x_interior_per_vertex.append(x_intersect)
			y_interior_per_vertex.append(y_intersect)

		# Store the vertices and normal vector for each face of the polygon