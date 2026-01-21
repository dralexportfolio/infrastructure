##########################################
### Import needed general dependencies ###
##########################################
# Add paths for internal modules
# Import dependencies
from copy import deepcopy
from pathlib import Path
from sys import path
# Get the shared infrastructure folder
infrastructure_folder = Path(__file__).parent.parent
# Add the needed paths
path.insert(0, str(infrastructure_folder.joinpath("common_needs")))

# Internal modules
from color_helper import RGB
from type_helper import isListWithNumericEntries, isNumeric, tolerantlyCompare

# External modules
from copy import deepcopy
from io import BytesIO
from math import acos, cos, pi, sin, sqrt, tan
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from numpy import array, matmul, zeros
from numpy.linalg import det, inv, norm
from PIL import Image
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
										   "_x_lower",
										   "_x_midpoint_per_edge",
										   ",x_upper",
										   "_x_value_per_vertex",
										   "_x_values_per_face",
										   "_y_lower",
										   "_y_midpoint_per_edge",
										   "_y_upper",
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
		for edge_index in range(self._n_vertices):
			# Compute the midpoint coordinates
			self._x_midpoint_per_edge.append((self._x_value_per_vertex[edge_index] + self._x_value_per_vertex[(edge_index + 1) % self._n_vertices]) / 2)
			self._y_midpoint_per_edge.append((self._y_value_per_vertex[edge_index] + self._y_value_per_vertex[(edge_index + 1) % self._n_vertices]) / 2)
			# Compute the externally facing unit normal vector
			x_component = self._y_value_per_vertex[(edge_index + 1) % self._n_vertices] - self._y_value_per_vertex[edge_index]
			y_component = self._x_value_per_vertex[edge_index] - self._x_value_per_vertex[(edge_index + 1) % self._n_vertices]
			normal_vector = array([[x_component], [y_component]], dtype = float)
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
		self._x_lower = None
		self._x_upper = None
		self._x_values_per_face = None
		self._y_lower = None
		self._y_upper = None
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
		for vertex_index_1 in range(n_vertices - 1):
			for vertex_index_2 in range(vertex_index_1 + 1, n_vertices):
				delta_x = x_value_per_vertex[vertex_index_2] - x_value_per_vertex[vertex_index_1]
				delta_y = y_value_per_vertex[vertex_index_2] - y_value_per_vertex[vertex_index_1]
				assert delta_x**2 + delta_y**2 > 0, "Polygon::_processInputs: Provided values for 'x_value_per_vertex' and 'y_value_per_vertex' must represent distinct points in the plane"

		# Make sure that none of the line segments defining the edges intersect each other
		for vertex_index_1 in range(n_vertices - 1):
			# Get the vector defining the 1st edge
			delta_x_1 = x_value_per_vertex[vertex_index_1 + 1] - x_value_per_vertex[vertex_index_1]
			delta_y_1 = y_value_per_vertex[vertex_index_1 + 1] - y_value_per_vertex[vertex_index_1]
			direction_1 = array([[delta_x_1], [delta_y_1]], dtype = float)

			# Loop over the other edges
			for vertex_index_2 in range(vertex_index_1 + 1, n_vertices):
				# Get the vector defining the 2nd edge
				delta_x_2 = x_value_per_vertex[(vertex_index_2 + 1) % n_vertices] - x_value_per_vertex[vertex_index_2]
				delta_y_2 = y_value_per_vertex[(vertex_index_2 + 1) % n_vertices] - y_value_per_vertex[vertex_index_2]
				direction_2 = array([[delta_x_2], [delta_y_2]], dtype = float)

				# Create the matrix and vector needed to solve the relevant system
				needed_matrix = array([[direction_1[0, 0], -direction_2[0, 0]], [direction_1[1, 0], -direction_2[1, 0]]], dtype = float)
				needed_vector = array([[x_value_per_vertex[vertex_index_2] - x_value_per_vertex[vertex_index_1]], [y_value_per_vertex[vertex_index_2] - y_value_per_vertex[vertex_index_1]]], dtype = float)

				# Handle the various cases
				if tolerantlyCompare(det(needed_matrix), "!=", 0, include_numpy_flag = True):
					# Solve the needed system and determine if error needs to be raised
					needed_solution = matmul(inv(needed_matrix), needed_vector)
					condition_1 = tolerantlyCompare(needed_solution[0, 0], "<=", 0, include_numpy_flag = True)
					condition_2 = tolerantlyCompare(1, "<=", needed_solution[0, 0], include_numpy_flag = True)
					condition_3 = tolerantlyCompare(needed_solution[1, 0], "<=", 0, include_numpy_flag = True)
					condition_4 = tolerantlyCompare(1, "<=", needed_solution[1, 0], include_numpy_flag = True)
					assert condition_1 or condition_2 or condition_3 or condition_4, "Polygon::_processInputs: Provided values for 'x_value_per_vertex' and 'y_value_per_vertex' must represent a set of line segments which do not intersect each other"
				else:
					# Either or 0 or infinitely many solutions because directions are parallel, make sure there are 0 solutions
					# Create the matrix used to determine if the two edges lie on the same line
					parallelogram_matrix = array([[direction_1[0, 0], needed_vector[0, 0]], [direction_1[1, 0], needed_vector[1, 0]]], dtype = float)
					# Lie on the same line if determinant is 0, if so make sure there is no overlap
					if tolerantlyCompare(det(parallelogram_matrix), "==", 0, include_numpy_flag = True):
						# Parametrize the line so that the 1st edge is from 0 to 1, then find the bounds for the 2nd edge
						if direction_1[0, 0] != 0:
							start_2 = (x_value_per_vertex[vertex_index_2] - x_value_per_vertex[vertex_index_1]) / direction_1[0, 0]
							end_2 = start_2 + direction_2[0, 0] / direction_1[0, 0]
						else:
							start_2 = (y_value_per_vertex[vertex_index_2] - y_value_per_vertex[vertex_index_1]) / direction_1[1, 0]
							end_2 = start_2 + direction_2[1, 0] / direction_1[1, 0]
						# Determine if the line segments overlap and raise error if so
						condition_1 = tolerantlyCompare(start_2, "<=", 0, include_numpy_flag = True)
						condition_2 = tolerantlyCompare(1, "<=", start_2, include_numpy_flag = True)
						condition_3 = tolerantlyCompare(end_2, "<=", 0, include_numpy_flag = True)
						condition_4 = tolerantlyCompare(1, "<=", end_2, include_numpy_flag = True)
						assert (condition_1 or condition_2) and (condition_3 or condition_4), "Polygon::_processInputs: Provided values for 'x_value_per_vertex' and 'y_value_per_vertex' must represent a set of line segments which do not overlap when they lie on the same line"

		# Cycle through the vertices on the boundary of the polygon and determine the total angle turned, keep order if CCW and reverse if CW
		# Initialize the degree counter
		degree_counter = 0
		# Loop over the vertices and compute the signed degrees turned
		for vertex_index in range(n_vertices):
			# Get the vector defining the 1st edge
			delta_x_1 = x_value_per_vertex[vertex_index] - x_value_per_vertex[vertex_index - 1]
			delta_y_1 = y_value_per_vertex[vertex_index] - y_value_per_vertex[vertex_index - 1]
			direction_1 = array([[delta_x_1], [delta_y_1]], dtype = float)
			# Get the vector defining the 2nd edge
			delta_x_2 = x_value_per_vertex[(vertex_index + 1) % n_vertices] - x_value_per_vertex[vertex_index]
			delta_y_2 = y_value_per_vertex[(vertex_index + 1) % n_vertices] - y_value_per_vertex[vertex_index]
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

	### Define external functions for transforming the x-values and y-values of the polygon ###
	def rotate(self, angle:Any, x_center:Any = 0, y_center:Any = 0):
		# Rotate the vertices the given number of degrees about the given center point
		# Verify the inputs
		assert isNumeric(angle, include_numpy_flag = True) == True, "Polygon::rotate: Provided value for 'angle' must be numeric"
		assert -360 < angle and angle < 360, "Polygon::rotate: Provided value for 'angle' must be > -360 and < 360"
		assert isNumeric(x_center, include_numpy_flag = True) == True, "Polygon::rotate: Provided value for 'x_center' must be numeric"
		assert -float("inf") < x_center and x_center < float("inf"), "Polygon::rotate: Provided value for 'x_center' must be finite"
		assert isNumeric(y_center, include_numpy_flag = True) == True, "Polygon::rotate: Provided value for 'y_center' must be numeric"
		assert -float("inf") < y_center and y_center < float("inf"), "Polygon::rotate: Provided value for 'y_center' must be finite"

		# Set the preprocess flags to False and forget the previous bevel and sun information
		# Bevel info (in order of appearance from preprocessBevelInfo)
		self._preprocess_bevel_flag = False
		self._bevel_attitude = None
		self._bevel_size = None
		self._n_edges_per_face = None
		self._x_values_per_face = None
		self._y_values_per_face = None
		self._normal_vector_per_face = None
		self._x_lower = None
		self._x_upper = None
		self._y_lower = None
		self._y_upper = None
		self._render_figure = None
		self._render_axis = None
		self._patch_per_face = None
		# Sun info (in order of appearance from preprocessSunInfo)
		self._preprocess_sun_flag = False
		self._sun_angle = None
		self._sun_attitude = None
		self._raw_brightness_per_face = None

		# Get a shifted version of the stored x-values and y-values
		shifted_x_value_per_vertex = [x_value - x_center for x_value in self._x_value_per_vertex]
		shifted_y_value_per_vertex = [y_value - y_center for y_value in self._y_value_per_vertex]

		# Compute the needed rotation transformation
		rotated_x_value_per_vertex = []
		rotated_y_value_per_vertex = []
		for vertex_index in range(self._n_vertices):
			# Get the needed x-value and y-value
			x_value = shifted_x_value_per_vertex[vertex_index]
			y_value = shifted_y_value_per_vertex[vertex_index]

			# Apply the needed rotation and store
			rotated_x_value_per_vertex.append(cos(angle * pi / 180) * x_value - sin(angle * pi / 180) * y_value)
			rotated_y_value_per_vertex.append(sin(angle * pi / 180) * x_value + cos(angle * pi / 180) * y_value)

		# Update the internally stored values by shifting back
		self._x_value_per_vertex = [x_value + x_center for x_value in rotated_x_value_per_vertex]
		self._y_value_per_vertex = [y_value + y_center for y_value in rotated_y_value_per_vertex]

	def scale(self, factor:Any, x_center:Any = 0, y_center:Any = 0):
		# Scale the vertices by the given factor about the given center point
		# Verify the inputs
		assert isNumeric(factor, include_numpy_flag = True) == True, "Polygon::scale: Provided value for 'factor' must be numeric"
		assert 0 < factor and factor < float("inf"), "Polygon::scale: Provided value for 'factor' must be positive and finite"
		assert isNumeric(x_center, include_numpy_flag = True) == True, "Polygon::scale: Provided value for 'x_center' must be numeric"
		assert -float("inf") < x_center and x_center < float("inf"), "Polygon::scale: Provided value for 'x_center' must be finite"
		assert isNumeric(y_center, include_numpy_flag = True) == True, "Polygon::scale: Provided value for 'y_center' must be numeric"
		assert -float("inf") < y_center and y_center < float("inf"), "Polygon::scale: Provided value for 'y_center' must be finite"

		# Set the preprocess flags to False and forget the previous bevel and sun information
		# Bevel info (in order of appearance from preprocessBevelInfo)
		self._preprocess_bevel_flag = False
		self._bevel_attitude = None
		self._bevel_size = None
		self._n_edges_per_face = None
		self._x_values_per_face = None
		self._y_values_per_face = None
		self._normal_vector_per_face = None
		self._x_lower = None
		self._x_upper = None
		self._y_lower = None
		self._y_upper = None
		self._render_figure = None
		self._render_axis = None
		self._patch_per_face = None
		# Sun info (in order of appearance from preprocessSunInfo)
		self._preprocess_sun_flag = False
		self._sun_angle = None
		self._sun_attitude = None
		self._raw_brightness_per_face = None

		# Get a shifted version of the stored x-values and y-values
		shifted_x_value_per_vertex = [x_value - x_center for x_value in self._x_value_per_vertex]
		shifted_y_value_per_vertex = [y_value - y_center for y_value in self._y_value_per_vertex]

		# Compute the needed scaling transformation
		scaled_x_value_per_vertex = [factor * x_value for x_value in shifted_x_value_per_vertex]
		scaled_y_value_per_vertex = [factor * y_value for y_value in shifted_y_value_per_vertex]

		# Update the internally stored values by shifting back
		self._x_value_per_vertex = [x_value + x_center for x_value in rotated_x_value_per_vertex]
		self._y_value_per_vertex = [y_value + y_center for y_value in rotated_y_value_per_vertex]

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
		self._raw_brightness_per_face = None

		# Compute the vertex locations for the interior shape created by the bevel
		# Initialize the needed lists
		x_interior_per_vertex = []
		y_interior_per_vertex = []
		# Compute the needed values
		for vertex_index in range(self._n_vertices):
			# Compute the shifted base points for the needed lines
			shifted_x_1 = self._x_value_per_vertex[vertex_index - 1] - self._bevel_size * self._normal_vector_per_edge[vertex_index - 1][0, 0]
			shifted_y_1 = self._y_value_per_vertex[vertex_index - 1] - self._bevel_size * self._normal_vector_per_edge[vertex_index - 1][1, 0]
			shifted_x_2 = self._x_value_per_vertex[vertex_index] - self._bevel_size * self._normal_vector_per_edge[vertex_index][0, 0]
			shifted_y_2 = self._y_value_per_vertex[vertex_index] - self._bevel_size * self._normal_vector_per_edge[vertex_index][1, 0]
			# Get the vector defining the 1st edge
			delta_x_1 = self._x_value_per_vertex[vertex_index] - self._x_value_per_vertex[vertex_index - 1]
			delta_y_1 = self._y_value_per_vertex[vertex_index] - self._y_value_per_vertex[vertex_index - 1]
			direction_1 = array([[delta_x_1], [delta_y_1]], dtype = float)
			# Get the vector defining the 2nd edge
			delta_x_2 = self._x_value_per_vertex[(vertex_index + 1) % self._n_vertices] - self._x_value_per_vertex[vertex_index]
			delta_y_2 = self._y_value_per_vertex[(vertex_index + 1) % self._n_vertices] - self._y_value_per_vertex[vertex_index]
			direction_2 = array([[delta_x_2], [delta_y_2]], dtype = float)
			# Create the matrix and vector needed to solve the system
			needed_matrix = array([[direction_1[0, 0], -direction_2[0, 0]], [direction_1[1, 0], -direction_2[1, 0]]], dtype = float)
			needed_vector = array([[shifted_x_2 - shifted_x_1], [shifted_y_2 - shifted_y_1]], dtype = float)
			# Handle the various cases
			if tolerantlyCompare(det(needed_matrix), "!=", 0, include_numpy_flag = True):
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
		# Initialize the needed lists
		self._n_edges_per_face = []
		self._x_values_per_face = []
		self._y_values_per_face = []
		self._normal_vector_per_face = []
		# Compute the z-component for normal vectors associated with edges
		z_component = tan((90 - self._bevel_attitude) * pi / 180)
		# Add the information for each face
		for face_index in range(self._n_vertices + 1):
			# Get the needed value for this face
			if face_index < self._n_vertices:
				# Use the information of this edge for the vertices
				current_n_vertices = 4
				current_x_values = [self._x_value_per_vertex[face_index],
									self._x_value_per_vertex[(face_index + 1) % self._n_vertices],
									x_interior_per_vertex[(face_index + 1) % self._n_vertices],
									x_interior_per_vertex[face_index]]
				current_y_values = [self._y_value_per_vertex[face_index],
									self._y_value_per_vertex[(face_index + 1) % self._n_vertices],
									y_interior_per_vertex[(face_index + 1) % self._n_vertices],
									y_interior_per_vertex[face_index]]
				# Use the edge's normal vector and the bevel attitude to get the face's normal vector
				x_component = self._normal_vector_per_edge[face_index][0, 0]
				y_component = self._normal_vector_per_edge[face_index][1, 0]
				current_normal_vector = array([[x_component], [y_component], [z_component]], dtype = float)
			else:
				# Use the interior vertices of this face
				current_n_vertices = self._n_vertices
				current_x_values = x_interior_per_vertex
				current_y_values = y_interior_per_vertex
				# Use the z-axis for this face's normal vector
				current_normal_vector = array([[0], [0], [1]], dtype = float)
			# Make sure this face would end up being a valid Polygon object, should only fail if bevel size is too large
			try:
				Polygon(n_vertices = current_n_vertices, x_value_per_vertex = current_x_values, y_value_per_vertex = current_y_values)
			except:
				assert False, "Polygon::preprocessBevelInfo: Provided value for 'bevel_size' is too large, ended creating invalid face shapes"
			# Store the information
			self._n_edges_per_face.append(current_n_vertices)
			self._x_values_per_face.append(current_x_values)
			self._y_values_per_face.append(current_y_values)
			self._normal_vector_per_face.append(current_normal_vector / norm(current_normal_vector))

		# Compute and store the bounds of the render image
		self._x_lower = min(self._x_value_per_vertex)
		self._x_upper = max(self._x_value_per_vertex)
		self._y_lower = min(self._y_value_per_vertex)
		self._y_upper = max(self._y_value_per_vertex)

		# Create the figure and axis to which to render, crop it to the needed size (in normalized figure coordinates), and adjust the axis as needed
		self._render_figure, self._render_axis = plt.subplots(figsize = (self._x_upper - self._x_lower, self._y_upper - self._y_lower))
		self._render_figure.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, wspace = 0, hspace = 0)
		self._render_axis.set_xlim(left = self._x_lower, right = self._x_upper)
		self._render_axis.set_ylim(bottom = self._y_lower, top = self._y_upper)
		self._render_axis.axis("off")

		# Create the patch objects without any color and save them for future use
		self._patch_per_face = []
		for face_index in range(self._n_vertices + 1):
			# Create an array containing the vertices of this face
			vertex_array = zeros((self._n_edges_per_face[face_index], 2), dtype = float)
			for row_index in range(self._n_edges_per_face[face_index]):
				vertex_array[row_index, 0] = self._x_values_per_face[face_index][row_index]
				vertex_array[row_index, 1] = self._y_values_per_face[face_index][row_index]

			# Create the needed patch and add it to the plot
			self._patch_per_face.append(self._render_axis.add_patch(patches.Polygon(vertex_array, closed = True, edgecolor = None, linewidth = 0)))

		# Set the preprocess bevel flag to True
		self._preprocess_bevel_flag = True

	def preprocessSunInfo(self, sun_angle:Any, sun_attitude:Any):
		# Preprocess all information related to the sun
		# Only proceed if the bevel information has been precomputed
		assert self._preprocess_bevel_flag == True, "Polygon::preprocessSunInfo: Only able to preprocess sun information once bevel information has been preprocessed"

		# Verify the inputs
		assert isNumeric(sun_angle, include_numpy_flag = True) == True, "Polygon::preprocessSunInfo: Provided value for 'sun_angle' must be numeric"
		assert 0 <= sun_angle and sun_angle < 360, "Polygon::preprocessSunInfo: Provided value for 'sun_angle' must be >= 0 and < 360"
		assert isNumeric(sun_attitude, include_numpy_flag = True) == True, "Polygon::preprocessSunInfo: Provided value for 'sun_attitude' must be numeric"
		assert 0 <= sun_attitude and sun_attitude <= 90, "Polygon::preprocessSunInfo: Provided value for 'sun_attitude' must be >= 0 and <= 90"
		assert sun_attitude > self._bevel_attitude, "Polygon::preprocessSunInfo: Provided value for 'sun_attitude' must be greater than stored bevel attitude (i.e. " + str(self._bevel_attitude) + ")"

		# Store the provided values
		self._sun_angle = sun_angle
		self._sun_attitude = sun_attitude

		# Set the preprocess sun flag to False
		self._preprocess_sun_flag = False

		# Use the above information to compute the unit vector defining the direction of the sun
		# Compute the components
		x_component = cos(self._sun_angle * pi / 180)
		y_component = sin(self._sun_angle * pi / 180)
		z_component = tan(self._sun_attitude * pi / 180)
		# Define the unit vector
		sun_vector = array([[x_component], [y_component], [z_component]], dtype = float)
		sun_vector /= norm(sun_vector)

		# Compute the raw brightness associated with each face
		# Initialize the needed list
		self._raw_brightness_per_face = []
		# Compute the needed values
		for face_index in range(self._n_vertices + 1):
			# Compute the cosine of the angle between the sun direction and the normal vector of the face, store this as the brightness
			normal_vector = self._normal_vector_per_face[face_index]
			cos_angle = sun_vector[0, 0] * normal_vector[0, 0] + sun_vector[1, 0] * normal_vector[1, 0] + sun_vector[2, 0] * normal_vector[2, 0]
			self._raw_brightness_per_face.append(cos_angle)

		# Set the preprocess sun flag to True
		self._preprocess_sun_flag = True

	### Define external functions needed for rendering the polygon ###
	def computeRenderInfo(self, min_brightness:Any = 0, max_brightness:Any = 1, tint_shade:RGB = RGB((255, 255, 255)), x_shift:Any = 0, y_shift:Any = 0) -> dict:
		# Return a dictionary of values relevant to the rendering of the polygon (accounting for an optional shift)
		# Only proceed if the bevel and sun information have been preprocessed
		assert self._preprocess_bevel_flag == True, "Polygon::computeRenderInfo: Only able to return render information once bevel information has been preprocessed"

		# Verify the inputs
		assert isNumeric(min_brightness, include_numpy_flag = True) == True, "Polygon::computeRenderInfo: Provided value for 'min_brightness' must be numeric"
		assert 0 <= min_brightness and min_brightness < 1, "Polygon::computeRenderInfo: Provided value for 'min_brightness' must be >= 0 and < 1"
		assert isNumeric(max_brightness, include_numpy_flag = True) == True, "Polygon::computeRenderInfo: Provided value for 'max_brightness' must be numeric"
		assert 0 < max_brightness and max_brightness <= 1, "Polygon::computeRenderInfo: Provided value for 'max_brightness' must be > 0 and <= 1"
		assert max_brightness >= min_brightness, "Polygon::computeRenderInfo: Provided value for 'max_brightness' must be greater than or equal to provided value for 'min_brightness'"
		assert type(tint_shade) == RGB, "Polygon::computeRenderInfo: Provided value for 'tint_shade' must be a RGB object"
		assert isNumeric(x_shift, include_numpy_flag = True) == True, "Polygon::computeRenderInfo: Provided value for 'x_shift' must be numeric"
		assert -float("inf") < x_shift and x_shift < float("inf"), "Polygon::computeRenderInfo: Provided value for 'x_shift' must be finite"
		assert isNumeric(y_shift, include_numpy_flag = True) == True, "Polygon::computeRenderInfo: Provided value for 'y_shift' must be numeric"
		assert -float("inf") < y_shift and y_shift < float("inf"), "Polygon::computeRenderInfo: Provided value for 'y_shift' must be finite"

		# Initialize the dictionary of results with relevant face information
		render_info = {
			"n_faces": self._n_vertices + 1,
			"n_edges_per_face": self._n_edges_per_face,
			"x_values_per_face": [[x_value + x_shift for x_value in x_values] for x_values in self._x_values_per_face],
			"y_values_per_face": [[y_value + y_shift for y_value in y_values] for y_values in self._y_values_per_face],
			"rgb_values_per_face": []
		}

		# Add the face color information if the sun info has been preprocessed
		# Get the tint shade values as a numpy array
		tint_shade_array = array(tint_shade.asTupleFloat(), dtype = float)
		# Compute the RGB values as a numpy array for each face
		for face_index in range(self._n_vertices + 1):
			# Compute the adjusted brightness for this face
			# Compute the color needed for this face
			if self._preprocess_sun_flag == True:
				# Linearly interpolate between the minimum and maximum brightness values according to the raw brightness
				adjusted_brightness = min_brightness + (max_brightness - min_brightness) * self._raw_brightness_per_face[face_index]
			else:
				# Raw brightness not computed because sun information not preprocessed, so just use the maximum brightness
				adjusted_brightness = max_brightness
			# Compute the RGB values and add to the needed list
			render_info["rgb_values_per_face"].append(adjusted_brightness * tint_shade_array)

		# Return the results
		return render_info

	def render(self, dpi:int, min_brightness:Any = 0, max_brightness:Any = 1, tint_shade:RGB = RGB((255, 255, 255))) -> Image.Image:
		# Return a PIL image render of the polygon with the preprocessed settings
		# Only proceed if the bevel and sun information have been preprocessed
		assert self._preprocess_bevel_flag == True, "Polygon::render: Only able to render polygon image once bevel information has been preprocessed"
		assert self._preprocess_sun_flag == True, "Polygon::render: Only able to render polygon image once sun information has been preprocessed"

		# Verify the inputs which won't be handled by computeRenderInfo
		assert type(dpi) == int, "Polygon::render: Provided value for 'dpi' must be an int object"
		assert 72 <= dpi and dpi <= 900, "Polygon::render: Provided value for 'dpi' must be >= 72 and <= 900"

		# Perform additional input verification and compute the RGB colors using computeRenderInfo without any shift
		rgb_values_per_face = self.computeRenderInfo(min_brightness = min_brightness, max_brightness = max_brightness, tint_shade = tint_shade, x_shift = 0, y_shift = 0)["rgb_values_per_face"]

		# Draw each of the patches in the needed lighting
		for face_index in range(self._n_vertices + 1):
			self._patch_per_face[face_index].set_facecolor(rgb_values_per_face[face_index])

		# Redraw the canvas now that all faces have been updated
		self._render_figure.canvas.draw_idle()

		# Create a buffer to which the image will be saved
		image_buffer = BytesIO()

		# Save the figure to the buffer
		self._render_figure.savefig(image_buffer, dpi = dpi, format = "png", transparent = True)

		# Rewind to the beginning of the buffer and load the render as a PIL image
		image_buffer.seek(0)
		polygon_render = Image.open(image_buffer)

		# Return the result
		return polygon_render

	### Define an external function for accessing internal information ###
	def getInfo(self) -> dict:
		# Return a dictionary of information relevant to this polygon
		# Initialize the dictionary to return
		polygon_info = {}

		# Add in the number of vertices (and edges)
		polygon_info["n_vertices"] = self._n_vertices

		# Add in x-values and y-values of the vertices
		polygon_info["x_value_per_vertex"] = self._x_value_per_vertex
		polygon_info["y_value_per_vertex"] = self._y_value_per_vertex

		# Add in the midpoint information for each edge
		polygon_info["x_midpoint_per_edge"] = self._x_midpoint_per_edge
		polygon_info["y_midpoint_per_edge"] = self._y_midpoint_per_edge

		# Add in the externally facing unit normal for each edge
		polygon_info["normal_vector_per_edge"] = self._normal_vector_per_edge

		# Add in the vertex information related to faces
		polygon_info["n_edges_per_face"] = self._n_edges_per_face
		polygon_info["x_values_per_face"] = self._x_values_per_face
		polygon_info["y_values_per_face"] = self._y_values_per_face

		# Add in the needed bevel information
		polygon_info["preprocess_bevel_flag"] = self._preprocess_bevel_flag
		polygon_info["bevel_attitude"] = self._bevel_attitude
		polygon_info["bevel_size"] = self._bevel_size

		# Add in the needed sun information
		polygon_info["preprocess_sun_flag"] = self._preprocess_sun_flag
		polygon_info["sun_angle"] = self._sun_angle
		polygon_info["sun_attitude"] = self._sun_attitude

		# Add in the bounds of the render image
		polygon_info["x_lower"] = self._x_lower
		polygon_info["x_upper"] = self._x_upper
		polygon_info["y_lower"] = self._y_lower
		polygon_info["y_upper"] = self._y_upper

		# Return the results
		return polygon_info

	### Define a deepcopy function compatible with PrivateAttributesDecorator ###
	def deepcopy(self):
		# Create a copy of this object and return it
		# Initialize a Polygon object with the same inputs
		copy_of_self = type(self)(n_vertices = deepcopy(self._n_vertices),
								  x_value_per_vertex = deepcopy(self._x_value_per_vertex),
								  y_value_per_vertex = deepcopy(self._y_value_per_vertex))

		# Preprocess the same bevel information (if needed)
		if self._preprocess_bevel_flag == True:
			copy_of_self.preprocessBevelInfo(bevel_attitude = deepcopy(self._bevel_attitude),
											 bevel_size = deepcopy(self._bevel_size))

		# Preprocess the same sun information (if needed)
		if self._preprocess_sun_flag == True:
			copy_of_self.preprocessSunInfo(sun_angle = deepcopy(self._sun_angle),
										   sun_attitude = deepcopy(self._sun_attitude))

		# Return the copied object
		return copy_of_self


###########################################################
### Define commonly used Polygon objects for future use ###
###########################################################
# Define a simple 1x1 square
SQUARE_1x1 = Polygon(n_vertices = 4, x_value_per_vertex = [0, 1, 1, 0], y_value_per_vertex = [0, 0, 1, 1])

# Define the four possible 1x1 triangles
TRIANGLE_1x1_NE = Polygon(n_vertices = 3, x_value_per_vertex = [1, 0, 1], y_value_per_vertex = [1, 1, 0])
TRIANGLE_1x1_NW = Polygon(n_vertices = 3, x_value_per_vertex = [0, 0, 1], y_value_per_vertex = [1, 0, 1])
TRIANGLE_1x1_SE = Polygon(n_vertices = 3, x_value_per_vertex = [1, 1, 0], y_value_per_vertex = [0, 1, 0])
TRIANGLE_1x1_SW = Polygon(n_vertices = 3, x_value_per_vertex = [0, 1, 0], y_value_per_vertex = [0, 0, 1])

# Define the 2x3 and 3x2 hexagons
HEXAGON_2x3 = Polygon(n_vertices = 6, x_value_per_vertex = [1, 2, 2, 1, 0, 0], y_value_per_vertex = [0, 1, 2, 3, 2, 1])
HEXAGON_3x2 = Polygon(n_vertices = 6, x_value_per_vertex = [1, 2, 3, 2, 1, 0], y_value_per_vertex = [0, 0, 1, 2, 2, 1])

# Define the regular hexagons
HEXAGON_REGULAR_TALL = Polygon(n_vertices = 6, x_value_per_vertex = [sqrt(3) / 2, sqrt(3), sqrt(3), sqrt(3) / 2, 0, 0], y_value_per_vertex = [0, 1 / 2, 3 / 2, 2, 3 / 2, 1 / 2])
HEXAGON_REGULAR_WIDE = Polygon(n_vertices = 6, x_value_per_vertex = [1 / 2, 3 / 2, 2, 3 / 2, 1 / 2, 0], y_value_per_vertex = [0, 0, sqrt(3) / 2, sqrt(3), sqrt(3), sqrt(3) / 2])