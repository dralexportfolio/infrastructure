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
from type_helper import isListWithNumericEntries

# External modules
from numpy import array, float64
from numpy.linalg import norm
from PrivateAttributesDecorator import private_attributes_dec

####################################################################
### Define the polygon class as a basis for commonly used shapes ###
####################################################################
# Create the decorator needed for making the attributes private
polygon_decorator = private_attributes_dec("_bevel_attitude"            # class variables
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
                                           "_verifyInputs")             # internal functions

# Define the class with private attributes
@polygon_decorator
class Polygon:
    ### Initialize the class ###
    def __init__(self, n_vertices:int, x_value_per_vertex:list, y_value_per_vertex:list):
        # Verify the inputs
        self._verifyInputs(n_vertices = n_vertices, x_value_per_vertex = x_value_per_vertex, y_value_per_vertex = y_value_per_vertex)

        # Store the provided values
        self._n_vertices = n_vertices
        self._x_value_per_vertex = x_value_per_vertex
        self._y_value_per_vertex = y_value_per_vertex

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

    ### Define an internal function to verify that the provided inputs values are valid ###
    def _verifyInputs(self, n_vertices:int, x_value_per_vertex:list, y_value_per_vertex:list):
        # Verify the inputs provided upon class initialization
        # Check the types and list lengths
        assert type(n_vertices) == int, "Polygon::_verifyInputs: Provided value for 'n_vertices' must be an int object"
        assert n_vertices >= 3, "Polygon::_verifyInputs: Provided value for 'n_vertices' must be >= 3"
        assert isListWithNumericEntries(x_value_per_vertex, include_numpy_flag = True) == True, "Polygon::_verifyInputs: Provided value for 'x_value_per_vertex' must be a list object containing numeric entries"
        assert len(x_value_per_vertex) == n_vertices, "Polygon::_verifyInputs: Provided value for 'x_value_per_vertex' must be a list of length equal to 'n_vertices'"
        assert -float("inf") < min(x_value_per_vertex) and max(x_value_per_vertex) < float("inf"), "Polygon::_verifyInputs: Provided value for 'x_value_per_vertex' must be a list of all finite entries"
        assert isListWithNumericEntries(y_value_per_vertex, include_numpy_flag = True) == True, "Polygon::_verifyInputs: Provided value for 'y_value_per_vertex' must be a list object containing numeric entries"
        assert len(y_value_per_vertex) == n_vertices, "Polygon::_verifyInputs: Provided value for 'y_value_per_vertex' must be a list of length equal to 'n_vertices'"
        assert -float("inf") < min(y_value_per_vertex) and max(y_value_per_vertex) < float("inf"), "Polygon::_verifyInputs: Provided value for 'y_value_per_vertex' must be a list of all finite entries"

        # Make sure that all vertex value pairs are distinct
        for index_1 in range(n_vertices - 1):
            for index_2 in range(index_1 + 1, n_vertices):
                delta_x = x_value_per_vertex[index_2] - x_value_per_vertex[index_1]
                delta_y = y_value_per_vertex[index_2] - y_value_per_vertex[index_1]
                assert delta_x**2 + delta_y**2 > 0, "Polygon::_verifyInputs: Provided values for 'x_value_per_vertex' and 'y_value_per_vertex' must represent distinct points in the plane"