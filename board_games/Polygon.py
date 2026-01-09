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
from type_helper import isNumeric

# External modules
from PrivateAttributesDecorator import private_attributes_dec

####################################################################
### Define the polygon class as a basis for commonly used shapes ###
####################################################################
# Create the decorator needed for making the attributes private
polygon_decorator = private_attributes_dec()

# Define the class with private attributes
@polygon_decorator
class Polygon:
    ### Initialize the class ###
    def __init__(self):
        pass