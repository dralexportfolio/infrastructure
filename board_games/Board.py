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
board_decorator = private_attributes_dec("_bevel_attitude"				# class variables
										 "_bevel_size",
										 "_sun_angle",
										 "_sun_attitude",
										 "_processInputs")				# internal functions

# Define the class with private attributes
@board_decorator
class Board:
	### Initialize the class ###
	def __init__(self):
		pass