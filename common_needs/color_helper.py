##########################################
### Import needed general dependencies ###
##########################################
# Internal modules
from type_helper import isListWithNumericEntries, isNumeric

# External modules
from math import sqrt
from plotly.express import colors
from PrivateAttributesDecorator import private_attributes_dec
from typing import Any, Tuple


#############################################
### Define an generalized RGB color class ###
#############################################
# Create the decorator needed for making the attributes private
rgb_decorator = private_attributes_dec("_red_value",					# class variables
									   "_green_value",
									   "_blue_value")
									   
# Define the class with private attributes
@rgb_decorator
class RGB:
	### Initialize the class ###
	def __init__(self, input_value:Any):
		# Attempt to extract the red, green and blue values from the input value
		if type(input_value) in [list, tuple]:
			# Verify the inputs
			assert len(input_value) == 3, "RGB::__init__: If provided value for 'input_value' is a list or tuple then it must be of length 3"
			assert type(input_value[0]) == type(input_value[1]), "RGB::__init__: If provided value for 'input_value' is a list or tuple then 1st and 2nd entries must be of the same type"
			assert type(input_value[0]) == type(input_value[2]), "RGB::__init__: If provided value for 'input_value' is a list or tuple then 1st and 3rd entries must be of the same type"
			assert isNumeric(input_value[0], include_numpy_flag = False) == True, "RGB::__init__: If provided value for 'input_value' is a list or tuple then 1st entry must be a float or int object"
			# Convert floats to integers (if needed)
			if type(input_value[0]) == float:
				input_value = (int(255.0 * input_value[0]), int(255.0 * input_value[1]), int(255.0 * input_value[2]))
			# Extract the red, green and blue values
			red_value = input_value[0]
			green_value = input_value[1]
			blue_value = input_value[2]
		elif type(input_value) == str:
			# Verify the inputs
			assert len(input_value) > 0, "RGB::__init__: If provided value for 'input_value' is a string then it must be non-empty"
			# Handle the various cases
			if input_value.startswith("rgb("):
				# Make sure it has the correct ending
				assert input_value.endswith(")"), "RGB::__init__: If provided value for 'input_value' is a string beginning with 'rgb(' then it must end with ')'"
				# Get the split of the middle segment w.r.t. commas and make sure it is length 3
				middle_split = input_value[4:-1].split(",")
				assert len(middle_split) == 3, "RGB::__init__: If provided value for 'input_value' is a string beginning with 'rgb(' then it must contain exactly two commas"
				# Attempt to get the 1st part of the middle split as an integer
				assert len(middle_split[0]) > 0, "RGB::__init__: If provided value for 'input_value' is a string beginning with 'rgb(' then it must have non-empty red section"
				for character in middle_split[0]:
					assert character in "0123456789", "RGB::__init__: If provided value for 'input_value' is a string beginning with 'rgb(' then it must have red section containing only the digits 0 through 9"
				red_value = int(middle_split[0])
				# Attempt to get the 2nd part of the middle split as an integer
				assert len(middle_split[1]) > 0, "RGB::__init__: If provided value for 'input_value' is a string beginning with 'rgb(' then it must have non-empty green section"
				for character in middle_split[1]:
					assert character in "0123456789", "RGB::__init__: If provided value for 'input_value' is a string beginning with 'rgb(' then it must have green section containing only the digits 0 through 9"
				green_value = int(middle_split[1])
				# Attempt to get the 3rd part of the middle split as an integer
				assert len(middle_split[2]) > 0, "RGB::__init__: If provided value for 'input_value' is a string beginning with 'rgb(' then it must have non-empty blue section"
				for character in middle_split[2]:
					assert character in "0123456789", "RGB::__init__: If provided value for 'input_value' is a string beginning with 'rgb(' then it must have blue section containing only the digits 0 through 9"
				blue_value = int(middle_split[2])
			elif input_value.startswith("#"):
				# Make sure the string is the correct length
				assert len(input_value) == 7, "RGB::__init__: If provided value for 'input_value' is a string beginning with '#' then it must be of length 7"
				# Make sure the individual characters are correct
				for character in input_value[1:]:
					assert character in "0123456789abcdef", "RGB::__init__: If provided value for 'input_value' is a string beginning with '#' then it must contain only the digits 0 through 9 and letters a through f"
				# Extract the red, green and blue values
				red_value = int(input_value[1:3], 16)
				green_value = int(input_value[3:5], 16)
				blue_value = int(input_value[5:7], 16)
			else:
				assert False, "RGB::__init__: If provided value for 'input_value' is a string then it must begin with 'rgb(' or '#'"
		else:
			# Raise error, incorrect type
			assert False, "RGB::__init__: Provided value for 'input_value' must be a list, str or tuple object"
		
		# Verify the extracted values
		assert 0 <= red_value and red_value <= 255, "RGB::__init__: Provided value for 'input_value' must red value representing a number >= 0 and <= 255"
		assert 0 <= green_value and green_value <= 255, "RGB::__init__: Provided value for 'input_value' must green value representing a number >= 0 and <= 255"
		assert 0 <= blue_value and blue_value <= 255, "RGB::__init__: Provided value for 'input_value' must blue value representing a number >= 0 and <= 255"
		
		# Store the provided values
		self._red_value = red_value
		self._green_value = green_value
		self._blue_value = blue_value
		
	### Define functions for giving the RGB values in various formats ###
	def asStringHex(self) -> str:
		# Return RGB values a string of hexidecimal digits
		# Get the red value as a string of length 2
		red_value_str = hex(self._red_value)[2:]
		if len(red_value_str) == 1:
			red_value_str = "0" + red_value_str
			
		# Get the green value as a string of length 2
		green_value_str = hex(self._green_value)[2:]
		if len(green_value_str) == 1:
			green_value_str = "0" + green_value_str
			
		# Get the blue value as a string of length 2
		blue_value_str = hex(self._blue_value)[2:]
		if len(blue_value_str) == 1:
			blue_value_str = "0" + blue_value_str
			
		# Return the result
		return "#" + red_value_str + green_value_str + blue_value_str
		
	def asStringTuple(self) -> str:
		# Return RGB values as a string of values between 0 and 255
		return "rgb(" + str(self._red_value) + "," + str(self._green_value) + "," + str(self._blue_value) + ")"
		
	def asTupleFloat(self) -> Tuple[float, float, float]:
		# Return RGB values as a tuple of values between 0 and 1
		return (self._red_value / 255.0, self._green_value / 255.0, self._blue_value / 255.0)
		
	def asTupleInt(self) -> Tuple[int, int, int]:
		# Return RGB values as a tuple of values between 0 and 255
		return (self._red_value, self._green_value, self._blue_value)
		
	### Define helpful magic methods ###	
	def __eq__(self, other:Any) -> bool:
		# Verify if this RGB object is equal to something else
		if type(other) == RGB:
			return str(self) == str(other)
		else:
			return False
			
	def __neq__(self, other:Any) -> bool:
		# Verify if this RGB object is equal to something else
		if type(other) == RGB:
			return str(self) != str(other)
		else:
			return True
			
	def __str__(self) -> str:
		# Return a string representation of this RGB object
		return self.asStringTuple()
		
#print(RGB("rgb(0,24,255)"))
#print(RGB("#3f5b12"))
#print(RGB((32, 56, 180)))
#print(RGB([0.1, 0.4, 0.6]))
		
		
##################################################
### Define a generalized RGB color scale class ###
##################################################
# Create the decorator needed for making the attributes private
rgb_scale_decorator = private_attributes_dec("_rgb_values",				# class variables
											 "_parameter_values")
									   
# Define the class with private attributes
@rgb_scale_decorator
class RGBScale:
	### Initialize the class ###
	def __init__(rgb_values:list, parameter_values:list = None):
		# Verify the inputs
		print(rgb_values)
		assert type(rgb_values) == list, "RGBScale::__init__: Provided value for 'rgb_values' must be a list object"
		assert len(rgb_values) > 0, "RGBScale::__init__: Provided value for 'rgb_values' must be a non-empty list"
		for rgb_value in rgb_values:
			assert type(rgb_value) == RGB, "RGBScale::__init__: Provided value for 'rgb_values' must be a list of RGB objects"
		if parameter_values is not None:
			assert type(parameter_values) == list, "RGBScale::__init__: If provided, value for 'parameter_values' must be a list object"
			assert len(parameter_values) == len(rgb_values), "RGBScale::__init__: If provided, value for 'parameter_values' must be of the same length as 'rgb_values'"
			assert isListWithNumericEntries(parameter_values, include_numpy_flag = False) == True, "RGBScale::__init__: If provided, value for 'parameter_values' must be a list of float and int objects"
			assert parameter_values == sorted(parameter_values, reverse = False), "RGBScale::__init__: If provided, value for 'parameter_values' must be an increasing list of numbers"
			
		# Store the provided values
		self._rgb_values = rgb_values
		self._parameter_values = parameter_values


##########################################################
### Create helpers related to plotly colors and scales ###
##########################################################
# Get all plotly scale root names as a list
ALL_PLOTLY_SCALE_ROOT_NAMES = colors.named_colorscales()

# Get all plotly scales as a dictionary of RGB scale ojects
ALL_PLOTLY_SCALES = {}
for scale_type in ["cyclical", "diverging", "qualitative", "sequential"]:
	ALL_PLOTLY_SCALES[scale_type] = {}
	for scale_name in dir(getattr(colors, scale_type)):
		if scale_name.lower() in ALL_PLOTLY_SCALE_ROOT_NAMES or scale_name.lower() + "_r" in ALL_PLOTLY_SCALE_ROOT_NAMES:
			print(type(getattr(getattr(colors, scale_type), scale_name)))
			ALL_PLOTLY_SCALES[scale_type][scale_name] = getattr(getattr(colors, scale_type), scale_name)
print(ALL_PLOTLY_SCALES)


#################################################################
### Define a function for generating RGB values on a spectrum ###
#################################################################
def getRGB(parameter:Any, red_values:list = [255, 0, 0, 255, 255, 255], green_values:list = [0, 0, 255, 255, 127, 0], blue_values:list = [255, 255, 0, 0, 0, 0]) -> Tuple[int, int, int]:
	# Compute an RGB tuple using the provided inputs (note: input values between 0 and 1, output values between 0 and 255)
	# Verify the inputs
	assert isNumeric(parameter, include_numpy_flag = False) == True, "getRGB: Provided value for 'parameter' must be a float or int object"
	assert 0 <= parameter and parameter <= 1, "getRGB: Provided value for 'parameter' must be >= 0 and <= 1"
	assert type(red_values) == list, "getRGB: Provided value for 'red_values' must be a list object"
	for red_value in red_values:
		assert isNumeric(red_value, include_numpy_flag = False) == True, "getRGB: Provided value for 'red_values' must be a list of float or int objects"
		assert 0 <= red_value and red_value <= 255, "getRGB: Provided value for 'red_values' must be a list of numbers >= 0 and <= 255"
	assert type(green_values) == list, "getRGB: Provided value for 'green_values' must be a list object"
	for green_value in green_values:
		assert isNumeric(green_value, include_numpy_flag = False) == True, "getRGB: Provided value for 'green_values' must be a list of float or int objects"
		assert 0 <= green_value and red_value <= 255, "getRGB: Provided value for 'green_values' must be a list of numbers >= 0 and <= 255"
	assert type(blue_values) == list, "getRGB: Provided value for 'blue_values' must be a list object"
	for blue_value in blue_values:
		assert isNumeric(blue_value, include_numpy_flag = False) == True, "getRGB: Provided value for 'blue_values' must be a list of float or int objects"
		assert 0 <= blue_value and blue_value <= 255, "getRGB: Provided value for 'blue_values' must be a list of numbers >= 0 and <= 255"
		
	# Compute the non-normalized red value to use
	if len(red_values) == 0:
		# No value provided, use middle amount
		non_normalized_red = 127
	elif len(red_values) == 1:
		# Single value provided, use that value
		non_normalized_red = red_values[0]
	else:
		# Multiple values provided, linearly interpolate
		denominator = len(red_values) - 1
		for index in range(1, denominator + 1):
			if parameter <= index / denominator:
				non_normalized_red = red_values[index - 1] * (index - denominator * parameter) + red_values[index] * (denominator * parameter - index + 1)
				break
				
	# Compute the non-normalized green value to use
	if len(green_values) == 0:
		# No value provided, use middle amount
		non_normalized_green = 127
	elif len(green_values) == 1:
		# Single value provided, use that value
		non_normalized_green = green_values[0]
	else:
		# Multiple values provided, linearly interpolate
		denominator = len(green_values) - 1
		for index in range(1, denominator + 1):
			if parameter <= index / denominator:
				non_normalized_green = green_values[index - 1] * (index - denominator * parameter) + green_values[index] * (denominator * parameter - index + 1)
				break
				
	# Compute the non-normalized blue value to use
	if len(blue_values) == 0:
		# No value provided, use middle amount
		non_normalized_blue = 127
	elif len(blue_values) == 1:
		# Single value provided, use that value
		non_normalized_blue = blue_values[0]
	else:
		# Multiple values provided, linearly interpolate
		denominator = len(blue_values) - 1
		for index in range(1, denominator + 1):
			if parameter <= index / denominator:
				non_normalized_blue = blue_values[index - 1] * (index - denominator * parameter) + blue_values[index] * (denominator * parameter - index + 1)
				break
				
	# Compute the normalizing value
	rgb_normalizer = sqrt(non_normalized_red**2 + non_normalized_green**2 + non_normalized_blue**2)
	
	# Compute the normalized RGB values to use
	normalized_red = int(255 * non_normalized_red / rgb_normalizer)
	normalized_green = int(255 * non_normalized_green / rgb_normalizer)
	normalized_blue = int(255 * non_normalized_blue / rgb_normalizer)
	
	# Return the results
	return (normalized_red, normalized_green, normalized_blue)
