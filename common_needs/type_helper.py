##########################################
### Import needed general dependencies ###
##########################################
# External modules
from numpy import float16, float32, float64, int8, int16, int32, int64
from typing import Any


############################################################################
### Define functions needed for handling various types of numeric checks ###
############################################################################
# Define a function for verifying if a value is numeric
def isNumeric(value_to_check:Any, include_numpy_flag:bool = False) -> bool:
	# Determine if the provided value is of the allowed numeric type
	# Verify the inputs
	assert type(include_numpy_flag) == bool, "isNumeric: Provided value for 'include_numpy_flag' must be a bool object"

	# Construct the list of allowed types
	allowed_types = [float, int]
	if include_numpy_flag == True:
		allowed_types += [float16, float32, float64, int8, int16, int32, int64]
		
	# Test if the provided value is of an allowed type
	allowed_type_flag = type(value_to_check) in allowed_types
		
	# Return the results
	return allowed_type_flag


#################################################################################
### Define a helper function for comparing values within a rounding threshold ###
#################################################################################
def tolerantlyCompare(value_1:Any, operator:str, value_2:Any, threshold:Any = 10**-6, include_numpy_flag:bool = False) -> bool:
	# Perform the needed comparison of the values to be within the given threshold
	# Verify that relevant inputs are numeric
	message_suffix = "a float or int object" if include_numpy_flag == False else "numeric"
	assert isNumeric(value_1, include_numpy_flag = include_numpy_flag) == True, "tolerantlyCompare: Provided value for 'value_1' must be " + message_suffix
	assert isNumeric(value_2, include_numpy_flag = include_numpy_flag) == True, "tolerantlyCompare: Provided value for 'value_2' must be " + message_suffix
	assert isNumeric(threshold, include_numpy_flag = include_numpy_flag) == True, "tolerantlyCompare: Provided value for 'threshold' must be " + message_suffix

	# Verify any additional conditions
	assert operator in ["==", "!=", ">=", ">", "<=", "<"], "tolerantlyCompare: Provided value for 'operator' must be '==', '!=', '>=', '>', '<=' or '<'"
	assert 0 <= threshold and threshold < float("inf"), "tolerantlyCompare: Provided value for 'threshold' must be non-negative and finite"

	# Handle the various cases
	if operator == "==":
		return abs(value_1 - value_2) <= threshold
	elif operator == "!=":
		return abs(value_1 - value_2) > threshold
	elif operator == ">=":
		return value_1 >= value_2 - threshold
	elif operator == ">":
		return value_1 > value_2 - threshold
	elif operator == "<=":
		return value_1 <= value_2 + threshold
	else:
		return value_1 < value_2 + threshold


#########################################################################################
### Define functions for verifying that dictionaries have entries of the needed types ###
#########################################################################################
# Numeric types
def isDictionaryWithNumericKeys(dict_to_check:dict, include_numpy_flag:bool = False) -> bool:
	# Determine if the keys of the provided dictionary are all numeric
	# Verify the inputs
	assert type(dict_to_check) == dict, "isDictionaryWithNumericKeys: Provided value for 'dict_to_check' must be a dict object"
	
	# Check the needed type(s), break if not
	for key in dict_to_check:
		if isNumeric(key, include_numpy_flag = include_numpy_flag) == False:
			return False
	
	# Return True because break didn't occur
	return True
	
def isDictionaryWithNumericValues(dict_to_check:dict, include_numpy_flag:bool = False) -> bool:
	# Determine if the values of the provided dictionary are all numeric
	# Verify the inputs
	assert type(dict_to_check) == dict, "isDictionaryWithNumericValues: Provided value for 'dict_to_check' must be a dict object"
	
	# Check the needed type(s), break if not
	for key in dict_to_check:
		if isNumeric(dict_to_check[key], include_numpy_flag = include_numpy_flag) == False:
			return False
	
	# Return True because break didn't occur
	return True

# String type
def isDictionaryWithStringKeys(dict_to_check:dict, allow_empty_flag:bool = True) -> bool:
	# Determine if the keys of the provided dictionary are all strings
	# Verify the inputs
	assert type(dict_to_check) == dict, "isDictionaryWithStringKeys: Provided value for 'dict_to_check' must be a dict object"
	
	# Check the needed type(s), break if not
	for key in dict_to_check:
		if type(key) != str:
			return False
		elif allow_empty_flag == False and len(key) == 0:
			return False
	
	# Return True because break didn't occur
	return True
	
def isDictionaryWithStringValues(dict_to_check:dict, allow_empty_flag:bool = True) -> bool:
	# Determine if the values of the provided dictionary are all strings
	# Verify the inputs
	assert type(dict_to_check) == dict, "isDictionaryWithStringValues: Provided value for 'dict_to_check' must be a dict object"
	
	# Check the needed type(s), break if not
	for key in dict_to_check:
		if type(dict_to_check[key]) != str:
			return False
		elif allow_empty_flag == False and len(dict_to_check[key]) == 0:
			return False
	
	# Return True because break didn't occur
	return True

# Tuple type
def isDictionaryWithTupleKeys(dict_to_check:dict, allow_empty_flag:bool = True) -> bool:
	# Determine if the keys of the provided dictionary are all tuples
	# Verify the inputs
	assert type(dict_to_check) == dict, "isDictionaryWithTupleKeys: Provided value for 'dict_to_check' must be a dict object"
	
	# Check the needed type(s), break if not
	for key in dict_to_check:
		if type(key) != tuple:
			return False
		elif allow_empty_flag == False and len(key) == 0:
			return False
	
	# Return True because break didn't occur
	return True

def isDictionaryWithTupleValues(dict_to_check:dict, allow_empty_flag:bool = True) -> bool:
	# Determine if the values of the provided dictionary are all tuples
	# Verify the inputs
	assert type(dict_to_check) == dict, "isDictionaryWithTupleValues: Provided value for 'dict_to_check' must be a dict object"
	
	# Check the needed type(s), break if not
	for key in dict_to_check:
		if type(dict_to_check[key]) != tuple:
			return False
		elif allow_empty_flag == False and len(dict_to_check[key]) == 0:
			return False
	
	# Return True because break didn't occur
	return True

# List type
def isDictionaryWithListValues(dict_to_check:dict, allow_empty_flag:bool = True) -> bool:
	# Determine if the values of the provided dictionary are all lists
	# Verify the inputs
	assert type(dict_to_check) == dict, "isDictionaryWithListValues: Provided value for 'dict_to_check' must be a dict object"
	
	# Check the needed type(s), break if not
	for key in dict_to_check:
		if type(dict_to_check[key]) != list:
			return False
		elif allow_empty_flag == False and len(dict_to_check[key]) == 0:
			return False
	
	# Return True because break didn't occur
	return True

# Dictionary type
def isDictionaryWithDictionaryValues(dict_to_check:dict, allow_empty_flag:bool = True) -> bool:
	# Determine if the values of the provided dictionary are all dictionaries
	# Verify the inputs
	assert type(dict_to_check) == dict, "isDictionaryWithDictionary: Provided value for 'dict_to_check' must be a dict object"
	
	# Check the needed type(s), break if not
	for key in dict_to_check:
		if type(dict_to_check[key]) != dict:
			return False
		elif allow_empty_flag == False and len(dict_to_check[key]) == 0:
			return False
	
	# Return True because break didn't occur
	return True

##################################################################################
### Define functions for verifying that lists have entries of the needed types ###
##################################################################################
# Numeric types
def isListWithNumericEntries(list_to_check:list, include_numpy_flag:bool = False) -> bool:
	# Determine if the entries of the provided list are all numeric
	# Verify the inputs
	assert type(list_to_check) == list, "isListWithNumericEntries: Provided value for 'list_to_check' must be a list object"
	
	# Check the needed type(s), break if not
	for entry in list_to_check:
		if isNumeric(entry, include_numpy_flag = include_numpy_flag) == False:
			return False
	
	# Return True because break didn't occur
	return True
	
# String type
def isListWithStringEntries(list_to_check:list, allow_empty_flag:bool = True) -> bool:
	# Determine if the entries of the provided list are all strings
	# Verify the inputs
	assert type(list_to_check) == list, "isListWithStringEntries: Provided value for 'list_to_check' must be a list object"
	
	# Check the needed type(s), break if not
	for entry in list_to_check:
		if type(entry) != str:
			return False
		elif allow_empty_flag == False and len(entry) == 0:
			return False
	
	# Return True because break didn't occur
	return True
	
# Tuple type
def isListWithTupleEntries(list_to_check:list, allow_empty_flag:bool = True) -> bool:
	# Determine if the entries of the provided list are all tuples
	# Verify the inputs
	assert type(list_to_check) == list, "isListWithTupleEntries: Provided value for 'list_to_check' must be a list object"
	
	# Check the needed type(s), break if not
	for entry in list_to_check:
		if type(entry) != tuple:
			return False
		elif allow_empty_flag == False and len(entry) == 0:
			return False
	
	# Return True because break didn't occur
	return True
	
# List type
def isListWithListEntries(list_to_check:list, allow_empty_flag:bool = True) -> bool:
	# Determine if the entries of the provided list are all lists
	# Verify the inputs
	assert type(list_to_check) == list, "isListWithListEntries: Provided value for 'list_to_check' must be a list object"
	
	# Check the needed type(s), break if not
	for entry in list_to_check:
		if type(entry) != list:
			return False
		elif allow_empty_flag == False and len(entry) == 0:
			return False
	
	# Return True because break didn't occur
	return True
	
# Dictionary type
def isListWithDictionaryEntries(list_to_check:list, allow_empty_flag:bool = True) -> bool:
	# Determine if the entries of the provided list are all dictionaries
	# Verify the inputs
	assert type(list_to_check) == list, "isListWithDictionaryEntries: Provided value for 'list_to_check' must be a list object"
	
	# Check the needed type(s), break if not
	for entry in list_to_check:
		if type(entry) != dict:
			return False
		elif allow_empty_flag == False and len(entry) == 0:
			return False
	
	# Return True because break didn't occur
	return True
