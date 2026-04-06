##########################################
### Import needed general dependencies ###
##########################################
# Internal modules
from type_helper import isListWithStringEntries

# External modules
from inspect import currentframe, getouterframes
from typing import Any


#####################################################################
### Define a decorator used to make attributes of a class private ###
#####################################################################
def privacyDecorator(attribute_names:list, deepcopy_flag:bool = True):
	### Verify the inputs ###
	assert isListWithStringEntries(attribute_names, allow_empty_flag = False) == True, "privacyDecorator: Provided value for 'attribute_names' must be a list object with non-empty str objects as entries"
	assert len(set(attribute_names)) == len(attribute_names), "privacyDecorator: Provided value for 'attribute_names' must be a list of unique entries"
	assert type(deepcopy_flag) == bool, "privacyDecorator: Provided value for 'deepcopy_flag' must be a bool object"

	### Define the part of the decorator which receives information related to the actual class ###
	def implementPrivacy(class_to_decorate:type):
		# Implement the various privacy features needed for getting and setting class values
		# Verify the inputs
		assert hasattr(class_to_decorate, "__getattribute__"), "privacyDecorator::implementPrivacy: Provided value for 'class_to_decorate' must have '__getattribute__' as an attribute"
		assert hasattr(class_to_decorate, "__setattr__"), "privacyDecorator::implementPrivacy: Provided value for 'class_to_decorate' must have '__setattr__' as an attribute"
		assert hasattr(class_to_decorate, "__delattr__"), "privacyDecorator::implementPrivacy: Provided value for 'class_to_decorate' must have '__delattr__' as an attribute"

		# Retrieve the needed magic methods from the class
		# Always get the frist 3 magic methods
		class_to_decorate.__original__getattribute__ = class_to_decorate.__getattribute__
		class_to_decorate.__original__setattr__ = class_to_decorate.__setattr__
		class_to_decorate.__original__delattr__ = class_to_decorate.__delattr__
		# Only get the last one if it exists
		if hasattr(class_to_decorate, "__getattr__") == True:
			class_to_decorate.__original__getattr__ = class_to_decorate.__getattr__

		# Define a shared function used for determining if access is allowed
		def _determineIfAccessAllowed(attribute_name:str) -> bool:
			# Determine if gatekeeping is necessary for the given attribute
			# Verify the inputs
			assert type(attribute_name) == str, "privacyDecorator::implementPrivacy::_determineIfAccessAllowed: Provided value for 'attribute_name' must be a str object"

			# Get the current frame of the program
			current_frame = currentframe()

			# End early with a True value if the caller of the operation was itself class which is being decorated
			# Note: Need two levels of back here due to _determineIfAccessAllowed being shared by the modified class methods
			calling_frame_local_variables = current_frame.f_back.f_back.f_locals
			if "self" in calling_frame_local_variables:
				if isinstance(calling_frame_local_variables["self"], class_to_decorate) == True:
					return True

			# End early with a True value if the operation is related to an allowed deepcopy operation
			if deepcopy_flag == True:
				for previous_frame in getouterframes(current_frame):
					if previous_frame.filename.endswith("copy.py") and previous_frame.function == "deepcopy":
						return True

			# End early with a False value if the attribute is protected from external access
			# Handle the case of being in the explicitly defined list of private names
			if attribute_name in attribute_names:
				return False
			# Handle the cases of implicitly defined private names (i.e. double underscore or name mangling)
			if attribute_name.startswith("__"):
				return False
			if attribute_name.startswith("_" + class_to_decorate.__name__ + "__"):
				return False

			# Return True to indicate that all checks have passed
			return True

		# Define the function implementing the modified __getattribute__ functionality
		def __modified__getattribute__(self, attribute_name:str) -> Any:
			# Return the original function's value if access is allowed, otherwise raise an error
			if determineIfAccessAllowed(attribute_name = attribute_name) == True:
				return class_to_decorate.__original__getattribute__(self, attribute_name)
			else:
				raise AttributeError("privacyDecorator::implementPrivacy::__modified__getattribute__: Unable to call '__getattribute__' on the attribute named '" + attribute_name + "' due to it being private")

		# Define the function implementing the modified __getattr__ functionality
		def __modified__getattr__(self, attribute_name:str) -> Any:
			# Return the original function's value if access is allowed, otherwise raise an error
			if determineIfAccessAllowed(attribute_name = attribute_name) == True:
				return class_to_decorate.__original__getattr__(self, attribute_name)
			else:
				raise AttributeError("privacyDecorator::implementPrivacy::__modified__getattr__: Unable to call '__getattr__' on the attribute named '" + attribute_name + "' due to it being private")

		# Define the function implementing the modified __setattr__ functionality
		def __modified__setattr__(self, attribute_name:str, attribute_value:Any):
			# Return the original function's value if access is allowed, otherwise raise an error
			if determineIfAccessAllowed(attribute_name = attribute_name) == True:
				return class_to_decorate.__original__setattr__(self, attribute_name, attribute_value)
			else:
				raise AttributeError("privacyDecorator::implementPrivacy::__modified__setattr__: Unable to call '__setattr__' on the attribute named '" + attribute_name + "' due to it being private")

		# Define the function implementing the modified __delattr__ functionality
		def __modified__delattr__(self, attribute_name:str):
			# Return the original function's value if access is allowed, otherwise raise an error
			if determineIfAccessAllowed(attribute_name = attribute_name) == True:
				return class_to_decorate.__original__delattr__(self, attribute_name)
			else:
				raise AttributeError("privacyDecorator::implementPrivacy::__modified__delattr__: Unable to call '__delattr__' on the attribute named '" + attribute_name + "' due to it being private")

		# Replace the class's original attributes with the modified ones
		# Always set the first 3 magic methods
		class_to_decorate.__getattribute__ = __modified__getattribute__
		class_to_decorate.__setattr__ = __modified__setattr__
		class_to_decorate.__delattr__ = __modified__delattr__
		# Only set the last one if it exists
		if hasattr(class_to_decorate, "__getattr__") == True:
			class_to_decorate.__getattr__ = __modified__getattr__

		# Return the modified class object
		return class_to_decorate

	### Return the middle function implementing the privacy rules ###
	return implementPrivacy