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
def privacyDecorator(attribute_names:list, deepcopy_flag:bool = False):
	### Verify the inputs ###
	assert isListWithStringEntries(attribute_names, allow_empty_flag = False) == True, "privacyDecorator: Provided value for 'attribute_names' must be a list object with non-empty str objects as entries"
	assert len(attribute_names) > 0, "privacyDecorator: Provided value for 'attribute_names' must be a list of length > 0"
	assert len(set(attribute_names)) == len(attribute_names), "privacyDecorator: Provided value for 'attribute_names' must be a list of unique entries"
	assert type(deepcopy_flag) == bool, "privacyDecorator: Provided value for 'deepcopy_flag' must be a bool object"

	### Define the part of the decorator which receives information related to the actual class ###
	def implementPrivacy(class_to_decorate:type):
		# Implement the various privacy features needed for getting and setting class values
		# Verify the inputs
		assert type(class_to_decorate) == type, "privacyDecorator::implementPrivacy: Provided value for 'class_to_decorate' must be a class (i.e. this decorator must be attached to a class definition)"
		assert hasattr(class_to_decorate, "__getattribute__"), "privacyDecorator::implementPrivacy: Provided value for 'class_to_decorate' must have '__getattribute__' as an attribute"
		assert hasattr(class_to_decorate, "__setattr__"), "privacyDecorator::implementPrivacy: Provided value for 'class_to_decorate' must have '__setattr__' as an attribute"
		assert hasattr(class_to_decorate, "__delattr__"), "privacyDecorator::implementPrivacy: Provided value for 'class_to_decorate' must have '__delattr__' as an attribute"

		# Retrieve the needed magic methods from the class
		# Always get the frist 3 magic methods
		class_to_decorate.original__getattribute__ = class_to_decorate.__getattribute__
		class_to_decorate.original__setattr__ = class_to_decorate.__setattr__
		class_to_decorate.original__delattr__ = class_to_decorate.__delattr__
		# Only get the last one if it exists
		if hasattr(class_to_decorate, "__getattr__") == True:
			class_to_decorate.original__getattr__ = class_to_decorate.__getattr__

		# Define a shared function used for determining if access is allowed
		def determineIfAccessAllowed(attribute_name:str) -> bool:
			return True

		# Define the function implementing the modified __getattribute__ functionality
		def modified__getattribute__(self, attribute_name:str) -> Any:
			# Return the original function's value if access is allowed, otherwise raise an error
			if determineIfAccessAllowed(attribute_name = attribute_name) == True:
				return class_to_decorate.original__getattribute__(self, attribute_name)
			else:
				raise AttributeError("privacyDecorator::implementPrivacy: Unable to call __getattribute__ on the attribute named '" + attribute_name + "' due to it being private")

		# Define the function implementing the modified __getattr__ functionality
		def modified__getattr__(self, attribute_name:str) -> Any:
			# Return the original function's value if access is allowed, otherwise raise an error
			if determineIfAccessAllowed(attribute_name = attribute_name) == True:
				return class_to_decorate.original__getattr__(self, attribute_name)
			else:
				raise AttributeError("privacyDecorator::implementPrivacy: Unable to call __getattr__ on the attribute named '" + attribute_name + "' due to it being private")

		# Define the function implementing the modified __setattr__ functionality
		def modified__setattr__(self, attribute_name:str, attribute_value:Any):
			# Return the original function's value if access is allowed, otherwise raise an error
			if determineIfAccessAllowed(attribute_name = attribute_name) == True:
				return class_to_decorate.original__setattr__(self, attribute_name, attribute_value)
			else:
				raise AttributeError("privacyDecorator::implementPrivacy: Unable to call __setattr__ on the attribute named '" + attribute_name + "' due to it being private")

		# Define the function implementing the modified __delattr__ functionality
		def modified__delattr__(self, attribute_name:str):
			# Return the original function's value if access is allowed, otherwise raise an error
			if determineIfAccessAllowed(attribute_name = attribute_name) == True:
				return class_to_decorate.original__delattr__(self, attribute_name)
			else:
				raise AttributeError("privacyDecorator::implementPrivacy: Unable to call __delattr__ on the attribute named '" + attribute_name + "' due to it being private")

		# Replace the class's original attributes with the modified ones
		# Always set the first 3 magic methods
		class_to_decorate.__getattribute__ = modified__getattribute__
		class_to_decorate.__setattr__ = modified__setattr__
		class_to_decorate.__delattr__ = modified__delattr__
		# Only get the last one if it exists
		if hasattr(class_to_decorate, "__getattr__") == True:
			class_to_decorate.__getattr__ = modified__getattr__

		# Return the modified class object
		return class_to_decorate

	### Return the middle function implementing the privacy rules ###
	return implementPrivacy

@privacyDecorator(["_value"])
class Test:
	def __init__(self):
		self.sddlgknasflkgnaldfkfnlkdfnh = 0

		print("TEST 1:")
		self.asfbasfnomomomen()

	def asfbasfnomomomen(self):
		frame = currentframe().f_back
		print("    Section 1:")
		print("       ", frame.f_code.co_name)
		print("       ", frame.f_locals)

		print("    Section 2:")
		for frame in getouterframes(currentframe()):
			print("       ", frame)
		return None

a = Test()
print("TEST 2:")
a.asfbasfnomomomen()