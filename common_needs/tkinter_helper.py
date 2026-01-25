##########################################
### Import needed general dependencies ###
##########################################
# Internal modules
from type_helper import isListWithStringEntries, isNumeric

# External modules
from pathlib import Path, PosixPath, WindowsPath
from tkinter import Canvas, Tk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename
from typing import Any, Union


##################################################################
### Define functions for getting paths using file dialog boxes ###
##################################################################
def askDirectory(title:str = "Please Select A Directory To Open") -> Union[PosixPath, WindowsPath]:
	# Create a pathlib object referring to an existing local directory (return None if none selected)
	# Verify the inputs
	assert type(title) == str, "askDirectory: Provided value for 'title' must be a str object"

	# Make sure a separate tkinter window doesn't open
	Tk().withdraw()
	
	# Use the file dialog to ask for a directory to open
	directory_str = askdirectory(title = title)
	
	# Handle the various cases
	if len(directory_str) == 0:
		# Cancel button clicked, return None
		return None
	else:
		# Convert the directory string to a pathlib object and return it
		return Path(directory_str)

def askOpenFilename(allowed_extensions:list = None, title:str = "Please Select A File To Open") -> Union[PosixPath, WindowsPath]:
	# Create a pathlib object referring to an existing local file (return None if none selected)
	# Verify the inputs
	if allowed_extensions is not None:
		assert isListWithStringEntries(allowed_extensions, allow_empty_flag = False) == True, "askOpenFilename: If provided, value for 'allowed_extensions' must be a list object with non-empty strings as entries"
		assert len(allowed_extensions) > 0, "askOpenFilename: If provided, value for 'allowed_extensions' must be a non-empty list"
		for extension in allowed_extensions:
			for letter in extension.lower():
				assert letter in "abcdefghijklmnopqrstuvwxyz", "askOpenFilename: If provided, value for 'allowed_extensions' must have entries containing only the letters a through z"
	assert type(title) == str, "askOpenFilename: Provided value for 'title' must be a str object"

	# Construct the list of allowed extensions as needed by tkinter
	# Initialize the list
	tk_allowed_extensions = []
	# Add in the provided types (if needed)
	if allowed_extensions is not None:
		for extension in allowed_extensions:
			tk_allowed_extensions.append((extension.upper() + " Files", "*." + extension.lower()))
	# Add in all file extensions (if needed)
	if allowed_extensions is None:
		tk_allowed_extensions.append(("All Files", "*.*"))
	# Convert to a tuple
	tk_allowed_extensions = tuple(tk_allowed_extensions)
	
	# Make sure a separate tkinter window doesn't open
	Tk().withdraw()
	
	# Use the file dialog to ask for a filename to open
	filename_str = askopenfilename(filetypes = tk_allowed_extensions, title = title)
	
	# Make sure the filename ends in an allowed extension (if needed)
	if allowed_extensions is not None:
		# Initialize a flag indicating if a valid extension was found
		extension_found_flag = False
		# Search for a valid extension
		for extension in allowed_extensions:
			if filename_str.lower().endswith(extension) == True:
				extension_found_flag = True
				break
		# Raise error if not found
		assert extension_found_flag == True, "askOpenFilename: Selected filename doesn't end with an allowed extension"
	
	# Handle the various cases
	if len(filename_str) == 0:
		# Cancel button clicked, return None
		return None
	else:
		# Convert the filename string to a pathlib object and return it
		return Path(filename_str)

def askSaveFilename(allowed_extensions:list = None, title:str = "Please Select A Save Destination") -> Union[PosixPath, WindowsPath]:
	# Create a pathlib object referring to a new local file (return None if none selected)
	# Verify the inputs
	if allowed_extensions is not None:
		assert isListWithStringEntries(allowed_extensions, allow_empty_flag = False) == True, "askSaveFilename: If provided, value for 'allowed_extensions' must be a list object with non-empty strings as entries"
		assert len(allowed_extensions) > 0, "askSaveFilename: If provided, value for 'allowed_extensions' must be a non-empty list"
		for extension in allowed_extensions:
			for letter in extension.lower():
				assert letter in "abcdefghijklmnopqrstuvwxyz", "askSaveFilename: If provided, value for 'allowed_extensions' must have entries containing only the letters a through z"
	assert type(title) == str, "askSaveFilename: Provided value for 'title' must be a str object"

	# Construct the list of allowed extensions as needed by tkinter
	# Initialize the list
	tk_allowed_extensions = []
	# Add in the provided types (if needed)
	if allowed_extensions is not None:
		for extension in allowed_extensions:
			tk_allowed_extensions.append((extension.upper() + " Files", "*." + extension.lower()))
	# Add in all file extensions (if needed)
	if allowed_extensions is None:
		tk_allowed_extensions.append(("All Files", "*.*"))
	# Convert to a tuple
	tk_allowed_extensions = tuple(tk_allowed_extensions)
	
	# Make sure a separate tkinter window doesn't open
	Tk().withdraw()
	
	# Use the file dialog to ask for a filename to save to
	filename_str = asksaveasfilename(filetypes = tk_allowed_extensions, title = title)
	
	# Make sure the filename ends in an allowed extension (if needed)
	if allowed_extensions is not None:
		# Initialize a flag indicating if a valid extension was found
		extension_found_flag = False
		# Search for a valid extension
		for extension in allowed_extensions:
			if filename_str.lower().endswith(extension) == True:
				extension_found_flag = True
				break
		# Raise error if not found
		assert extension_found_flag == True, "askSaveFilename: Selected filename doesn't end with an allowed extension"
	
	# Handle the various cases
	if len(filename_str) == 0:
		# Cancel button clicked, return None
		return None
	else:
		# Convert the filename string to a pathlib object and return it
		return Path(filename_str)


#########################################################################################
### Define functions which allow for creation and modification of frames and canvases ###
#########################################################################################
def createFrame(width_parameter:Any, height_parameter:Any, title:str, resizable_flag:bool) -> Tk:
	# Create a tkinter frame object with the given parameters
	# Verify the inputs
	assert isNumeric(width_parameter, include_numpy_flag = False) == True, "createFrame: Provided value for 'width_parameter' must be a float or int object"
	assert isNumeric(height_parameter, include_numpy_flag = False) == True, "createFrame: Provided value for 'height_parameter' must be a float or int object"
	assert type(title) == str, "createFrame: Provided value for 'title' must be a str object"
	assert type(resizable_flag) == str, "createFrame: Provided value for 'resizable_flag' must be a bool object"

	# Create the frame object to return
	frame_to_return = Tk()

	# Fetch the width and height of the screen on which the frame appears
	screen_width = frame_to_return.winfo_screenwidth()
	screen_height = frame_to_return.winfo_screenheight()

	# Compute the frame dimensions based on whether the parameters are floats or ints
	# Handle the width information
	if type(width_parameter) == float:
		# Compute frame width as a portion of the screen width
		frame_width = int(width_parameter * screen_width)
	else:
		# Set the frame width as the raw provided value
		frame_width = width_parameter
	# Handle the height information
	if type(height_parameter) == float:
		# Compute frame height as a portion of the screen height
		frame_height = int(height_parameter * screen_height)
	else:
		# Set the frame height as the raw provided value
		frame_height = height_parameter

	# Verify that the computed frame dimensions are valid
	assert 0 < frame_width and frame_width <= screen_width, "createFrame: Provided value for 'width_parameter' must result in a positive frame width which is <= the associated screen width"
	assert 0 < frame_height and frame_height <= screen_height, "createFrame: Provided value for 'height_parameter' must result in a positive frame height which is <= the associated screen height"

	# Set the dimensions and title of the frame
	frame_to_return.geometry(str(frame_width) + "x" + str(frame_height))
	frame_to_return.title(title)

	# Set the resizability parameter of the frame
	frame_to_return.resizable(resizable_flag, resizable_flag)

	# Return the results
	return frame_to_return

def createCanvas(frame:Tk) -> Canvas:
	pass


####################################################
### Define functions for creating canvas objects ###
####################################################