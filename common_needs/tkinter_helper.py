##########################################
### Import needed general dependencies ###
##########################################
# Internal modules
from color_helper import RGB
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


#####################################################################
### Define functions which allow for creating frames and canvases ###
#####################################################################
def createWindow(width_parameter:Any, height_parameter:Any, title:str, resizable_flag:bool = False) -> Tk:
	# Create a tkinter frame object with the given parameters
	# Verify the inputs
	assert isNumeric(width_parameter, include_numpy_flag = False) == True, "createWindow: Provided value for 'width_parameter' must be a float or int object"
	assert isNumeric(height_parameter, include_numpy_flag = False) == True, "createWindow: Provided value for 'height_parameter' must be a float or int object"
	assert type(title) == str, "createWindow: Provided value for 'title' must be a str object"
	assert type(resizable_flag) == bool, "createWindow: Provided value for 'resizable_flag' must be a bool object"

	# Create the window object to return
	window_to_return = Tk()

	# Fetch the width and height of the screen on which the window appears
	screen_width = window_to_return.winfo_screenwidth()
	screen_height = window_to_return.winfo_screenheight()

	# Compute the window dimensions based on whether the parameters are floats or ints
	# Handle the width information
	if type(width_parameter) == float:
		# Compute window width as a portion of the screen width
		window_width = int(width_parameter * screen_width)
	else:
		# Set the window width as the raw provided value
		window_width = width_parameter
	# Handle the height information
	if type(height_parameter) == float:
		# Compute window height as a portion of the screen height
		window_height = int(height_parameter * screen_height)
	else:
		# Set the window height as the raw provided value
		window_height = height_parameter

	# Verify that the computed window dimensions are valid
	assert 0 < window_width and window_width <= screen_width, "createWindow: Provided value for 'width_parameter' must result in a positive window width which is <= the associated screen width"
	assert 0 < window_height and window_height <= screen_height, "createWindow: Provided value for 'height_parameter' must result in a positive window height which is <= the associated screen height"

	# Set the dimensions and title of the window
	window_to_return.geometry(str(window_width) + "x" + str(window_height))
	window_to_return.title(title)

	# Set the resizability parameter of the window
	window_to_return.resizable(resizable_flag, resizable_flag)

	# Update the window to make sure the updates apply
	window_to_return.update_idletasks()

	# Return the results
	return window_to_return

def createCanvas(used_window:Tk, fill_color:RGB) -> Canvas:
	# Create a canvas on the associated window
	# Verify the inputs
	assert type(used_window) == Tk, "createCanvas: Provided value for 'used_window' must be a Tk object"
	assert type(fill_color) == RGB, "createCanvas: Provided value for 'fill_color' must be an RGB object"

	# Fetch the width and height of the window
	window_width = used_window.winfo_width()
	window_height = used_window.winfo_height()

	# Create the canvas object to return
	canvas_to_return = Canvas(used_window, width = window_width, height = window_height)

	# Set the color of the canvas
	canvas_to_return.configure(bg = fill_color.asStringHex())

	# Pack the canvas object
	canvas_to_return.pack()

	# Update the associated window to make sure the updates apply
	canvas_to_return.winfo_toplevel().update_idletasks()

	# Return the results
	return canvas_to_return


####################################################
### Define functions for creating canvas objects ###
####################################################
def createRectangle(used_canvas:Canvas, tl_x_parameter:Any, tl_y_parameter:Any, br_x_parameter:Any, br_y_parameter:Any, fill_color:RGB):
	# Create a rectangle object with the given parameters on the given canvas
	# Verify the inputs
	assert type(used_canvas) == Canvas, "createRectangle: Provided value for 'used_canvas' must be a Canvas object"
	assert isNumeric(tl_x_parameter, include_numpy_flag = False) == True, "createRectangle: Provided value for 'tl_x_parameter' must be a float or int object"
	assert isNumeric(tl_y_parameter, include_numpy_flag = False) == True, "createRectangle: Provided value for 'tl_y_parameter' must be a float or int object"
	assert isNumeric(br_x_parameter, include_numpy_flag = False) == True, "createRectangle: Provided value for 'br_x_parameter' must be a float or int object"
	assert isNumeric(br_y_parameter, include_numpy_flag = False) == True, "createRectangle: Provided value for 'br_y_parameter' must be a float or int object"
	assert type(fill_color) == RGB, "createRectangle: Provided value for 'fill_color' must be an RGB object"

	# Fetch the width and height of the canvas
	canvas_width = used_canvas.winfo_width()
	canvas_height = used_canvas.winfo_height()

	# Compute the rectangle vertex locations based on whether the parameters are floats or ints
	# Handle the top-left x-value
	if type(tl_x_parameter) == float:
		# Compute x-value as a portion of the canvas width
		tl_x_value = int(tl_x_parameter * canvas_width)
	else:
		# Set the x-value as the raw provided value
		tl_x_value = tl_x_parameter
	# Handle the top-left y-value
	if type(tl_y_parameter) == float:
		# Compute y-value as a portion of the canvas height
		tl_y_value = int(tl_y_parameter * canvas_height)
	else:
		# Set the y-value as the raw provided value
		tl_y_value = tl_y_parameter
	# Handle the bottom-right x-value
	if type(br_x_parameter) == float:
		# Compute x-value as a portion of the canvas width
		br_x_value = int(br_x_parameter * canvas_width)
	else:
		# Set the x-value as the raw provided value
		br_x_value = br_x_parameter
	# Handle the bottom-right y-value
	if type(br_y_parameter) == float:
		# Compute y-value as a portion of the canvas height
		br_y_value = int(br_y_parameter * canvas_height)
	else:
		# Set the y-value as the raw provided value
		br_y_value = br_y_parameter

	# Verify that the computed rectangle vertex locations are valid
	assert 0 <= tl_x_value and tl_x_value <= canvas_width, "createFrame: Provided value for 'tl_x_parameter' must result in a non-negative top-left x-value which is <= the associated canvas width"
	assert 0 <= tl_y_value and tl_y_value <= canvas_height, "createFrame: Provided value for 'tl_y_parameter' must result in a non-negative top-left y-value which is <= the associated canvas height"
	assert 0 <= br_x_value and br_x_value <= canvas_width, "createFrame: Provided value for 'br_x_parameter' must result in a non-negative top-left x-value which is <= the associated canvas width"
	assert 0 <= br_y_value and br_y_value <= canvas_height, "createFrame: Provided value for 'br_y_parameter' must result in a non-negative top-left y-value which is <= the associated canvas height"
	assert tl_x_value < br_x_value, "createFrame: Provided values for 'tl_x_parameter' and 'br_x_parameter' must result in a top-left x-value which is < the bottom-right x-value"
	assert tl_y_value < br_y_value, "createFrame: Provided values for 'tl_y_parameter' and 'br_y_parameter' must result in a top-left y-value which is < the bottom-right y-value"

	# Draw the needed rectangle object
	used_canvas.create_rectangle(tl_x_value, tl_y_value, br_x_value, br_y_value, fill = fill_color.asStringHex())