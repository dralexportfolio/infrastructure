##########################################
### Import needed general dependencies ###
##########################################
# External modules
from pathlib import Path, PosixPath, WindowsPath
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename
from typing import Union

# Internal modules
from type_helper import isListWithStringEntries


##################################################################
### Define functions for getting paths using file dialog boxes ###
##################################################################
def askDirectory() -> Union[PosixPath, WindowsPath]:
	# Create a pathlib object referring to an existing local directory (return None if none selected)
	# Make sure a separate tkinter window doesn't open
	Tk().withdraw()
	
	# Use the file dialog to ask for a directory to open
	directory_str = askdirectory(title = "Please Select A Directory")
	
	# Handle the various cases
	if len(directory_str) == 0:
		# Cancel button clicked, return None
		return None
	else:
		# Convert the directory string to a pathlib object and return it
		return Path(directory_str)

def askOpenFilename(allowed_extensions:list = None) -> Union[PosixPath, WindowsPath]:
	# Create a pathlib object referring to an existing local file (return None if none selected)
	# Verify the inputs
	if allowed_extensions is not None:
		assert isListWithStringEntries(allowed_extensions, allow_empty_flag = False) == True, "askOpenFilename: If provided, value for 'allowed_extensions' must be a list object with non-empty strings as entries"
		assert len(allowed_extensions) > 0, "askOpenFilename: If provided, value for 'allowed_extensions' must be a non-empty list"
		for extension in allowed_extensions:
			for letter in extension.lower():
				assert letter in "abcdefghijklmnopqrstuvwxyz", "askOpenFilename: If provided, value for 'allowed_extensions' must have entries containing only the letters a through z"
		
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
	filename_str = askopenfilename(title = "Please Select A File", filetypes = tk_allowed_extensions)
	
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

def askSaveFilename(allowed_extensions:list = None) -> Union[PosixPath, WindowsPath]:
	# Create a pathlib object referring to a new local file (return None if none selected)
	# Verify the inputs
	if allowed_extensions is not None:
		assert isListWithStringEntries(allowed_extensions, allow_empty_flag = False) == True, "askSaveFilename: If provided, value for 'allowed_extensions' must be a list object with non-empty strings as entries"
		assert len(allowed_extensions) > 0, "askSaveFilename: If provided, value for 'allowed_extensions' must be a non-empty list"
		for extension in allowed_extensions:
			for letter in extension.lower():
				assert letter in "abcdefghijklmnopqrstuvwxyz", "askSaveFilename: If provided, value for 'allowed_extensions' must have entries containing only the letters a through z"
		
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
	filename_str = asksaveasfilename(filetypes = tk_allowed_extensions)
	
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