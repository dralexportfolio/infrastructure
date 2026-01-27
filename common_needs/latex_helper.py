##########################################
### Import needed general dependencies ###
##########################################
# Internal modules
from tkinter_helper import askOpenFilename, askSaveFilename

# External modules
import matplotlib.pyplot as plt


#########################################################
### Set any needed parameters for the equation render ###
#########################################################
# Set an optional fixed width and height for the figure
fixed_width = 12
fixed_height = None

# Define the render DPI
dpi = 300

# Tell the matplotlib engine the needed LaTeX settings
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "text.latex.preamble": r"\usepackage{amsmath, amssymb, amsfonts}"})


#####################################################################################
### Define the functionality needed for loading a LaTeX equation and rendering it ###
#####################################################################################
# Get a filename to load and end early if not selected
txt_filename_path = askOpenFilename(allowed_extensions = ["txt"])
assert txt_filename_path is not None, "Unable to proceed with rendering equation because no filename was selected"

# Read the string from the selected file, modifying it to remove new lines and unnecessary spaces
with open(txt_filename_path, "r", encoding = "utf-8") as file:
	latex_string = file.read().replace("\n", " ").strip()

# Create the figure and hide the axis
fig = plt.figure()
plt.axis("off")

# Display the text in the center of the figure as a raw string using LaTeX to make sure backslashes are treated correctly
rendered_text = plt.text(0.5, 0.5, r"${%s}$" % latex_string, ha = "center", va = "center", size = 20)

# Automatically adjust the size of the figure to match the displayed text
# Build the image without showing it
fig.canvas.draw()
# Get the bounding box of the rendered text
bbox = rendered_text.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# Compute the needed width and height based on the bounding box and provided fixed values
used_width = bbox.width + 1
used_height = bbox.height + 0.5
if fixed_width is not None:
	used_width = max(used_width, fixed_width)
if fixed_height is not None:
	used_height = max(used_height, fixed_height)
# Set the figure size using the computed values
fig.set_size_inches(used_width, used_height)

# Get the filename to save to and end early if not selected
png_filename_path = askSaveFilename(allowed_extensions = ["png"])
assert png_filename_path is not None, "Unable to proceed with saving rendered equation because no filename was selected"

# Save the figure to the needed file using a transparent background
plt.savefig(png_filename_path, dpi = dpi, transparent = True)