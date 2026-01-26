##########################################
### Import needed general dependencies ###
##########################################
# Internal modules
from tkinter_helper import askOpenFilename, askSaveFilename

# External modules
import matplotlib.pyplot as plt


#####################################################################################
### Define the functionality needed for loading a LaTeX equation and rendering it ###
#####################################################################################
# Tell the matplotlib engine the needed LaTeX settings
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath, amssymb, amsfonts}"})

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
# Set the figure size using the bounding box
fig.set_size_inches(bbox.width + 1, bbox.height + 0.5)

# Get the filename to save to and end early if not selected
png_filename_path = askSaveFilename(allowed_extensions = ["png"])
assert png_filename_path is not None, "Unable to proceed with saving rendered equation because no filename was selected"

# Save the figure to the needed file using a transparent background
plt.savefig(png_filename_path, dpi = 300, transparent = True)