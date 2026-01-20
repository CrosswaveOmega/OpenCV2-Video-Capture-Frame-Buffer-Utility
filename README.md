# Video Capture Frame Screenshot Utility
Keeps a buffer of video frames taken from a capture card so the user can select and save specific frames as png screenshots through a gui interface.

# Use Case:
I needed software that could take in a live video feed from a capture card, determine how different each frame was to the other, and allow me to save chosen differing frames into an output file while dynamically changing the folder to be saved to with a key press.

Simply recording a video with OBS and getting stills from the feed would have been impractical; I needed crisp frames for a wiki in a very fast paced title.

Frame difference is determined by downscaling each image to grayscale and then determining how much of a shift there is between grayscale frames.



## Requirements

Requires Python 3.12 at minimum

** opencv-python
** numpy
** ffmpeg
### Usage
* Create a new virtual enviornment with venv, and install opencv-python and numpy.
* Start the video_frame_buffer script, select your capture card in the dropdown.
* If movement is happening in the capture card's feed, frames will start appearing in the window that pops up. 
* Click a frame to save it as a png in the specified folder within video_frame_buffer_output
 * You have to enter a name with your keyboard before the image can be saved.

### CONFIG FILE PARAMETERS
[FrameViewer]
* name = string, The name of folder of the name to save
* file_prefix = string, What each saved image's file name will start with
* grid_rows = integer, how many visible rows there should be in the Buffer Window
* grid_cols = integer, how many columns of frames there should be in the buffer window
* change_threshold = float from 0.0 to 1.0, how different each frame should be from each other to warrant being added to the buffer.  set this to 0 to show every frame.
* pixel_change_min = integer, this is the minimum shift in grayscale value for each pixel to count as "changed".  Depending on your caputre card, you may need to set this to 1.
* pixel_change_max = integer, this is the maximum shift in grayscale value for a pixel to count as "changed".  Leave this at 256.
* history_scrolling_limit = integer, how many rows should be stored 

