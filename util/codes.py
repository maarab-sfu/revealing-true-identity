"""
This file contains constant fields that are used by different modules. The last two (swir_rect and rgb_rect) are not used anymore. 
The first two specify the ground truth label values (used in driver.py when we read the ground truth files) and the next two specify the ground truth labels used internally by the 
dataloader (h5_dataset.py)
"""

MAKEUP_ATTACK = "m50110004"
# MAKEUP_ATTACK = "m5"
BONA_FIDE = "m00000000"
BF = 0
ATTACK = 1
swir_rect = [[100,70,110,130],[0,0,63,63]]
rgb_rect = [[265,377,660,948],[250,653,531,627]]
