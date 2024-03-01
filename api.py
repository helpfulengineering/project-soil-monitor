import pandas as pd
from landsatxplore.earthexplorer import EarthExplorer
import os
from landsatxplore.api import API
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
from xml.etree import ElementTree
import requests
from requests_futures.sessions import FuturesSession
import tarfile
from usgs import USGS_API, USGSError
from usgs import xsi, payloads
import json

# Your USGS  credentials
username = "xxxxxxxxx"
password = "xxxxxxxxx"

api = API(username, password)

# Search for Landsat TM scenes
scenes = api.search( dataset='landsat_ot_c2_l2', latitude=50.85, longitude=-4.35, start_date='2023-01-01', end_date='2023-10-01', max_cloud_cover=10)

api.logout()

print(f"{len(scenes)} scenes found.")
for scene in scenes:
    print(scene['acquisition_date'].strftime('%Y-%m-%d'))

# Create a DataFrame from the scenes
df_scenes = pd.DataFrame(scenes)
df_scenes = df_scenes[['display_id','wrs_path', 'wrs_row','satellite','cloud_cover','acquisition_date']]
df_scenes.sort_values('acquisition_date', ascending=False, inplace=True)
print(df_scenes)

i = int(input("Enter a scene ID :- "))
ee = EarthExplorer(username, password)

# Select the first scene
ID = scenes[i].get("display_id")

try: 
    ee.download(ID, output_dir='./data')
    print('{} succesful'.format(ID))
    
# Additional error handling
except:
    if os.path.isfile('./data/{}.tar'.format(ID)):
        print('{} error but file exists'.format(ID))
    else:
        print('{} error'.format(ID))

ee.logout()

tar = tarfile.open('./data/{}.tar'.format(ID))
tar.extractall('./data/{}'.format(ID))
tar.close()

# Load Blue (B2), Green (B3) and Red (B4) bands
B2 = tiff.imread('./data/{}/{}_SR_B2.TIF'.format(ID, ID))
B3 = tiff.imread('./data/{}/{}_SR_B3.TIF'.format(ID, ID))
B4 = tiff.imread('./data/{}/{}_SR_B4.TIF'.format(ID, ID))


# Stack and scale bands
RGB = np.dstack((B4, B3, B2))
RGB = np.clip(RGB*0.0000275-0.2, 0, 1)

# Clip to enhance contrast
RGB = np.clip(RGB,0,0.2)/0.2

# Display RGB image
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(RGB)
ax.set_axis_off()
plt.show()