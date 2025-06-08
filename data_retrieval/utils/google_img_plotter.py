import matplotlib.pyplot as plt

import sys
sys.path.append('../utils')

from google_tile_downloader import download_google_map_tile


def plot_google_tiles(lon, lat, api_key, session_token):

    imgs = []
    for z in range(23):
        try:
            imgs.append(download_google_map_tile(lon=lon, lat=lat, z=z, api_key=api_key, session_token=session_token))
        except Exception:
            pass

    fig = plt.figure(figsize=(30, 30))
    for z in range(21):
        if z < len(imgs):
            fig.add_subplot(5, 5, z + 1) 
            plt.imshow(imgs[z]) 
            plt.axis('off') 
            plt.title(f'{z=}') 
    plt.show()
