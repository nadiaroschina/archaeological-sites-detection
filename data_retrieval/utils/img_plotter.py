import matplotlib.pyplot as plt

import sys
sys.path.append('../utils')

from tile_downloader import download_map_tile


def plot_tiles(lon, lat):

    imgs = []
    for z in range(21):
        try:
            imgs.append(download_map_tile(lon=lon, lat=lat, z=z, pixel_size=256))
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
