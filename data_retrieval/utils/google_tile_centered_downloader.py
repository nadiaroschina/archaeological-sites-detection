import numpy as np
import requests

from PIL import Image
import io

from typing import Tuple


def mercator_project(lat, lon) -> Tuple[float, float]:
    
    mercator = -np.log(np.tan((0.25 + lat / 360) * np.pi))

    xp = 256 * (lon / 360 + 0.5)
    yp = 128 * (1 +  mercator / np.pi)

    return xp, yp


def inverse_mercator_project(xp, yp) -> Tuple[float, float]:
  
  lon = 360 * (xp / 256 - 0.5)

  mercator = (yp / 128 - 1) * np.pi
  lat = (np.arctan(np.exp(-mercator)) / np.pi - 0.25) * 360


  return lat, lon


def get_tile_xy(lat, lon, z) -> Tuple[float, float]:

    xp, yp = mercator_project(lat, lon)

    scale = 2 ** z

    x = int(np.floor(xp * scale / 256))
    y = int(np.floor(yp * scale / 256))

    return x, y


def download_google_map_tile(x, y, z, session_token, api_key):

    url = f'https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}'
    params = {
        'session': session_token,
        'key': api_key
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception
    

def get_tile_center_coords(lat, lon, z):

    x, y = get_tile_xy(lat, lon, z)
    scale = 2 ** z
    lat_tilecenter, lon_tilecenter = inverse_mercator_project((x + 0.5) * 256 / scale, (y + 0.5) * 256 / scale)

    return lat_tilecenter, lon_tilecenter


def download_centered_image(lat, lon, z, session_token, api_key):

    x, y = get_tile_xy(lat, lon, z)

    scale = 2 ** z
    lat_tilecenter, lon_tilecenter = inverse_mercator_project((x + 0.5) * 256 / scale, (y + 0.5) * 256 / scale)

    tiles = []
    for xt in range(x - 1, x + 2):
        for yt in range(y - 1, y + 2):
            tiles.append(download_google_map_tile(xt, yt, z, session_token, api_key))

    merged_image = Image.new('RGB', (256 * 3, 256 * 3))
    for i, tile in enumerate(tiles):
        merged_image.paste(tile, ((i // 3) * 256, (i % 3) * 256))

    xp, yp = mercator_project(lat, lon)
    xp, yp = xp * scale, yp * scale

    xp_tile, yp_tile = mercator_project(lat_tilecenter, lon_tilecenter)
    xp_tile, yp_tile = xp_tile * scale, yp_tile * scale

    offset_x, offset_y = 256 - xp_tile + xp,  256 - yp_tile + yp
    final_image = merged_image.crop((offset_x, offset_y, offset_x + 256, offset_y + 256))

    return final_image