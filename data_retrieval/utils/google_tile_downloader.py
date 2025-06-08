import numpy as np
import requests

from PIL import Image
import io

def get_point(lat, lon):
    mercator = -np.log(np.tan((0.25 + lat / 360) * np.pi))
    x = 256 * (lon / 360 + 0.5)
    y = 128 * (1 +  mercator / np.pi)
    return x, y


def get_google_xy_coords(lat, lon, z):

    mercator = -np.log(np.tan((0.25 + lat / 360) * np.pi))

    xp = 256 * (lon / 360 + 0.5)
    yp = 128 * (1 +  mercator / np.pi)
    scale = 2 ** z

    x = int(np.floor(xp * scale / 256))
    y = int(np.floor(yp * scale / 256))

    return x, y


def download_google_map_tile(lat, lon, z, session_token, api_key):

    x, y = get_google_xy_coords(lat, lon, z)

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
