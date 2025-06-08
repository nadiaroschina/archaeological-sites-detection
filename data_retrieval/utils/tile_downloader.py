import numpy as np
import requests

from PIL import Image
import io

def get_xy_coords(lat, lon, z):

    elip = 0.0818191908426

    beta = np.pi * lat / 180
    phi = (1 - elip * np.sin(beta)) / (1 + elip * np.sin(beta))
    theta = np.tan(np.pi / 4 +  beta / 2) * np.float_power(phi, elip / 2)
    
    rho = np.float_power(2, z + 7)
    x_p = rho * (1 + lon / 180)
    y_p = rho * (1 - np.log(theta) / np.pi)

    x = int(np.floor(x_p / 256))
    y = int(np.floor(y_p / 256))

    return x, y


def download_map_tile(lat, lon, z, pixel_size=256):

    x, y = get_xy_coords(lat, lon, z)
    scale = pixel_size / 256

    url = f'https://core-sat.maps.yandex.net/tiles?&x={x}&y={y}&z={z}&scale={scale}&lang=en_RU&client_id=yandex-web-maps'
    response = requests.get(url)

    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception
