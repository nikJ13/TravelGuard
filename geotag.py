from PIL.ExifTags import TAGS
from PIL import Image
from urllib.request import urlopen
import cv2
import numpy as np
url='http://192.168.0.101:8080/shot.jpg'

def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()






from PIL.ExifTags import GPSTAGS

def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging

def get_decimal_from_dms(dms, ref):

    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1] / 60.0
    seconds = dms[2][0] / dms[2][1] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)

def get_coordinates(geotags):
    lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])

    lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

    return (lat,lon)


while True:
	imageResp=urlopen(url)
	imgNp=np.array(bytearray(imageResp.read()),dtype=np.uint8)
	frame1=cv2.imdecode(imgNp,-1)
	cv2.imwrite('test2.jpg',frame1)
	exif = get_exif('test2.jpg')
	geotags = get_geotagging(exif)
	print(get_coordinates(geotags))
