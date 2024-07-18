import socket
import numpy as np
import os
import geopandas as gpd
import mercantile
import rasterio
import affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, calculate_default_transform
import h5py
from datetime import datetime, timedelta
import requests

username = os.environ["EARTHDATA_USER"]
password = os.environ["EARTHDATA_PASS"]

os.chdir(os.path.dirname(__file__))

ZOOM = 3
TILE_SIZE = 128
RESAMPLING = Resampling.bilinear
# Output stored under ROOT_PATH/raw/
#ROOT_PATH = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/daily_predictions/input_datasets/smap_36"
ROOT_PATH = "data"
os.makedirs(ROOT_PATH, exist_ok=True)

'''
    Downloads SMAP Soil moisture from 2 days back in .h5 format
    The download path changes every year (majorly the product version of the product)
'''
def download_smap_automatically(year=None, month=None, day=None, n_days_before=2):
    if year is None or month is None or day is None:
        current_date = datetime.now()
        year, month, day = (current_date - timedelta(days=n_days_before)).strftime("%Y-%m-%d").split("-")

    out_dir = ROOT_PATH + "/raw/"
    os.makedirs(out_dir, exist_ok=True)

    filename_v1 = get_filename(year, month, day, '001')
    filename_v2 = get_filename(year, month, day, '002')

    path_v1 = os.path.join(out_dir, filename_v1)
    path_v2 = os.path.join(out_dir, filename_v2)

    if os.path.isfile(path_v1) or os.path.isfile(path_v2):
        return

    with requests.Session() as session:
        session.auth = (username, password)

        filename = filename_v1
        filepath = path_v1
        response = retrieve_file(session, year, month, day, filename_v1)

        if not response.ok:
            print(f'Error downloading with file_version="001". Retrying with file_version="002"')
            filename = filename_v2
            filepath = path_v2
            response = retrieve_file(session, year, month, day, filename_v2)

        assert response.ok, 'Problem downloading data! Reason: {}'.format(response.reason)
    
        with open(filepath, 'wb') as f:
            f.write(response.content)

        print(filename + ' downloaded')
        print('Downloading SMAP data for: ' + str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2))

def get_filename(year, month, day, file_version):
    return 'SMAP_L3_SM_P_{}{}{}_R19240_{}.h5'.format(year, month, day, file_version)

def retrieve_file(session, year, month, day, filename):
    host = 'https://n5eil01u.ecs.nsidc.org'
    version = '009'  # product version
    url_path = '{}/SMAP/SPL3SMP.{}/{}.{}.{}/'.format(host, version, year, month, day)
    
    smap_data_path = url_path + filename

    print("PATHL ", smap_data_path)

    response = session.get(smap_data_path)

    if response.status_code == 401:
        # why have you done this to us, HTTP
        response = session.get(response.url)

    return response


def list_all_bands_in_h5_file(file_path):
    def list_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            if 'Soil_Moisture_Retrieval_Data_PM/soil_' in name:
                print(name)

    with h5py.File(file_path, 'r') as file:
        file.visititems(list_datasets)


def create_geotiff(data, output_file):
    src_crs = CRS.from_epsg(6933) # EASE-Grid 2.0 Global
    dst_crs = CRS.from_epsg(3857) # WGS 84 / Pseudo-Mercator

    # https://nsidc.org/data/user-resources/help-center/guide-ease-grids#anchor-36-km-resolution-ease-grids
    x_topleft = -17367530.44
    y_topleft = 7314540.83
    grid_units = 36032.22

    x_bottomright = x_topleft + grid_units * data.shape[1]
    y_bottomright = y_topleft - grid_units * data.shape[0]

    src_transform = affine.Affine.from_gdal(x_topleft, grid_units, 0, y_topleft, 0, -grid_units)

    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, data.shape[1], data.shape[0],
        left=x_topleft, bottom=y_bottomright, right=x_bottomright, top=y_topleft
    )

    nodata = -9999.0

    with rasterio.open(
        output_file, 'w', driver='GTiff',
        width=width, height=height, count=1, crs=dst_crs, 
        transform=transform, dtype=data.dtype, nodata=nodata
    ) as dataset:
        reproject(
            source=data, destination=rasterio.band(dataset, 1),
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=RESAMPLING,
            src_nodata=nodata
        )


def load_file_h5():
    file_path = ROOT_PATH + "/raw/"
    am_dataset_name = "Soil_Moisture_Retrieval_Data_AM/soil_moisture"
    pm_dataset_name = "Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm"

    for f in os.listdir(file_path):
        if not f.endswith(".h5"): continue
        with h5py.File(file_path + f, 'r') as file:
            if am_dataset_name in file and pm_dataset_name in file:
                am_data = file[am_dataset_name][()]
                pm_data = file[pm_dataset_name][()]
                merged_data = am_data.copy()
                merged_data[am_data == -9999.0] = pm_data[am_data == -9999.0]

                output_file = f"{file_path}{f.split('_')[4]}.tif"
                create_geotiff(merged_data, output_file)

            elif am_dataset_name in file:
                am_data = file[am_dataset_name][()]
                output_file = f"{file_path}{f.split('_')[4]}.tif"
                create_geotiff(am_data, output_file)

            elif pm_dataset_name in file:
                pm_data = file[pm_dataset_name][()]
                output_file = f"{file_path}{f.split('_')[4]}.tif"
                create_geotiff(pm_data, output_file)

            else:
                print('No soil moisture band found for: ', f)
        #os.remove(file_path + f)


def chop_in_quadhash():
    #quadhash_dir = next(d for d in os.listdir() if os.path.isdir(d) and d.startswith(f"quadshape_{ZOOM}_"))
    #quadhashes = gpd.read_file(os.path.join(quadhash_dir, 'quadhash.shp'))
    #tiles = mercantile.tiles(-180, -85, 180, 85, [3], True)
    tiles = mercantile.tiles(-125, 24, -66, 50, [ZOOM], True)

    in_path = ROOT_PATH + "/raw/"
    out_path = ROOT_PATH + f"/split_{ZOOM}/"
    os.makedirs(out_path, exist_ok=True)

    count = 0
    for tile in tiles:
        count += 1
        quadkey = mercantile.quadkey(tile)
        os.makedirs(out_path + quadkey, exist_ok=True)
        bounds = mercantile.bounds(quadkey)

        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])
        for f in os.listdir(in_path):
            gdal.Translate(out_path + qua + '/' + f, in_path + f, projWin=window)

            x = gdal.Open(out_path + qua + '/' + f).ReadAsArray()
            if np.min(x) == np.max(x) == -9999.0:
                os.remove(out_path + qua + '/' + f)

    remove_empty_folders()


def remove_empty_folders():
    in_path = ROOT_PATH + "/split_3/"
    tot = len(os.listdir(in_path))
    count = 0
    for q in os.listdir(in_path):
        if len(os.listdir(in_path + q)) == 0:
            print("No files in :", q)
            count += 1
            os.rmdir(in_path + q)
    print(count, "/", tot)

if __name__ == '__main__':
    # Provide either year,mm,dd or number of days to look back
    # Remove for loop below if want one day prediction
    for i in range(2,3): #(2,50)
        download_smap_automatically(year=None, month='04', day='04', n_days_before=i)
    load_file_h5()

    #chop_in_quadhash()
