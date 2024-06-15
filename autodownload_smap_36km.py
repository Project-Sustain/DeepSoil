import socket
import subprocess
import gdal
import numpy as np
import mercantile, fiona
import rasterio as rio
from rasterio import mask as msk
import random
import geopy.distance
import os, osr
import geopandas as gpd
import shutil
import h5py
import pickle
import json
import calendar
from datetime import datetime, timedelta
import requests

username = "YOUR-USERNAME"
password = "YOUR-PASSWORD"

# Output stored under ROOT_PATH/raw/
ROOT_PATH = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/daily_predictions/input_datasets/smap_36"
os.makedirs(ROOT_PATH, exist_ok=True)

'''
    Downloads SMAP Soil moisture from 2 days back in .h5 format
    The download path changes every year (majorly the product version of the product)
'''
def download_smap_automatically(year=None, month=None, day=None, n_days_before=2):
    if year is None or month is None or day is None:
        current_date = datetime.now()
        year, month, day = (current_date - timedelta(days=n_days_before)).strftime("%Y-%m-%d").split("-")

    host = 'https://n5eil01u.ecs.nsidc.org'
    version = '009'  # product version
    url_path = '{}/SMAP/SPL3SMP.{}/{}.{}.{}/'.format(host, version, year, month, day)

    file_version = '001'
    filename = 'SMAP_L3_SM_P_{}{}{}_R19240_{}.h5'.format(year, month, day, file_version)
    smap_data_path = url_path + filename
    out_dir = ROOT_PATH + "/raw/"
    os.makedirs(out_dir, exist_ok=True)

    with requests.Session() as session:
        session.auth = (username, password)
        filepath = os.path.join(out_dir, filename)
        response = session.get(smap_data_path)

        if not response.ok:
            print('Error downloading with file_version="001". Retrying with file_version="002"')

            file_version = '002'
            filename = 'SMAP_L3_SM_P_{}{}{}_R19240_{}.h5'.format(year, month, day, file_version)
            smap_data_path = url_path + filename
            filepath = os.path.join(out_dir, filename)
            response = session.get(smap_data_path)

        if response.status_code == 401:
            response = session.get(response.url)
        assert response.ok, 'Problem downloading data! Reason: {}'.format(response.reason)

        with open(filepath, 'wb') as f:
            f.write(response.content)

        print(filename + ' downloaded')
        print('Downloading SMAP data for: ' + str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2))
    load_file_h5()

def list_all_bands_in_h5_file(file_path):
    def list_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            if 'Soil_Moisture_Retrieval_Data_PM/soil_' in name:
                print(name)

    with h5py.File(file_path, 'r') as file:
        file.visititems(list_datasets)


def create_geotiff(data, output_file):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_file, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(data)
    geotransform = (-180.00, 0.0174532925199433, 0, 85.0445, 0, -0.0174532925199433)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection("EPSG:4326")
    dataset = None


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
        os.remove(file_path + f)

if __name__ == '__main__':
    # Provide either year,mm,dd or number of days to look back
    download_smap_automatically(year='2024', month='04', day='04', n_days_before=2)
