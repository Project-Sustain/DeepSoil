import socket
import subprocess
import gdal
import numpy as np
import mercantile,fiona
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

username = "paahukh22"
password = "Paahuni@1234"

def download_smap_automatically():
    # Downloads SMAP Soil moisture from yesterday in .h5 format
    current_date = datetime.now()
    year, month, day = (current_date - timedelta(days=1)).strftime("%Y-%m-%d").split("-")

    host = 'https://n5eil01u.ecs.nsidc.org/'
    version = '.005'
    url_path = '{}/SMAP/SPL3SMP_E{}/{}.{}.{}/'.format(host,version,year,month,day)
    filename = 'SMAP_L3_SM_P_E_{}{}{}_R18290_001.h5'.format(year,month,day)

    smap_data_path = url_path + filename

    DATA_DIR = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/smap/raw/"

    with requests.Session() as session:
        session.auth = (username, password)
        filepath = os.path.join(DATA_DIR, filename)
        response = session.get(smap_data_path)
        if response.status_code == 401:
            response = session.get(response.url)
        assert response.ok, 'Problem downloading data! Reason: {}'.format(response.reason)

        with open(filepath, 'wb') as f:
            f.write(response.content)

        print(filename + ' downloaded')
        print('Downloading SMAP data for: ' + str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2))

def list_all_bands_in_h5_file(file_path):
    def list_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            if 'Soil_Moisture_Retrieval_Data_PM/soil_moi' in name:
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
    file_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/smap/raw/"
    am_dataset_name = "Soil_Moisture_Retrieval_Data_AM/soil_moisture"
    pm_dataset_name = "Soil_Moisture_Retrieval_Data_PM/soil_moisture_dca_pm"

    for f in os.listdir(file_path):
        if not f.endswith(".h5"): continue
        with h5py.File(file_path + f, 'r') as file:
            if am_dataset_name in file and pm_dataset_name in file:
                am_data = file[am_dataset_name][()]
                pm_data = file[pm_dataset_name][()]
                merged_data = am_data.copy()
                merged_data[am_data == -9999.0] = pm_data[am_data == -9999.0]

                output_file = f"{file_path}{f.split('.h5')[0]}.tif"
                create_geotiff(merged_data, output_file)

            elif am_dataset_name in file:
                am_data = file[am_dataset_name][()]
                output_file = f"{file_path}{f.split('.h5')[0]}.tif"
                create_geotiff(am_data, output_file)

            elif pm_dataset_name in file:
                pm_data = file[pm_dataset_name][()]
                output_file = f"{file_path}{f.split('.h5')[0]}.tif"
                create_geotiff(pm_data, output_file)

            else:
                print('No soil moisture band found for: ', f)
        os.remove(file_path + f)

def transfer_tifs_to_all_machines():
    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/smap/raw/"
    for send_to_lattice in range(176, 224):
        if send_to_lattice == int(socket.gethostname().split("-")[1]):
            continue
        out_path = "/s/lattice-" + str(send_to_lattice) + "/b/nobackup/galileo/sm_predictions/input_datasets/smap/raw/"

        files_to_transfer = os.listdir(in_path)

        for file_name in files_to_transfer:
            source_file = os.path.join(in_path, file_name)
            command = ['scp', source_file, 'paahuni@lattice-' + str(send_to_lattice) + ":" + out_path]
            subprocess.run(command)

        print("Sent to lattice-", send_to_lattice)

def chop_in_quadhash():
    quadhash_dir = next(d for d in os.listdir() if os.path.isdir(d) and d.startswith("quadshape_12_"))
    quadhashes = gpd.read_file(os.path.join(quadhash_dir, 'quadhash.shp'))

    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/smap/raw/"
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/smap/split_14/"

    count = 0
    total = len(quadhashes)
    for ind, row in quadhashes.iterrows():
        count += 1
        print("processing:  ", count, "/", total)
        poly, qua = row["geometry"], row["Quadkey"]
        os.makedirs(out_path + qua, exist_ok=True)
        bounds = list(poly.exterior.coords)
        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

        for f in os.listdir(in_path):
            gdal.Translate(out_path + qua + '/' + f,  in_path + f, projWin=window)

            x = gdal.Open(out_path + qua + '/' + f).ReadAsArray()
            if np.min(x) == np.max(x) == -9999.0:
                os.remove(out_path + qua + '/' + f)

    remove_empty_folders()

def remove_empty_folders():
    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/smap/split_14/"
    tot = len(os.listdir(in_path))
    count = 0
    for q in os.listdir(in_path):
        if len(os.listdir(in_path + q)) == 0:
            print("No files in :", q)
            count += 1
            os.rmdir(in_path + q)
    print(count, "/", tot)

if __name__ == '__main__':
    download_smap_automatically()
    load_file_h5()
    chop_in_quadhash()

