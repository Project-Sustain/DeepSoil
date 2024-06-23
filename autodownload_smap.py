import socket
import subprocess
import gdal
import numpy as np
import os
import geopandas as gpd
import h5py
from datetime import datetime, timedelta
import requests

username = "YOUR-NAME"
password = "YOUR-PASSWORD"

# Output will get stored in ROOT_PATH/split_14
ROOT_PATH = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/daily_predictions/input_datasets/smap"
os.makedirs(ROOT_PATH, exist_ok=True)

os.chdir(os.path.dirname(__file__))

def download_smap_with_version(year, month, day, url_path, file_version='001'):
    filename = 'SMAP_L3_SM_P_E_{}{}{}_R19240_{}.h5'.format(year, month, day, file_version)
    smap_data_path = url_path + filename
    out_dir = ROOT_PATH + "/raw/"
    os.makedirs(out_dir, exist_ok=True)
    with requests.Session() as session:
        session.auth = (username, password)
        filepath = os.path.join(out_dir, filename)
        response = session.get(smap_data_path)

        if not response.ok:
            print('Error downloading with file_version={}'.format(file_version))
            response = session.get(response.url)
            print("Possible reason: ", response.reason)
            return None

        with open(filepath, 'wb') as f:
            f.write(response.content)

        print('Successfully downloaded', filename, " with version ", file_version)
        return 1

'''
    Downloads SMAP Soil moisture from 2 days back in .h5 format
    The download path changes every year (majorly the product version of the product)
'''
def download_smap_automatically(year=None, month=None, day=None, n_days_before=2):
    if year is None or month is None or day is None:
        current_date = datetime.now()
        year, month, day = (current_date - timedelta(days=n_days_before)).strftime("%Y-%m-%d").split("-")
        print("\nLooking for: {}/{}/{}".format(year, month, day))

    host = 'https://n5eil01u.ecs.nsidc.org/'
    version = '.006'                            # product version
    url_path = '{}/SMAP/SPL3SMP_E{}/{}.{}.{}/'.format(host, version, year, month, day)

    out = download_smap_with_version(year, month, day, url_path, file_version='001')
    count_v = 2
    while out is None and count_v <= 9:
        print("Trying new new version: ", count_v)
        out = download_smap_with_version(year, month, day, url_path, file_version=str(count_v).zfill(3))
        count_v += 1
    if out is None:
        print("No data found with any versions")
    else:
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


def chop_in_quadhash():
    quadhash_dir = next(d for d in os.listdir() if os.path.isdir(d) and d.startswith("quadshape_12_"))
    quadhashes = gpd.read_file(os.path.join(quadhash_dir, 'quadhash.shp'))

    in_path = ROOT_PATH + "/raw/"
    out_path = ROOT_PATH + "/split_14/"
    os.makedirs(out_path, exist_ok=True)

    count = 0
    total = len(quadhashes)
    for ind, row in quadhashes.iterrows():
        count += 1
        poly, qua = row["geometry"], row["Quadkey"]
        os.makedirs(out_path + qua, exist_ok=True)
        bounds = list(poly.exterior.coords)
        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

        for f in os.listdir(in_path):
            fnew = f.split("_")[5] + ".tif"
            gdal.Translate(out_path + qua + '/' + fnew, in_path + f, projWin=window)

            x = gdal.Open(out_path + qua + '/' + fnew).ReadAsArray()
            if np.min(x) == np.max(x) == -9999.0:
                os.remove(out_path + qua + '/' + fnew)

    remove_empty_folders()
    for f in os.listdir(in_path):
        os.remove(in_path + f)

def remove_empty_folders():
    in_path = ROOT_PATH + "/split_14/"
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
    download_smap_automatically(year='2024', month=None, day='04', n_days_before=2)
    chop_in_quadhash()
