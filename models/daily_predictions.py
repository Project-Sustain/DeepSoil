import os
import shutil
import socket
import subprocess
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from daily_data_loader import QuadhashDatasetAllQuads, Dataloader
import torch
import torch.nn as nn
import gdal
from osgeo import osr
import datetime
np.set_printoptions(suppress=True)
import cv2
import math

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from dataset_preprocessing import autodownload_gridmet, autodownload_smap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_PATH = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/"


def load_model_weights(folder):
    out_path = OUT_PATH + "outputs/" + str(folder) + "/"
    model_path = out_path + "model_weights.pth"
    loaded_model = torch.jit.load(model_path)
    return loaded_model


def generate_shape_dic():
    inp_path = OUT_PATH + "input_datasets/hru/split_14/"
    shap_map = {}
    for q in os.listdir(inp_path):
        dates = os.listdir(inp_path + q)
        sm_hru = gdal.Open(os.path.join(inp_path, q, dates[0]))
        geotransform = sm_hru.GetGeoTransform()
        if not os.path.exists(OUT_PATH + "daily_predictions/projection_info"):
            projection = sm_hru.GetProjection()
            with open(OUT_PATH + "daily_predictions/projection_info", 'w') as f:
                f.write(projection)
        shap_map[q] = {
            'RasterXSize': sm_hru.RasterXSize,
            'RasterYSize': sm_hru.RasterYSize,
            'Geotransform': geotransform
        }
        sm_hru = None

    shape_map_file = os.path.join(OUT_PATH, "daily_predictions", 'shape_map.json')
    print("saving at: ", shape_map_file)
    with open(shape_map_file, 'w') as f:
        json.dump(shap_map, f, indent=4)

    return


def load_shape_map_dict():
    shape_map_file = os.path.join(OUT_PATH, "daily_predictions", 'shape_map.json')
    with open(shape_map_file, 'r') as f:
        shape_map_dict = json.load(f)

    return shape_map_dict

def load_projection():
    with open(OUT_PATH + "daily_predictions/projection_info", 'r') as f:
        projection = f.read()

    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)

    print("Projection loaded successfully:")
    return projection

def perform_inferences_tiffs(folder, shape_map_dict, projection, dates):
    model2 = load_model_weights(folder)
    model2 = model2.to(device).float()
    model2 = model2.eval()

    batch_size = 256
    test_dataset = QuadhashDatasetAllQuads(dates=dates, folder=folder)
    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    os.makedirs(OUT_PATH + "daily_predictions/outputs", exist_ok=True)

    with torch.no_grad():
        for (input_sample, allquads, dates) in dataloader_new:
            input_sample = input_sample.to(torch.float32).to(device)
            output_sample = model2(input_sample).cpu().numpy()

            for i in range(output_sample.shape[0]):
                d = dates[i]

                output_geotiff_path = OUT_PATH + "daily_predictions/outputs/model_output_" + d[:4] + "_" + d[
                                                                                                           4:6] + "_" + d[
                                                                                                                        6:8]
                os.makedirs(output_geotiff_path, exist_ok=True)
                q = allquads[i]

                info = shape_map_dict.get(q)

                resized_out_image = cv2.resize(output_sample[i][0], (info['RasterXSize'], info['RasterYSize']))
                driver = gdal.GetDriverByName('GTiff')
                b_dataset = driver.Create(output_geotiff_path + "/" + q + ".tif", info['RasterXSize'],
                                          info['RasterYSize'], 1,
                                          gdal.GDT_Float32)
                b_dataset.SetProjection(projection)
                b_dataset.SetGeoTransform(info['Geotransform'])
                band = b_dataset.GetRasterBand(1)
                band.WriteArray(resized_out_image)
                b_dataset = None

            else:
                continue


def download_gridmet_dataset(year=None, month=None, day=None, n_days_before=3):
    doy_needed = autodownload_gridmet.download_nc_file(year=year, month=month, day=day, n_days_before=n_days_before)
    autodownload_gridmet.convert_to_tif(doy_needed)
    autodownload_gridmet.chop_in_quadhash()


def download_smap_dataset(year=None, month=None, day=None, n_days_before=2):
    autodownload_smap.download_smap_automatically(year=year, month=month, day=day, n_days_before=n_days_before)
    autodownload_smap.chop_in_quadhash()

def get_date_string(year, month, day, n_days_before):
    if year is None or month is None or day is None:
        current_date = datetime.date.today()
        delta = datetime.timedelta(days=n_days_before)
        date_string = (current_date - delta).strftime('%Y%m%d')
    else:
        date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)
    return date_string

if __name__ == '__main__':
    year, month, day, n_days_before = 2024, None, 1, 2
    dates = get_date_string(year, month, day, n_days_before)
    download_gridmet_dataset(year=year, month=month, day=day, n_days_before=n_days_before)
    download_smap_dataset(year=year, month=month, day=day, n_days_before=n_days_before)

    if not os.path.exists(os.path.join(OUT_PATH, "daily_predictions", 'shape_map.json')):
        generate_shape_dic()
    projection = load_projection()
    shape_map_dict = load_shape_map_dict()

    meta_data_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/outputs/meta_files/"
    number_of_quads_per_model = len(os.listdir(meta_data_path))

    inp_p = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/nlcd/split_14/"
    total = math.ceil(len(os.listdir(inp_p)) / number_of_quads_per_model)

    for folder in range(1, total + 1):
        perform_inferences_tiffs(folder=folder, shape_map_dict=shape_map_dict, projection=projection, dates=dates)
        print("Finished generating tiffs for folder:", folder)
