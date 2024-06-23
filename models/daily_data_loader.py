import socket
import subprocess
import gdal
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import random
from skimage import exposure
from skimage.transform import resize
np.set_printoptions(suppress=True)
import datetime

class Dataloader:
    def __init__(self, height=64, width=64):
        self.train = []
        self.test = []
        self.height = height
        self.width = width

        root_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/"

        self.polaris_path = root_path + "sm_predictions/input_datasets/polaris/split_14/"
        self.nlcd_path = root_path + "sm_predictions/input_datasets/nlcd/split_14/"
        self.gNATSGO_path = root_path + "sm_predictions/input_datasets/gnatsgo/split_14/"
        self.dem_path = root_path + "sm_predictions/input_datasets/dem/split_14/"
        self.koppen_path = root_path + "sm_predictions/input_datasets/koppen/split_14/"

        self.smap_path = root_path + "sm_predictions/daily_predictions/input_datasets/smap/split_14/"
        self.gridmet_path = root_path + "sm_predictions/daily_predictions/input_datasets/gridmet/split_14/"
        self.lai_path = root_path + "sm_predictions/input_datasets/MCD15A3H/split_14/"
        self.landsat8_path = root_path + "sm_predictions/input_datasets/landsat8L2/split_14/"
        self.meta_path = root_path + "sm_predictions/outputs/meta_files/"

    def scale_image_polaris(self, data_array):
        min_polaris = [0, 0, 0, 0.5, 0, 0, 0, 0, -1, -2, -1, -2, -2]
        max_polaris = [100, 100, 75, 2, 0.81, 0.25, 2.5, 10.2, 2, 1.7, 0.6, 1.2, 0.4]

        scaled_image = np.zeros_like(data_array, dtype=float)

        for band in range(data_array.shape[-1]):
            min_val = min_polaris[band]
            max_val = max_polaris[band]
            if band <= 8:
                scaled_band = (data_array[:, :, band] - min_val) / (max_val - min_val)
                scaled_band[scaled_band < 0] = -1
            else:
                scaled_band = data_array[:, :, band]
                scaled_band[scaled_band < min_val] = -1

            scaled_image[:, :, band] = scaled_band

        scaled_image[:, :, -1] = 10 ** scaled_image[:, :, -1]
        return scaled_image

    def scale_image_gnatsgo_30m(self, data_array):
        data_array[data_array == -9999] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)

        masked_data_array[:, :, 0] = masked_data_array[:, :, 0] / 270.0  # aws_0-5
        masked_data_array[:, :, 1] = masked_data_array[:, :, 1] / 50.0  # tk_0-5
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def scale_image_gnatsgo_90m(self, data_array):
        data_array[data_array == -9999] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)

        masked_data_array[:, :, 0] = masked_data_array[:, :, 0] / 1000.0  #porosity
        min_val, max_val = 0, 300  # awc
        masked_data_array[:, :, 1] = (masked_data_array[:, :, 1] - min_val) / (max_val - min_val)
        min_val, max_val = 0, 500  # fc
        masked_data_array[:, :, 2] = (masked_data_array[:, :, 2] - min_val) / (max_val - min_val)
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def scale_nlcd(self, data_array):
        x_min = 11.0
        x_max = 95.0
        data_array = data_array.astype(np.float64)

        data_array[data_array == 0.0] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)
        masked_data_array = (masked_data_array - x_min) / (x_max - x_min)
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def get_max_occuring_val_in_array(self, arr):
        values, counts = np.unique(arr, return_counts=True)
        max_count_index = np.argmax(counts)
        most_frequent = values[max_count_index]
        return np.array([most_frequent])

    def scale_koppen(self, data_array):
        x_min = 1
        x_max = 30.0
        data_array = data_array.astype(np.float64)

        data_array[data_array == 0.0] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)
        masked_data_array = (masked_data_array - x_min) / (x_max - x_min)
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def scale_dem(self, data_array):
        x_max = 4500
        x_min = -122.34724
        data_array = (data_array - x_min) / (x_max - x_min)
        return data_array

    def scale_gridmet(self, data):
        data[data == 32767.0] = np.nan
        mins = [0, 0, 0, 1.05, 0, 0, 225.54, 233.08, 0]
        maxs = [27.02, 17.27, 690.44, 100, 100, 455.61, 314.88, 327.14, 9.83]
        scaled_image = np.zeros_like(data, dtype=float)

        for band in range(data.shape[-1]):
            min_val = mins[band]
            max_val = maxs[band]
            scaled_band = (data[:, :, band] - min_val) / (max_val - min_val)
            scaled_band[scaled_band < 0] = -1
            scaled_band[scaled_band > 1] = -1
            scaled_image[:, :, band] = scaled_band

        scaled_image = np.nan_to_num(np.array(scaled_image), nan=-1)
        return scaled_image

    def scale_image_smap(self, data_array):
        data_array[data_array == -9999] = np.nan
        data_array[data_array < 0] = 0
        data_array[data_array > 1] = 1
        data_array = np.nan_to_num(data_array, nan=-1)
        return data_array

    def scale_image_lai(self, data_array):
        data_array = data_array.astype(np.float64)
        data_array[data_array < 0] = np.nan
        data_array[data_array > 100] = np.nan
        data_array = data_array * 0.1
        data_array = data_array / 10
        data_array = np.nan_to_num(data_array, nan=-1)
        return data_array

    def resize_image(self, data, height=64, width=64):
        '''
        Resize input multidimensional image
        :param data: input dataset of shape - (height, width, num_of_channels)
        :param height: target height
        :param width: target width
        :param isNLCD: is data NLCD landcover
        :return: reshaped/resized dataset
        '''

        resized_array = resize(data, (height, width, data.shape[-1]), mode='constant', preserve_range=True, anti_aliasing=True)
        return resized_array

    def scale_image_landsat_30m(self, img):
        def normalize(band):
            return ((band - 1) / (60000.0 - 1))

        def scale_image_landsat_30m(imgs):
            imgs = imgs.astype(np.float32)
            imgs[imgs == 0] = np.nan
            imgs = normalize(imgs)
            imgs[imgs < 0] = 0
            imgs = np.nan_to_num(imgs, nan=-1)
            return imgs

        img = exposure.rescale_intensity(img,
                                             out_range=(1, 50000)).astype(np.int32)

        img = img.astype(np.float32)
        img[img == 0] = np.nan
        img = scale_image_landsat_30m(img)
        return img

    def load_static_datasets(self, quad):
        if quad is None or len(quad) != 14:
            return None

        if os.path.exists(self.polaris_path + quad + "/0_5_merged.tif"):
            image_polaris = gdal.Open(self.polaris_path + quad + "/0_5_merged.tif").ReadAsArray()
            image_polaris = self.scale_image_polaris(self.resize_image(np.transpose(image_polaris, (1, 2, 0))))
        else:
            return None

        if os.path.exists(self.nlcd_path + quad + "/nlcd.tif"):
            image_nlcd = gdal.Open(self.nlcd_path + quad + "/nlcd.tif").ReadAsArray()
            image_nlcd = image_nlcd.reshape(1, image_nlcd.shape[0], image_nlcd.shape[1])
            image_nlcd = self.scale_nlcd(self.resize_image(np.transpose(image_nlcd, (1, 2, 0))))
        else:
            return None

        if os.path.exists(self.gNATSGO_path + quad + "/30m.tif"):
            image_gnatsgo_30m = gdal.Open(self.gNATSGO_path + quad + "/30m.tif").ReadAsArray()
            image_gnatsgo_30m = self.scale_image_gnatsgo_30m(
                self.resize_image(np.transpose(image_gnatsgo_30m, (1, 2, 0))))
        else:
            return None

        if os.path.exists(self.gNATSGO_path + quad + "/90m.tif"):
            image_gnatsgo_90m = gdal.Open(self.gNATSGO_path + quad + "/90m.tif").ReadAsArray()
            image_gnatsgo_90m = self.scale_image_gnatsgo_90m(
                self.resize_image(np.transpose(image_gnatsgo_90m, (1, 2, 0))))
        else:
            return None

        if os.path.exists(self.dem_path + quad + "/final_elevation_30m.tif"):
            image_dem = gdal.Open(self.dem_path + quad + "/final_elevation_30m.tif").ReadAsArray()
            image_dem = image_dem.reshape(1, image_dem.shape[0], image_dem.shape[1])
            image_dem = self.scale_dem(self.resize_image(np.transpose(image_dem, (1, 2, 0))))
        else:
            return None

        if os.path.exists(self.koppen_path + quad + "/1km.tif"):
            image_koppen = gdal.Open(self.koppen_path + quad + "/1km.tif").ReadAsArray()
            image_koppen = image_koppen.reshape(1, image_koppen.shape[0], image_koppen.shape[1])
            image_koppen = self.scale_koppen(self.get_max_occuring_val_in_array(np.transpose(image_koppen, (1, 2, 0))))
            image_koppen = np.full((64, 64, 1), image_koppen[0], dtype=np.uint8)
        else:
            return None

        if os.path.exists(self.landsat8_path + quad + "/landsat_final_30m.tif"):
            image_landsat_30m = gdal.Open(self.landsat8_path + quad + "/landsat_final_30m.tif").ReadAsArray()
            image_landsat_30m = np.reshape(self.scale_image_landsat_30m(self.resize_image(np.transpose(image_landsat_30m, (1, 2, 0)))), (64,64,3))
        else:
            return None

        merged_image = np.concatenate(
            (image_gnatsgo_30m, image_gnatsgo_90m, image_koppen, image_dem, image_nlcd, image_polaris, image_landsat_30m), axis=2)
        return merged_image

    def load_daily_datasets(self, quad, date):
        if quad is None or len(quad) != 14:
            return None

        if os.path.exists(self.smap_path + quad[:12] + "/" + date):
            img_smap = gdal.Open(self.smap_path + quad[:12] + "/" + date).ReadAsArray()
            img_smap = img_smap.reshape(1, img_smap.shape[0], img_smap.shape[1])
            img_smap = self.scale_image_smap(self.resize_image(np.transpose(img_smap, (1, 2, 0))))
        else:
            return None

        if os.path.exists(self.gridmet_path + quad[:12] + "/" + date):
            img_grid = gdal.Open(self.gridmet_path + quad[:12] + "/" + date).ReadAsArray()
            img_grid = self.scale_gridmet(self.resize_image(np.transpose(img_grid, (1, 2, 0))))
        else:
            return None

        if os.path.exists(self.lai_path + quad + "/" + date):
            img_lai = gdal.Open(self.lai_path + quad + "/" + date).ReadAsArray()
            img_lai = img_lai.reshape(1, img_lai.shape[0], img_lai.shape[1])
            img_lai = self.scale_image_lai(self.resize_image(np.transpose(img_lai, (1, 2, 0))))
        else:
            img_lai = np.random.rand(64, 64, 1)
            # return None
        merged_image = np.concatenate((img_smap, img_grid, img_lai), axis=2)
        return merged_image

    def generate_train_test_data_all_quads(self, data_quad_list, dates):
        all_imgs_30m_daily, all_imgs_30m_static, all_target_sm, all_quads, all_dates = [], [], [], [], []
        dates_hru = [dates + ".tif"]

        for quad in data_quad_list:

            img_30m_static = self.load_static_datasets(quad)
            if img_30m_static is None:
                continue

            for i in range(len(dates_hru)):
                img_30m_daily = self.load_daily_datasets(quad, dates_hru[i])
                if img_30m_daily is None:
                    continue
                all_imgs_30m_static.append(img_30m_static)
                all_imgs_30m_daily.append(img_30m_daily)
                all_quads.append(quad)
                all_dates.append(dates_hru[i])

        all_imgs_30m_daily = np.array(all_imgs_30m_daily)
        all_imgs_30m_static = np.array(all_imgs_30m_static)
        all_quads = np.array(all_quads)
        all_dates = np.array(all_dates)
        return all_imgs_30m_daily, all_imgs_30m_static, all_quads, all_dates

    def read_folder_quads(self, folder):
        path = self.meta_path + "/quads_" + str(folder) + ".txt"
        quad_list = []
        print("Loading from meta data")
        with open(path, 'r') as file:
            for quad in file:
                quad = quad.strip()
                quad_list.append(quad)
        return quad_list

class QuadhashDatasetAllQuads(Dataset):
    def __init__(self, dates, folder=None):
        self.data_loader = Dataloader()

        if folder is None:
            print("No trained model meta data found for folder: ", folder)
            return
        else:
            quadhashes = self.data_loader.read_folder_quads(folder)

        total_quad = len(quadhashes)
        print("Loading input dataset for quadhashes trained using folder {} number of quads : {}".format(folder, total_quad))

        self.all_input_daily, self.all_input_static, self.all_quads, self.dates = self.data_loader.generate_train_test_data_all_quads(quadhashes, dates)
        print("Samples found: ", len(self.all_quads))

    def __len__(self):
        return len(self.all_quads)

    def __getitem__(self, index):

        all_input_1 = self.all_input_daily[index]  # daily
        all_input_2 = self.all_input_static[index]  # static
        all_quads = self.all_quads[index]
        all_months = self.dates[index]

        merged_image = np.concatenate((all_input_1, all_input_2), axis=2)

        merged_image = torch.tensor(merged_image).permute(2, 0, 1)
        return merged_image, all_quads, all_months


if __name__ == '__main__':
    batch_size = 5
    my_loader = Dataloader()
    train_dataset = QuadhashDatasetAllQuads('20240611', folder=1)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    for all_input_1, all_quads, all_months in train_dataloader:
        print(all_input_1.shape, len(all_quads), len(all_months))
        break

