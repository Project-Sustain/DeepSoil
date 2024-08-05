import gdal
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import random
from skimage import exposure
from skimage.transform import resize
np.set_printoptions(suppress=True)
import csv
from datetime import datetime, timedelta

class Dataloader:
    def __init__(self, height=64, width=64):
        self.train = []
        self.test = []
        self.height = height
        self.width = width

        root_path = "/s/chopin/f/proj/fineET/"
        # root_path = "/s/lovelace/f/nobackup/shrideep/sustain/"

        self.polaris_path = root_path + "sm_predictions/input_datasets/polaris/split_14/"
        self.nlcd_path = root_path + "sm_predictions/input_datasets/nlcd/split_14/"
        self.gNATSGO_path = root_path + "sm_predictions/input_datasets/gnatsgo/split_14/"
        self.dem_path = root_path + "sm_predictions/input_datasets/dem/split_14/"
        self.koppen_path = root_path + "sm_predictions/input_datasets/koppen/split_14/"
        self.station_path = root_path + "sm_predictions/input_datasets/station_data/split/5/"
        self.smap_path = root_path + "sm_predictions/input_datasets/smap/split_14/"
        self.gridmet_path = root_path + "sm_predictions/input_datasets/gridmet/split_14/"
        self.lai_path = root_path + "sm_predictions/input_datasets/MCD15A3H/split_14/"
        self.hru_sm_path = root_path + "sm_predictions/input_datasets/hru_station_only/split_14/"
        self.landsat8_path = root_path + "sm_predictions/input_datasets/landsat8L2/split_14/"

    def scale_image_polaris(self, data_array):
        # ['silt', 'sand', 'clay', 'bd', 'theta_s', 'theta_r', 'ksat', 'ph', 'om', 'n',  'lambda', 'hb', 'alpha']
        # [0,        1,      2,    3,        4,         5,       6,     7,     8,   9,      10,     11,     12]
        min_polaris = [0,     0,  0, 0.5,    0,    0,   0,    0, -1,  -2,  -1,  -2,  -2]
        max_polaris = [100, 100, 75,   2, 0.81, 0.25, 2.5, 10.2,  2, 1.7, 0.6, 1.2, 0.4]

        scaled_image = np.zeros_like(data_array, dtype=float)

        for band in range(data_array.shape[-1]):
            min_val = min_polaris[band]
            max_val = max_polaris[band]
            if band <= 8 and band != 4 and band != 5 and band != 6:
                scaled_band = (data_array[:, :, band] - min_val) / (max_val - min_val)
                scaled_band[scaled_band < 0] = -1
            else:
                scaled_band = data_array[:, :, band]
                scaled_band[scaled_band < min_val] = -1

            scaled_image[:, :, band] = scaled_band

        scaled_image[:, :, 12] = 10 ** scaled_image[:, :, 12]
        scaled_image[:, :, 11] = 10 ** scaled_image[:, :, 11]
        scaled_image[:, :, 6] = 10 ** scaled_image[:, :, 6]
        scaled_image[:, :, 8] = 10 ** scaled_image[:, :, 8]
        return scaled_image 

    def scale_image_gnatsgo_30m(self, data_array):
        data_array[data_array == -9999] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)

        masked_data_array[:, :, 0] = masked_data_array[:, :, 0] / 270.0  # aws_0-5
        masked_data_array[:, :, 1] = masked_data_array[:, :, 1] / 50.0   # tk_0-5
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def scale_image_gnatsgo_90m(self, data_array):
        data_array[data_array == -9999] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)
         # porosity
        masked_data_array[:, :, 0] = masked_data_array[:, :, 0] / 1000.0
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
        mins = [0,         0,      0, 1.05,   0,      0, 225.54, 233.08,    0]
        maxs = [27.02, 17.27, 690.44,  100, 100, 455.61, 314.88, 327.14, 9.83]
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


    def scale_hru(self, data_array):
        data_array[data_array == -9999] = np.nan
        data_array[data_array < 0] = 0
        data_array[data_array > 1] = 1
        data_array = np.nan_to_num(data_array, nan=-1)
        return data_array

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

        resized_array = resize(data, (height, width, data.shape[-1]), order=0, preserve_range=True, anti_aliasing=False)
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
        # img[img == -1] = np.nan
        # ndvi = np.where((img[:, :, -1] + img[:, :, 0]) == 0., 0,
        #                     (img[:, :, -1] - img[:, :, 0]) / (img[:, :, -1] + img[:, :, 0]))

        return img

    def load_static_datasets(self, quad):
        if quad is None or len(quad) != 14:
            return None, None

        # ['silt', 'sand', 'clay', 'bd', 'theta_s', 'theta_r', 'ksat', 'ph', 'om', 'n',  'lambda', 'hb', 'alpha']
        if os.path.exists(self.polaris_path + quad + "/0_5_merged.tif"):
            image_polaris = gdal.Open(self.polaris_path + quad + "/0_5_merged.tif").ReadAsArray()
            image_polaris = self.scale_image_polaris(self.resize_image(np.transpose(image_polaris, (1, 2, 0))))
        else:
            return None, None

        if os.path.exists(self.gNATSGO_path + quad + "/30m.tif"):
            image_gnatsgo_30m = gdal.Open(self.gNATSGO_path + quad + "/30m.tif").ReadAsArray()
            image_gnatsgo_30m = self.scale_image_gnatsgo_30m(
                self.resize_image(np.transpose(image_gnatsgo_30m, (1, 2, 0))))
        else:
            return None, None

        if os.path.exists(self.gNATSGO_path + quad + "/90m.tif"):
            image_gnatsgo_90m = gdal.Open(self.gNATSGO_path + quad + "/90m.tif").ReadAsArray()
            image_gnatsgo_90m = self.scale_image_gnatsgo_90m(
                self.resize_image(np.transpose(image_gnatsgo_90m, (1, 2, 0))))
        else:
            return None, None

        if os.path.exists(self.koppen_path + quad + "/1km.tif"):
            image_koppen = gdal.Open(self.koppen_path + quad + "/1km.tif").ReadAsArray()
            image_koppen = image_koppen.reshape(1, image_koppen.shape[0], image_koppen.shape[1])
            image_koppen = self.scale_koppen(self.get_max_occuring_val_in_array(np.transpose(image_koppen, (1, 2, 0))))
            image_koppen = np.full((64, 64, 1), image_koppen[0], dtype=np.uint8)
        else:
            return None, None

        if os.path.exists(self.dem_path + quad + "/final_elevation_30m.tif"):
            image_dem = gdal.Open(self.dem_path + quad + "/final_elevation_30m.tif").ReadAsArray()
            image_dem = image_dem.reshape(1, image_dem.shape[0], image_dem.shape[1])
            image_dem = self.scale_dem(self.resize_image(np.transpose(image_dem, (1, 2, 0))))
        else:
            return None, None

        if os.path.exists(self.nlcd_path + quad + "/nlcd.tif"):
            image_nlcd = gdal.Open(self.nlcd_path + quad + "/nlcd.tif").ReadAsArray().astype(int)
            image_nlcd = image_nlcd.reshape(1, image_nlcd.shape[0], image_nlcd.shape[1]).astype(int)
            unscaled_nlcd = resize(image_nlcd, (1, 64, 64), order=0, preserve_range=True, anti_aliasing=False)
            unscaled_nlcd = np.transpose(unscaled_nlcd, (1, 2, 0)).astype(int)
            image_nlcd = self.scale_nlcd(unscaled_nlcd)
        else:
            return None, None

        if os.path.exists(self.landsat8_path + quad + "/landsat_final_30m.tif"):
            image_landsat_30m = gdal.Open(self.landsat8_path + quad + "/landsat_final_30m.tif").ReadAsArray()
            image_landsat_30m = np.reshape(self.scale_image_landsat_30m(self.resize_image(np.transpose(image_landsat_30m, (1, 2, 0)))), (64,64,3))
        else:
            return None, None

        merged_image = np.concatenate(
            (image_polaris, image_gnatsgo_30m, image_gnatsgo_90m, image_koppen, image_dem, image_nlcd, image_landsat_30m), axis=2)

        return merged_image, unscaled_nlcd

    def load_daily_datasets(self, quad, date):
        if quad is None or len(quad) != 14:
            return None

        if os.path.exists(self.smap_path + quad[:12] + "/" + date):
            img_smap = gdal.Open(self.smap_path + quad[:12] + "/" + date).ReadAsArray()
            img_smap = img_smap.reshape(1, img_smap.shape[0], img_smap.shape[1])
            img_smap = self.scale_image_smap(self.resize_image(np.transpose(img_smap, (1, 2, 0))))
        else:
            return None

        # [etr, pet, pr, rmax, rmin, srad, tmmn, tmmx, vpd]
        if os.path.exists(self.gridmet_path + quad[:12] + "/" + date):
            img_grid = gdal.Open(self.gridmet_path + quad[:12] + "/" + date).ReadAsArray()
            img_grid = self.scale_gridmet(self.resize_image(np.transpose(img_grid, (1, 2, 0))))[:, :,
                       [0, 2, 3, 7]]
        else:
            return None



        merged_image = np.concatenate((img_smap, img_grid), axis=2)
        return merged_image

    def check_season(self, date_str):
        # Return True for spring or summer
        out = self.get_season(date_str)
        if out == 1 or out == 2 or out == 3:
            return True
        else:
            return False

    def get_season(self, date_str):
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        if month >= 3 and month <= 9:
            return 1
        else:
            return 4

        # if (month == 3 and day >= 20) or (month > 3 and month < 6) or (month == 6 and day <= 20):
        #     return 1  # Spring
        # elif (month == 6 and day >= 21) or (month > 6 and month < 9) or (month == 8 and day <= 21):
        #     return 2  # "Summer"
        # elif (month == 8 and day >= 22) or (month > 9 and month < 10) or (month == 10 and day <= 20):
        #     return 3  # "Autumn/Fall"
        # else:
        #     return 4  # "Winter"

    def load_station_quads(self):
        input_path = "/s/chopin/f/proj/fineET/sm_predictions/input_datasets/station_data/split/5/"
        quads_stations = set(os.listdir(input_path))
        okhal_path = "/s/chopin/f/proj/fineET/sm_predictions/input_datasets/hru/split_14/"
        quads_okh = set(os.listdir(okhal_path))
        overlaps = quads_stations.intersection(quads_okh)

        return overlaps

    def check_if_station_present(self, quad, date):
        if os.path.exists(self.station_path + quad + "/" + date.replace(".tif", ".txt")):
            return True
        return False

    def load_station_data(self, input_file):
        unique_rows = []
        with open(input_file.replace(".tif", ".txt"), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                lat, lon, moisture = row[0], row[1], row[2]
                unique_rows.append([lat, lon, moisture])
        return unique_rows

    def load_true_data(self, filename, quad):
        date = filename.split(".")[0]
        x_true = []
        y_true = []
        sm_true = []
        rows = self.load_station_data(self.station_path + quad + "/" + filename)

        if len(rows) <= 0:
            print("No station data found for spatial quad: ", quad, "  date: ", date)

        for row in rows:
            x_true.append(float(row[1]))
            y_true.append(float(row[0]))
            sm_true.append(float(row[2]))

        return x_true, y_true, sm_true

    def generate_lats_lon_list(self, quad, date, sm_hru, fix_it = True):
        x_station, y_station, sm_true = self.load_true_data(date, quad)
        arr_sm_hru = sm_hru.ReadAsArray()
        for t_s in range(len(x_station)):
            x_pixel = int((x_station[t_s] - sm_hru.GetGeoTransform()[0]) / sm_hru.GetGeoTransform()[1])
            y_pixel = int((y_station[t_s] - sm_hru.GetGeoTransform()[3]) / sm_hru.GetGeoTransform()[5])
            if 0 <= sm_true[t_s] <= 1.0:
                try:
                    # print("Replacing: ", arr_sm_hru[y_pixel, x_pixel], "with:", sm_true[t_s])
                    arr_sm_hru[y_pixel, x_pixel] = sm_true[t_s]           # [row no., col no.]
                except IndexError:
                        if y_pixel == arr_sm_hru.shape[0]:
                                y_pixel -= 1
                        if x_pixel == arr_sm_hru.shape[1]:
                                x_pixel -= 1
                        if y_pixel > arr_sm_hru.shape[0]:
                                y_pixel = arr_sm_hru.shape[0] - 1
                        if x_pixel > arr_sm_hru.shape[1]:
                                x_pixel = arr_sm_hru.shape[1] - 1

                        # print("Replacing Adjusted pixel: ", arr_sm_hru[y_pixel, x_pixel], "with:", sm_true[t_s])
                        arr_sm_hru[y_pixel, x_pixel] = sm_true[t_s]

        return arr_sm_hru

    def load_target_data(self, quad):
        dataset_hru = []
        dates = os.listdir(self.hru_sm_path + quad)
        filtered_dates = [date for date in dates if self.check_season(date)]

        for d in filtered_dates:
            hru_file = gdal.Open(self.hru_sm_path + quad + "/" + d)
            if self.check_if_station_present(quad, d):
                hru_file = self.generate_lats_lon_list(quad, d, hru_file)
            else:
                hru_file = hru_file.ReadAsArray()
            hru_file = hru_file.reshape(1, hru_file.shape[0], hru_file.shape[1])
            hru_file = self.scale_hru(self.resize_image(np.transpose(hru_file, (1, 2, 0))))
            dataset_hru.append(hru_file)

        return np.array(dataset_hru), filtered_dates


    def load_target_data_per_month(self, quad, corrected_hru=False, training=True):
        dataset_hru = []
        if corrected_hru:
            dates = os.listdir(self.hru_corrected_path_model_2 + quad)
        else:
            dates = os.listdir(self.hru_sm_path + quad)

        filtered_dates = [date for date in dates if date[:6] == '201806']

        for d in filtered_dates:
                if corrected_hru:
                    hru_file = gdal.Open(self.hru_corrected_path_model_2 + quad + "/" + d).ReadAsArray()
                else:
                    hru_file = gdal.Open(self.hru_sm_path + quad + "/" + d).ReadAsArray()

                hru_file = hru_file.reshape(1, hru_file.shape[0], hru_file.shape[1])
                hru_file = self.scale_hru(self.resize_image(np.transpose(hru_file, (1, 2, 0))))
                dataset_hru.append(hru_file)

        return np.array(dataset_hru), filtered_dates


    def generate_train_test_data(self, data_quad_list):
        all_imgs_30m_daily, all_imgs_30m_static, all_target_sm, all_quads, all_dates, img_30_unscaled_nlcd_daily = [], [], [], [], [], []
        count = 0
        for quad in data_quad_list:
            count += 1
            target_sm, dates_hru = self.load_target_data(quad)
            if len(dates_hru) == 0:
                continue

            img_30m_static, img_30_unscaled_nlcd = self.load_static_datasets(quad)
            if img_30m_static is None:
                continue

            for i in range(len(dates_hru)):
                img_30m_daily = self.load_daily_datasets(quad, dates_hru[i])
                if img_30m_daily is None:
                    continue

                if target_sm[i] is None:
                    continue

                all_imgs_30m_static.append(img_30m_static)
                all_imgs_30m_daily.append(img_30m_daily)
                img_30_unscaled_nlcd_daily.append(img_30_unscaled_nlcd)
                all_target_sm.append(target_sm[i])
                all_quads.append(quad)
                all_dates.append(dates_hru[i])

        all_imgs_30m_daily = np.array(all_imgs_30m_daily)
        all_imgs_30m_static = np.array(all_imgs_30m_static)
        all_target_sm = np.array(all_target_sm)
        img_30_unscaled_nlcd_daily = np.array(img_30_unscaled_nlcd_daily)
        all_quads = np.array(all_quads)
        all_dates = np.array(all_dates)

        return all_imgs_30m_daily, all_imgs_30m_static, all_target_sm, all_quads, all_dates, img_30_unscaled_nlcd_daily

    def generate_train_test_data_all_quads(self, data_quad_list, corrected_hru=False,
                                         training=True, one_per_quad=False):
        all_imgs_30m_daily, all_imgs_30m_static, all_target_sm, all_quads, all_dates = [], [], [], [], []
        count = 0
        for quad in data_quad_list:
            count += 1
            target_sm, dates_hru = self.load_target_data_per_month(quad, corrected_hru=corrected_hru, training=training)
            if len(dates_hru) == 0:
                continue

            img_30m_static = self.load_static_datasets(quad)
            if img_30m_static is None:
                continue

            for i in range(len(dates_hru)):
                img_30m_daily = self.load_daily_datasets(quad, dates_hru[i])
                if img_30m_daily is None:
                    continue

                if target_sm[i] is None:
                    continue

                all_imgs_30m_static.append(img_30m_static)
                all_imgs_30m_daily.append(img_30m_daily)
                all_target_sm.append(target_sm[i])
                all_quads.append(quad)
                all_dates.append(dates_hru[i])

        all_imgs_30m_daily = np.array(all_imgs_30m_daily)
        all_imgs_30m_static = np.array(all_imgs_30m_static)
        all_target_sm = np.array(all_target_sm)
        all_quads = np.array(all_quads)
        all_dates = np.array(all_dates)
        return all_imgs_30m_daily, all_imgs_30m_static, all_target_sm, all_quads, all_dates

    def generate_train_test_data_model_1(self, data_quad_list, corrected_hru=False,
                                         training=True):
        all_imgs_30m_static, all_target_sm, all_quads, all_dates = [], [], [], []
        count = 0
        for quad in data_quad_list:
            count += 1
            target_sm, dates_hru = self.load_target_data_per_month(quad, corrected_hru=corrected_hru, training=training)
            if len(dates_hru) == 0:
                continue

            img_30m_static = self.load_static_datasets(quad)
            if img_30m_static is None:
                continue

            all_imgs_30m_static.append(img_30m_static)
            all_quads.append(quad)
            all_target_sm.append(target_sm[0])
            all_dates.append(dates_hru[0])

        all_imgs_30m_static = np.array(all_imgs_30m_static)
        all_target_sm = np.array(all_target_sm)
        all_quads = np.array(all_quads)
        all_dates = np.array(all_dates)
        return all_imgs_30m_static, all_target_sm, all_quads, all_dates

    def load_all_datasets_in_memory_old(self, start_in=0, end_in=500):
        quadhashes = set(os.listdir(self.hru_sm_path))
        total_quad = len(quadhashes)
        # quadhashes = set(quadhashes[start_in:end_in])
        station_quads_set = self.load_station_quads()
        quadhashes = list(quadhashes.intersection(station_quads_set))
        quadhashes.sort()

        print(
            "Found total quadhashes for  {}/{}  start_in: {}, end_in: {}".format(len(quadhashes), total_quad,
                                                                                       start_in, end_in), flush=True)

        all_input_daily, all_input_static, all_target, all_quads, dates, img_30_unscaled_nlcd_daily = self.generate_train_test_data(
            quadhashes)

        combined_data = list(
            zip(all_input_daily, all_input_static, all_target, all_quads, dates, img_30_unscaled_nlcd_daily))
        np.random.seed(42)
        np.random.shuffle(combined_data)

        print("Total samples found: ", len(combined_data))
        if len(combined_data) > 16000:
            combined_data = combined_data[:16000]

        split_index = int(0.9 * len(combined_data))
        train_data = combined_data[:split_index]
        test_data = combined_data[split_index:]
        return train_data, test_data

    def load_all_datasets_in_memory(self, start_in=0, end_in=200):
        quadhashes = os.listdir("/s/chopin/f/proj/fineET/sm_predictions/input_datasets/hru/split_14/")
        total_quad = len(quadhashes)
        random.seed(42)
        random.shuffle(quadhashes)

        station_quads_set = self.load_station_quads()
        quadhashes = list(set(quadhashes).intersection(station_quads_set))
        print("Found station quads: ", len(quadhashes))
        quadhashes.sort()

        split_index = int(0.8 * len(quadhashes))
        train_data_quads = quadhashes[:split_index]
        test_data_quads = quadhashes[split_index:]

        quadhashes_others = quadhashes[start_in:end_in]
        train_data_quads.extend(quadhashes_others)
        print(
            "Found total quadhashes for  training {}/{}  and testing {}/{} start_in: {}, end_in: {}".format(len(train_data_quads), total_quad, len(test_data_quads), total_quad,
                                                                                 start_in, end_in), flush=True)

        all_input_daily, all_input_static, all_target, all_quads, dates, img_30_unscaled_nlcd_daily = self.generate_train_test_data(
            train_data_quads)
        train_data = list(
            zip(all_input_daily, all_input_static, all_target, all_quads, dates, img_30_unscaled_nlcd_daily))
        np.random.seed(42)
        np.random.shuffle(train_data)
        print("Total samples found for training: ", len(train_data))
        if len(train_data) > 16000:
            train_data = train_data[:16000]

        all_input_daily, all_input_static, all_target, all_quads, dates, img_30_unscaled_nlcd_daily = self.generate_train_test_data(
            test_data_quads)
        test_data = list(
            zip(all_input_daily, all_input_static, all_target, all_quads, dates, img_30_unscaled_nlcd_daily))
        np.random.seed(42)
        np.random.shuffle(test_data)

        print("Total samples found for testing: ", len(test_data))

        return train_data, test_data


def calculate_potential_retention(cn_based_on_amc):
            cn_based_on_amc[cn_based_on_amc < 30] = 30
            cn_based_on_amc[cn_based_on_amc > 100] = 100
            s_potential = (25400 / cn_based_on_amc) - 254
            return s_potential

def final_cn(amc_condition, CN1, CN2, CN3):

            cn_based_on_amc = torch.zeros_like(amc_condition, dtype=torch.float)

            cn_based_on_amc[amc_condition == 0] = CN1[amc_condition == 0]
            cn_based_on_amc[amc_condition == 1] = CN2[amc_condition == 1]
            cn_based_on_amc[amc_condition == 2] = CN3[amc_condition == 2]
            return cn_based_on_amc

def get_amc_conditions(last_5_days_prec, months):
            soil_amc = torch.zeros_like(last_5_days_prec)
            is_growing = False

            # Growing: March or April to September or October.
            if 3 <= months <= 10:
                is_growing = True

            if is_growing:
                condition_0 = (last_5_days_prec < 35.6)
                soil_amc[condition_0] = 0

                condition_1 = (last_5_days_prec >= 35.6) & (last_5_days_prec <= 53.3)
                soil_amc[condition_1] = 1

                condition_2 = (last_5_days_prec > 53.3)
                soil_amc[condition_2] = 2

            else:
                condition_0 = (last_5_days_prec < 12.7)
                soil_amc[condition_0] = 0

                condition_1 = (last_5_days_prec >= 12.7) & (last_5_days_prec <= 27.9)
                soil_amc[condition_1] = 1

                condition_2 = (last_5_days_prec > 27.9)
                soil_amc[condition_2] = 2

            return soil_amc

def load_last_5_days_precipitation(quad, date_string):
            root_path = "/s/chopin/f/proj/fineET/"
            gridmet_path = root_path + "sm_predictions/input_datasets/gridmet/split_14/"
            date_format = "%Y%m%d"
            date = datetime.strptime(date_string[:-4], date_format)
            previous_five_days = []

            for i in range(5):
                previous_day = date - timedelta(days=i + 1)
                previous_day_str = previous_day.strftime(date_format) + '.tif'
                previous_five_days.append(previous_day_str)

            total_prec = torch.zeros((64,64))
            for d in previous_five_days:
                if os.path.exists(gridmet_path + quad[:12] + "/" + d):
                    img_grid = gdal.Open(gridmet_path + quad[:12] + "/" + d).ReadAsArray()
                    total_prec += resize(img_grid[2, :, :], (64, 64), order=0, preserve_range=True, anti_aliasing=False)
                    # total_prec += np.mean(img_grid[2, :, :])
            # total_prec = torch.full((64, 64), total_prec)
            return total_prec

def calculate_CN1_3(CN2):
            return CN2 / (2.281 - (0.01281 * CN2)), CN2 / (0.427 + (0.00573 * CN2))

def calculate_CN2(land_cover, soil_category):
            cn_lookup = {
                11: [100, 100, 100, 100],
                12: [40, 40, 40, 40],
                21: [49, 69, 79, 84],
                22: [77, 86, 91, 94],
                23: [89, 92, 94, 95],
                24: [98, 98, 98, 98],
                31: [77, 86, 91, 94],
                41: [32, 48, 57, 63],
                42: [39, 58, 73, 80],
                43: [46, 60, 68, 74],
                52: [49, 68, 79, 84],
                71: [64, 71, 81, 89],
                81: [49, 69, 79, 84],
                82: [71, 80, 87, 90],
                90: [88, 89, 90, 91],
                95: [89, 90, 91, 92]
            }

            land_cover = land_cover.numpy()
            soil_category = soil_category.numpy()

            flatten_lan = land_cover.flatten()
            flatten_soil_category = soil_category.flatten()

            cns = []
            for p in range(0, len(flatten_lan)):
                cns.append(cn_lookup[flatten_lan[p]][flatten_soil_category[p]])

            cns = torch.tensor(cns, dtype=torch.float)
            cns = cns.reshape(land_cover.shape)

            return cns

# cite this- https://www.nrc.gov/docs/ML1421/ML14219A437.pdf
def get_soil_class(soil_texture):
            soil_class = torch.zeros_like(soil_texture, dtype=torch.int)

            # Condition 0: sand, loamy sand, or sandy loam
            condition_0 = (soil_texture == 0) | (soil_texture == 1)
            soil_class[condition_0] = 0

            # Condition 1: silt loam or loam
            condition_1 = (soil_texture == 2) | (soil_texture == 3)
            soil_class[condition_1] = 1

            # Condition 2: sandy clay loam
            condition_2 = (soil_texture == 4)
            soil_class[condition_2] = 2

            # Condition 3: clay loam, silty clay loam, sandy clay, silty clay, or clay
            condition_3 = (soil_texture >= 5)
            soil_class[condition_3] = 3

            return soil_class

def get_soil_texture(silt_per, sand_per, clay_per):
            texture = torch.zeros_like(sand_per, dtype=torch.int)

            # Condition 1: clay
            condition = (0 <= sand_per) & (sand_per <= 0.45) & (0 <= silt_per) & (silt_per <= 0.4) & (
                    0.4 <= clay_per) & (clay_per <= 1)
            texture[condition] = 1

            # Condition 2: sandy clay
            condition = (0.45 <= sand_per) & (sand_per <= 0.65) & (0 <= silt_per) & (silt_per <= 0.2) & (
                        0.35 <= clay_per) & (clay_per <= 0.55)
            texture[condition] = 2

            # Condition 3: silty clay
            condition = (0 <= sand_per) & (sand_per <= 0.2) & (0.4 <= silt_per) & (silt_per <= 0.6) & (
                        0.4 <= clay_per) & (clay_per <= 0.6)
            texture[condition] = 3

            # Condition 4: sandy clay loam
            condition = (0.45 <= sand_per) & (sand_per <= 0.8) & (0 <= silt_per) & (silt_per <= 0.28) & (
                        0.2 <= clay_per) & (clay_per <= 0.35)
            texture[condition] = 4

            # Condition 5: clay loam
            condition = (0.2 <= sand_per) & (sand_per <= 0.45) & (0.15 <= silt_per) & (silt_per <= 0.52) & (
                        0.27 <= clay_per) & (clay_per <= 0.4)
            texture[condition] = 5

            # Condition 6: silty clay loam
            condition = (0 <= sand_per) & (sand_per <= 0.2) & (0.4 <= silt_per) & (silt_per <= 0.73) & (
                        0.27 <= clay_per) & (clay_per <= 0.4)
            texture[condition] = 6

            # Condition 7: loam
            condition = (0.23 <= sand_per) & (sand_per <= 0.52) & (0.28 <= silt_per) & (silt_per <= 0.5) & (
                        0.07 <= clay_per) & (clay_per <= 0.27)
            texture[condition] = 7

            # Condition 8: sandy loam
            condition = (0.5 <= sand_per) & (sand_per <= 0.7) & (0 <= silt_per) & (silt_per <= 0.5) & (
                    0 <= clay_per) & (clay_per <= 0.2)
            texture[condition] = 8

            # Condition 9: silt loam
            condition = (0.2 <= sand_per) & (sand_per <= 0.5) & (0.74 <= silt_per) & (silt_per <= 0.88) & (
                        0 <= clay_per) & (clay_per <= 0.27)
            texture[condition] = 9

            # Condition 10: silt
            condition = (0 <= sand_per) & (sand_per <= 0.2) & (0.88 <= silt_per) & (silt_per <= 1.0) & (
                    0 <= clay_per) & (clay_per <= 0.12)
            texture[condition] = 10

            # Condition 11: loamy sand
            condition = (0.7 <= sand_per) & (sand_per <= 0.86) & (0 <= silt_per) & (silt_per <= 0.3) & (
                        0 <= clay_per) & (clay_per <= 0.15)
            texture[condition] = 11

            # Condition 12: sand
            condition = (0.86 <= sand_per) & (sand_per <= 1) & (0 <= silt_per) & (silt_per <= 0.14) & (
                    0 <= clay_per) & (clay_per <= 0.1)
            texture[condition] = 12


            return texture

def get_cn_final(all_input_1, all_quads, all_months, land_cover):
        texture = get_soil_texture(all_input_1[5], all_input_1[6], all_input_1[7])
        soil_classes = get_soil_class(texture)
        land_cover = land_cover.int()
        cns2 = calculate_CN2(land_cover, soil_classes)
        cns1, cns3 = calculate_CN1_3(cns2)
        month = int(all_months[4:6])
        total_prec = load_last_5_days_precipitation(all_quads, all_months)
        amc_condition = get_amc_conditions(total_prec, month)
        cn = final_cn(amc_condition, cns1, cns2, cns3)
        s_poten = calculate_potential_retention(cn)
        return s_poten / 593

class QuadhashDataset(Dataset):
    def __init__(self, data, training=True):
        self.all_input_daily, self.all_input_static, self.all_target, self.all_quads, self.dates, self.img_30_unscaled_nlcd_daily = zip(*data)
        if training:
            print("Total no. of samples returning for TRAINING: ", len(self.all_quads))
        else:
            print("Total no. of samples returning for TESTING: ", len(self.all_quads), flush=True)

    def __len__(self):
        return len(self.all_quads)

    def __getitem__(self, index):

        all_input_1 = self.all_input_daily[index]  # daily
        all_input_2 = self.all_input_static[index]  # static
        merged_image = np.concatenate((all_input_1, all_input_2), axis=2)
        merged_image = torch.tensor(merged_image).permute(2, 0, 1)
        # img_30_unscaled_nlcd_daily = self.img_30_unscaled_nlcd_daily[index]
        # img_30_unscaled_nlcd_daily = torch.tensor(img_30_unscaled_nlcd_daily).permute(2, 0, 1)
        # all_s_poten = torch.unsqueeze(get_cn_final(merged_image, all_quads, all_months, img_30_unscaled_nlcd_daily[0]), 0)
        all_target = self.all_target[index]
        all_quads = self.all_quads[index]
        all_months = self.dates[index]
        texture = torch.unsqueeze(get_soil_texture(merged_image[5], merged_image[6], merged_image[7]), 0)
        merged_image = torch.cat((merged_image, texture), dim=0)
        all_target = torch.tensor(all_target).permute(2, 0, 1)

        return merged_image, all_target, all_quads, all_months

class QuadhashDatasetAllQuads(Dataset):
    def __init__(self, training=True, one_per_quad=False, start_in = 0, end_in=500, corrected_hru=True):
        self.data_loader = Dataloader()

        quadhashes = os.listdir(self.data_loader.hru_sm_path)
        random_seed = 42
        random.seed(random_seed)
        random.shuffle(quadhashes)
        total_quad = len(quadhashes)
        quadhashes = quadhashes[start_in:end_in]

        print("Testing quadhashes for {}/{}, start- {} end- {} ".format(len(quadhashes), total_quad, start_in, end_in))

        self.all_input_daily, self.all_input_static, self.all_target, self.all_quads, self.dates = self.data_loader.generate_train_test_data_all_quads(
            quadhashes, corrected_hru=corrected_hru, one_per_quad=one_per_quad,
            training=training)

        print("Samples found: ", len(self.all_quads))

    def __len__(self):
        return len(self.all_quads)

    def __getitem__(self, index):

        all_input_1 = self.all_input_daily[index]  # daily
        all_input_2 = self.all_input_static[index]  # static

        all_target = self.all_target[index]
        all_quads = self.all_quads[index]
        all_months = self.dates[index]

        merged_image = np.concatenate((all_input_1, all_input_2), axis=2)

        merged_image = torch.tensor(merged_image).permute(2, 0, 1)
        all_target = torch.tensor(all_target).permute(2, 0, 1)

        return merged_image, all_target, all_quads, all_months

class QuadhashDatasetModel1(Dataset):
    def __init__(self, training=True, one_per_quad=False, corrected_hru=True):
        self.data_loader = Dataloader()

        random_seed = 42
        random.seed(random_seed)
        if corrected_hru:
            self.quadhashes = os.listdir(self.data_loader.hru_corrected_path_model_1_monthly)
        else:
            self.quadhashes = os.listdir(self.data_loader.hru_sm_path)

        self.training = training
        random.shuffle(self.quadhashes)
        print("Total quadhashes for  {} ".format(len(self.quadhashes)))
        if len(self.quadhashes) > 5500:
            self.quadhashes = random.sample(self.quadhashes, 5500)

        if training:
            self.quadhashes = self.quadhashes[:5000]
        else:
            self.quadhashes = self.quadhashes[5000:]

        self.all_input_static, self.all_target, self.all_quads, self.dates = self.data_loader.generate_train_test_data_model_1(
            self.quadhashes, corrected_hru=corrected_hru,
            training=training)

        print("No. of samples returning: ", len(self.all_quads))

    def __len__(self):
        return len(self.all_quads)

    def __getitem__(self, index):

        all_input_2 = self.all_input_static[index]  # static

        all_target = self.all_target[index]
        all_quads = self.all_quads[index]
        all_months = self.dates[index]

        all_input_2 = torch.tensor(all_input_2).permute(2, 0, 1)
        all_target = torch.tensor(all_target).permute(2, 0, 1)

        return all_input_2, all_target, all_quads, all_months


if __name__ == '__main__':
    batch_size = 5
    my_loader = Dataloader()

    train_data, _ = my_loader.load_all_datasets_in_memory(0, 3)
    train_dataset = QuadhashDataset(train_data, training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for all_input_1, all_target, all_quads, all_months in train_dataloader:
        print(all_input_1.shape, all_target.shape, len(all_quads), len(all_months))
        break

