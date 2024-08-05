import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vanGenuchten_dataset_loader import QuadhashDataset, QuadhashDatasetModel1, QuadhashDatasetAllQuads, Dataloader
import torch
import csv
import torch.nn as nn
import gdal
np.set_printoptions(suppress=True)
import cv2
import math
import joblib
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_no_of_parameters(model):
    numel_list = [p.numel() for p in model.parameters() if p.requires_grad == True]
    print("Total number of trainable parameters in the model: ", sum(numel_list))
    return sum(numel_list)


def get_input_batch(data, isTraining=True, batch_size=64, isModel1=False, shuffle = False):
    batch_size = batch_size
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isTraining:
        if isModel1:
            my_dataset = QuadhashDatasetModel1(training=True, corrected_hru=False)
        else:
            my_dataset = QuadhashDataset(data, training=True)
    else:
        if isModel1:
            my_dataset = QuadhashDatasetModel1(training=False, corrected_hru=False)
        else:
            my_dataset = QuadhashDataset(data, training=False)

    dataloader_new = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3)
    return dataloader_new


def plot_targed_inferred_only_samples(output_sample, target_img, l1_error, epoch, out_path, input_sample, training=True,
                                      PSNR_acc=0, image_ssim_nan=0):
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 3, 1)
    plt.title("Land Cover")
    input_sample1 = input_sample[25, :, :].cpu()
    input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.imshow(input_sample1)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title("Soil Texture")
    input_sample1 = input_sample[-1, :, :].cpu()
    # input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.imshow(input_sample1)
    plt.axis('off')
    plt.colorbar()
    #
    # plt.subplot(2, 3, 3)
    # plt.title("Polaris theta_s")
    # input_sample1 = input_sample[9, :, :].cpu()
    # # input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    # plt.imshow(input_sample1)
    # plt.axis('off')
    # plt.colorbar()

    plt.subplot(2, 3, 3)
    input_sample1 = input_sample[2, :, :].cpu()
    plt.title("Precipitation (mm) - " + str(input_sample1[0][0].numpy()))
    # input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.imshow(input_sample1)
    plt.axis('off')
    plt.colorbar()

    target_img = target_img.numpy()
    target_img[target_img < 0] = np.nan
    output_sample = torch.unsqueeze(output_sample[0], 0).numpy()

    plt.subplot(2, 3, 3)
    # input_sample1 = input_sample[2, :, :].cpu()
    plt.title("Error Map (MAE)")
    error_map =  np.ma.masked_equal(np.abs(target_img.transpose(1, 2, 0) - output_sample.transpose(1, 2, 0)), -1)
    # input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.imshow(error_map)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    input_sample1 = input_sample[0, :, :].cpu()
    input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.title("SMAP: " + str(input_sample1[0][0]))
    plt.imshow(input_sample1, cmap='cividis_r')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 5)


    sm_vmin = np.nanmin((output_sample, target_img))
    sm_vmax = np.max((output_sample, target_img))

    plt.title("Target Image")
    target_img = np.ma.masked_equal(target_img.transpose(1, 2, 0), -1)
    plt.imshow(target_img, cmap='cividis_r', vmin=sm_vmin, vmax=sm_vmax)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    output_sample = np.ma.masked_equal(output_sample.transpose(1, 2, 0), -1)
    plt.title("Generated Image \n L1 loss: " + str(round(l1_error.item(), 2)) + " PSNR: " + str(
        round(PSNR_acc.item(), 1)) + "dB" + "\nSSIM:" + str(image_ssim_nan))
    plt.imshow(output_sample, cmap='cividis_r', vmin=sm_vmin, vmax=sm_vmax)
    plt.axis('off')
    plt.colorbar()

    if training:
        plt.savefig(out_path + 'samples/' + str(epoch) + '.png')
    else:
        os.makedirs(out_path + 'testing', exist_ok=True)
        plt.savefig(out_path + 'testing/' + str(epoch) + '.png')
        print("Sacing at: ", out_path + 'testing/' + str(epoch) + '.png')
    plt.close()


def perform_inferences_tiffs(folder, iscorrected=False, start_in=0, end_in=500):
    model2 = load_model_weights(folder)
    model2 = model2.to(device).float()
    model2 = model2.eval()

    inp_path = "/s/lovelace/f/nobackup/shrideep/sustain/sm_predictions/input_datasets/hru/split_14/"
    # inp_path = "/s/chopin/f/proj/fineET/sm_predictions/input_datasets/hru/split_14/"

    batch_size = 256
    test_dataset = QuadhashDatasetAllQuads(training=False, corrected_hru=iscorrected, start_in=start_in, end_in=end_in)
    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    os.makedirs("/s/chopin/f/proj/fineET/sm_predictions/outputs_daily", exist_ok=True)

    with torch.no_grad():
        for (input_sample, target_img, allquads, dates) in dataloader_new:
            input_sample = input_sample.to(torch.float32).to(device)
            output_sample = model2(input_sample).cpu().numpy()

            for i in range(output_sample.shape[0]):
                d = dates[i]

                output_geotiff_path = "/s/chopin/f/proj/fineET/sm_predictions/outputs_daily/model_output_" + d[
                                                                                                             :4] + "_" + d[
                                                                                                                         4:6] + "_" + d[
                                                                                                                                      6:8]
                os.makedirs(output_geotiff_path, exist_ok=True)
                q = allquads[i]

                if os.path.exists(inp_path + q):
                    newf = os.listdir(inp_path + q)[0]
                    sm_hru = gdal.Open(inp_path + q + "/" + newf)

                    geotransform = sm_hru.GetGeoTransform()
                    projection = sm_hru.GetProjection()
                    resized_out_image = cv2.resize(output_sample[i][0], (sm_hru.RasterXSize, sm_hru.RasterYSize))
                    driver = gdal.GetDriverByName('GTiff')
                    b_dataset = driver.Create(output_geotiff_path + "/" + q + ".tif", sm_hru.RasterXSize,
                                              sm_hru.RasterYSize, 1,
                                              gdal.GDT_Float32)  # Change data type as needed

                    b_dataset.SetProjection(projection)
                    b_dataset.SetGeoTransform(geotransform)
                    band = b_dataset.GetRasterBand(1)
                    band.WriteArray(resized_out_image)
                    b_dataset = None
                    sm_hru = None
                else:
                    continue


def generator_loss(generated_image, target_img):
    l1_loss = nn.L1Loss()
    mask = (target_img != -1).float()
    l1_l = l1_loss(generated_image * mask, target_img * mask)
    return l1_l


def generator_acc(generated_image, target_img):
    mask = (target_img != -1).float()
    generated_image = generated_image * mask
    target_img = target_img * mask
    mse = nn.functional.mse_loss(generated_image, target_img)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr


def load_losses_train_test(dirno):
    out_path = "/s/chopin/f/proj/fineET/sm_predictions/outputs_richards/" + str(dirno) + "/"

    train_loss_file = out_path + 'train_loss.txt'
    test_loss_file = out_path + 'test_loss.txt'

    with open(train_loss_file, 'r') as f:
        train_loss_lines = f.readlines()
    train_loss_values = [float(line.strip()) for line in train_loss_lines]

    with open(test_loss_file, 'r') as f:
        test_loss_lines = f.readlines()
    test_loss_values = [float(line.strip()) for line in test_loss_lines]

    return train_loss_values, test_loss_values


def load_model_weights(folder):
    out_path = "/s/chopin/f/proj/fineET/sm_predictions/outputs_richards/" + str(folder) + "/"
    model_path = out_path + "model_weights.pth"
    loaded_model = torch.jit.load(model_path)
    return loaded_model


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.criterion2 = nn.MSELoss()

    def forward(self, input, target):
        mask = (target != -1).float()
        loss_mae = self.criterion(input * mask, target * mask)
        loss_mse = self.criterion2(input * mask, target * mask)

        return loss_mse, loss_mae

'''This unet architecture takes (64,64) and generates sm at 64,64,1'''
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, final_out=1):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        self.downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                  stride=2, padding=1, bias=False)
        self.downrelu = nn.LeakyReLU(0.2, True)
        self.downnorm = norm_layer(inner_nc)
        self.uprelu = nn.ReLU(True)
        self.upnorm = norm_layer(outer_nc)
        self.last_activation = nn.ReLU(True)

        if outermost:
            self.start_conv = nn.Conv2d(2, 512, kernel_size=3, stride=1, padding=1)
            self.start_conv3 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
            self.start_conv4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
            self.start_conv5 = nn.Conv2d(2, final_out, kernel_size=3, stride=1, padding=1)

            self.upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                             kernel_size=4, stride=2,
                                             padding=1)
            self.down = [self.downconv]
            self.up = [self.uprelu, self.upconv, nn.ReLU(True)]
            self.model = nn.Sequential(*self.down, submodule, *self.up)
        elif innermost:
            self.upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                             kernel_size=4, stride=2,
                                             padding=1, bias=False)
            self.down = [self.downrelu, self.downconv]
            self.up = [self.uprelu, self.upconv, self.upnorm]
            self.model = nn.Sequential(*self.down, *self.up)
        else:
            self.upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                             kernel_size=4, stride=2,
                                             padding=1, bias=False)
            self.down = [self.downrelu, self.downconv, self.downnorm]
            self.up = [self.uprelu, self.upconv, self.upnorm]

            if use_dropout:
                self.model = nn.Sequential(*self.down, submodule, *self.up, nn.Dropout(0.5))
            else:
                self.model = nn.Sequential(*self.down, submodule, *self.up)

    def forward(self, x):
        if self.outermost:
            # removed 29
            inp2 = x[:, [25, 29], :, :]
            x = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], :, :]

            input_conv = self.start_conv(inp2)
            input_conv = self.uprelu(input_conv)
            input_conv = self.start_conv3(input_conv)
            input_conv = self.uprelu(input_conv)
            input_conv = self.start_conv4(input_conv)
            input_conv = self.uprelu(input_conv)

            my_model = self.model(x)
            concatenated = torch.cat([input_conv, my_model], 1)
            newc = self.start_conv5(concatenated)
            newc = self.last_activation(newc)
            return newc
        else:
            return torch.cat([x, self.model(x)], 1)


'''This unet architecture takes (64,64) and generates sm 64,64,1'''
class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, nf=32, norm_layer=nn.BatchNorm2d, use_dropout=False, final_out=1):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, innermost=True,
                                             norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_dropout=use_dropout)

        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer, final_out=final_out)

    def forward(self, input):
        return self.model(input)


def load_station_quads():
        input_path = "/s/chopin/f/proj/fineET/sm_predictions/input_datasets/station_data/split/5/"
        quads_stations = set(os.listdir(input_path))
        okhal_path = "/s/chopin/f/proj/fineET/sm_predictions/input_datasets/hru/split_14/"
        quads_okh = set(os.listdir(okhal_path))
        overlaps = quads_stations.intersection(quads_okh)
        return list(overlaps)

def perform_inferences(folder, one_per_quad=False, test_data=None, isModel1=False, start_in=0, end_in=500):
    model2 = load_model_weights(folder)
    model2 = model2.to(device).float()
    model2 = model2.eval()
    if test_data is None:
        my_dataloader = Dataloader()
        _, test_data = my_dataloader.load_all_datasets_in_memory(start_in, end_in)
    print("Performing inference on folder: ", folder)
    batch_size = 128
    out_path = "/s/chopin/f/proj/fineET/sm_predictions/outputs_richards/" + str(folder) + "/"
    os.makedirs(out_path, exist_ok=True)
    if isModel1:
        test_dataset = QuadhashDatasetModel1(training=False, one_per_quad=one_per_quad, corrected_hru=False)
    else:
        test_dataset = QuadhashDataset(test_data, training=False)

    total_loss, total_psnr = [], []
    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    count = 0
    station_list = load_station_quads()
    with torch.no_grad():
        for (input_sample, target_img, allquads, months) in dataloader_new:
            input_sample = input_sample.to(torch.float32).to(device)
            output_sample = model2(input_sample).cpu()
            target_img = target_img.cpu()

            for i in range(input_sample.shape[0]):
                l1_error = generator_loss(output_sample[i][0], target_img[i][0])
                psnr_acc = generator_acc(output_sample[i][0], target_img[i][0])
                total_loss.append(round(l1_error.item(), 4))
                psna = psnr_acc.item()
                if psna == float('inf'):
                    continue
                total_psnr.append(round(psna, 1))

                if not allquads[i] in station_list:
                    continue

                count += 1
                if count <= 200:
                    x, imag_ssim_all = ssim(output_sample[i][0].numpy(), target_img[i][0].numpy(), data_range=1, full=True, win_size=5)
                    imag_ssim_all[target_img[i][0].numpy() < 0] = np.nan
                    image_ssim_nan = np.round(np.nanmean(imag_ssim_all), 2)
                    plot_targed_inferred_only_samples(output_sample[i], target_img[i], l1_error, count, out_path,
                                                      input_sample[i], training=False, PSNR_acc=psnr_acc, image_ssim_nan=image_ssim_nan)
    loss_tot = np.round(np.mean(np.array(total_loss)), 3)
    psnr_tot = np.round(np.nanmean(np.array(total_psnr)), 5)
    print("Samples tested: ", len(total_loss), flush=True)
    return loss_tot, psnr_tot

def USDA_triangle(silt, sand, clay):
    if sand <= 45 and silt <= 40 and clay >= 40:
        texture = "Clay"
        out = 1
    elif sand <= 65 and sand >= 45 and silt <= 20 and clay >= 35 and clay <= 55:
        texture = "Sandy Clay"
        out = 2
    elif sand <= 20 and silt >= 40 and silt <= 60 and clay >= 40 and clay <= 60:
        texture = "Silty Clay"
        out = 3
    elif sand >= 45 and sand <= 80 and silt <= 28 and clay >= 20 and clay <= 35:
        texture = "Sandy Clay Loam"
        out = 4
    elif sand >= 20 and sand <= 45 and silt >= 15 and silt <=53 and clay >= 27 and clay <= 40:
        texture = "Clay Loam"
        out = 5
    elif sand <= 20 and silt >= 40 and silt <= 73 and clay >= 27 and clay <= 40:
        texture = "Silty Clay Loam"
        out = 6
    elif sand >= 23 and sand <= 52 and silt >= 28 and silt <= 50 and clay >= 7 and clay <= 27:
        texture = "Loam"
        out = 7
    elif sand >= 43 and sand <= 85 and silt <= 50 and clay <= 20:
        texture = "Sandy Loam"
        out = 8
    elif sand <= 50 and silt >= 50 and silt <= 88 and clay <= 27:
        texture = "Silty Loam"
        out = 9
    elif sand <= 20 and silt >= 88 and clay <= 12:
        texture = "Silt"
        out = 10
    elif sand >= 70 and sand <= 85 and silt <= 30 and clay <= 15:
        texture = "Loamy Sand"
        out = 11
    elif sand >= 85 and silt <= 15 and clay <= 10:
        texture = "Sand"
        out = 12
    else:
        texture = "Not Available"
        out = 13
    return out, texture

def soil_class_mapping(class_number):
    if class_number == 1:
        return 'Clay'
    elif class_number == 2:
        return 'Sandy Clay'
    elif class_number == 3:
        return "Silty Clay"
    elif class_number == 4:
        return "Sandy Clay Loam"
    elif class_number == 5:
        return "Clay Loam"
    elif class_number == 6:
        return "Silty Clay Loam"
    elif class_number == 7:
        return "Loam"
    elif class_number == 8:
        return "Sandy Loam"
    elif class_number == 9:
        return "Silty Loam"
    elif class_number == 10:
        return "Silt"
    elif class_number == 11:
        return "Loamy Sand"
    elif class_number == 12:
        return "Sand"
    else:
        return "Not Available"

class CustomLoss_richards_psnr(nn.Module):
    def __init__(self, mae_weight=1, psnr_weight=1, lc_weight=0.4, ssim_weight=0.5, psi_penalty_weight=0.01, rule_sm_weight=10, initial_alpha=1.0):
        super(CustomLoss_richards_psnr, self).__init__()
        self.mae_criterion = nn.L1Loss()
        self.mse_criterion = nn.MSELoss()
        self.mae_weight = mae_weight
        self.psnr_penalty_weight = psnr_weight
        self.initial_alpha = initial_alpha
        self.psi_penalty_weight = psi_penalty_weight
        self.land_cover_penalty_weight = lc_weight
        self.rule_sm_weight = rule_sm_weight

    def psnr_loss(self, input, target, mask):
        input = input * mask
        target = target * mask
        mse = nn.functional.mse_loss(input, target)
        psnr = 20 * torch.log10(1 / torch.sqrt(mse))
        return 52 - psnr

    def soil_matric_potential(self, generated_image, input_img, mask):
        n = torch.round(torch.unsqueeze(input_img[:, 14, :, :], 1), decimals=3)
        alpha = torch.round(torch.unsqueeze(input_img[:, 17, :, :], 1), decimals=3)
        theta_r = torch.round(torch.unsqueeze(input_img[:, 10, :, :], 1), decimals=3)
        theta_s = torch.round(torch.unsqueeze(input_img[:, 9, :, :], 1), decimals=3)
        m = (n - 1) / n

        threshold = 0.01
        theta_diffs = torch.round(generated_image - theta_r, decimals=3)
        theta_diffs[torch.abs(theta_diffs) < threshold] = threshold
        theta_diffs_2 = torch.round(theta_s - theta_r, decimals=3)
        theta_diffs_2[torch.abs(theta_diffs_2) < threshold] = threshold

        se = theta_diffs / theta_diffs_2
        se[se <= 0] = 0
        se[se > 1] = 1
        se = torch.round(se, decimals=3)

        psi = ((theta_diffs_2 / theta_diffs) ** (1 / m)) - 1
        psi[psi < 0.001] = 0.001
        psi = (1 / alpha) * (psi ** (1 / n))

        nan_tensor = torch.tensor(float('nan'), device=generated_image.device, dtype=psi.dtype)

        psi = torch.clamp(psi, min=0, max=15000)
        # mean = psi.mean()
        # std_dev = psi.std()
        # psi = (psi - mean) / std_dev
        psi = torch.where(mask == 1, psi, nan_tensor)
        se = torch.where(mask == 1, se, nan_tensor)
        return psi, se

    def soil_unsaturated_hydaulic_conductivity(self, generated_image, input_img, se):
        mask = (generated_image != -1).float()
        n = torch.unsqueeze(input_img[:, 14, :, :], 1)
        m = (n - 1) / n
        k_saturated = torch.unsqueeze(input_img[:, 11, :, :], 1)
        term = ((1 - ((1 - (torch.clamp(se ** (1 / m), 0, 0.9))) ** m)) ** 2)
        k_unsaturated = k_saturated * torch.sqrt(se) * term
        nan_tensor = torch.tensor(float('nan'), device=generated_image.device, dtype=k_unsaturated.dtype)
        k_unsaturated = torch.where(mask == 1, k_unsaturated, nan_tensor)
        return k_unsaturated

    def mae_psi(self, generated_image, target_image, input_img, mask):
        predicted_psi, _ = self.soil_matric_potential(generated_image, input_img, mask)
        target_psi,_ = self.soil_matric_potential(target_image, input_img, mask)
        # predicted_psi = predicted_psi * mask
        # target_psi = target_psi * mask
        mask = ~(torch.isnan(predicted_psi) | torch.isnan(target_psi))

        predicted_psi_masked = predicted_psi[mask]/15000
        target_psi_masked = target_psi[mask]/15000

        mse = nn.functional.mse_loss(predicted_psi_masked, target_psi_masked)
        mae = nn.functional.l1_loss(predicted_psi_masked, target_psi_masked)
        return mse, mae


    def rule_wrt_saturated_unsaturated(self, generated_image, mask, input_img):
        generated_image = generated_image * mask
        theta_r = torch.unsqueeze(input_img[:, 5, :, :], 1)
        theta_s = torch.unsqueeze(input_img[:, 4, :, :], 1)
        lower_bound_penalty = torch.relu(theta_r - generated_image)
        upper_bound_penalty = torch.relu(generated_image - theta_s)
        penalty = torch.nanmean(lower_bound_penalty + upper_bound_penalty)
        return penalty

    def calculate_landcover_loss(self, input, target, mask, land_cover):
        unique_landc = torch.unique(land_cover)
        input = input * mask
        target = target * mask
        stds = []
        for val in unique_landc:
            mask_l = (land_cover == val).float()
            new_i = input * mask_l
            new_t = target * mask_l
            std_dev = new_i.std()
            std_dev_t = new_t.std()
            stds.append(torch.abs(std_dev_t - std_dev))

        std_tensor = torch.tensor(stds)
        return std_tensor.mean()

    # def mae_psi(self, predicted, target, mask, input_image):
    #     predicted_psi, predicted_se = self.soil_matric_potential(target, input_image, mask)
    #     input_unsaturated_conductivity = self.soil_unsaturated_hydaulic_conductivity(predicted, input_image, predicted_se)
    #
    #     X = input_image[:, [9, 10, 11, 14, 17, 29], :, :]
    #     X = torch.cat((X, input_unsaturated_conductivity), dim=1)
    #     X = X.reshape(-1, 7).cpu()
    #     # Takes input ['theta_s',  'theta_r',  'ksat',     'n',     'alpha', 'soil texture', 'unsaturated_conductivity']
    #     target_psi = self.load_predict_xgboost(X)
    #     predicted_psi = predicted_psi * mask
    #     predicted_psi = torch.nan_to_num(predicted_psi, nan=0.0)
    #     return self.mae_criterion(predicted_psi, target_psi)

    # def load_predict_xgboost(self, X_sat):
    #     xgb_regressor_loaded = joblib.load("/s/chopin/a/grad/paahuni/cl_3.8/deepSoil/models/h_psi_best_model")
    #     out = xgb_regressor_loaded.predict(X_sat)
    #     out = np.maximum(out, 0)
    #     out = out.reshape(-1, 1, 64, 64)
    #     out = torch.tensor(out, device=device, dtype=X_sat.dtype)
    #     return out
    def calculate_alpha_linear(self, epoch, num_epochs):
        return max(0, self.initial_alpha - epoch / num_epochs)

    def calculate_alpha_step(self, epoch, num_epochs):
        step_size = 50
        num_steps = num_epochs // step_size
        current_step = epoch // step_size
        alpha = max(0, self.initial_alpha - current_step * (1 / num_steps))
        return alpha

    def forward(self, predicted, target, land_cover=None, input_image=None, epoch=None, num_epochs=None, pixel_x=None, pixel_y=None):
        mask = (target != -1).float()
        mae_loss = self.mae_criterion(predicted[:, 0, :, :] * mask, target[:, 0, :, :] * mask)
        mse_loss = self.mse_criterion(predicted[:, 0, :, :] * mask, target[:, 0, :, :] * mask)

        psi_diffs_mse, psi_diffs_mae = self.mae_psi(predicted, target, input_image, mask)

        alpha = self.calculate_alpha_step(epoch, num_epochs)
        loss_final = alpha * mse_loss + (1 - alpha) * psi_diffs_mae

        return loss_final, mae_loss, alpha

    # def forward(self, predicted, target, land_cover=None, input_image=None):
    #     # landc_loss = self.calculate_landcover_loss(predicted, target, mask, land_cover)
    #     # psnr_loss_val = self.psnr_loss(predicted[:, 0, :, :], target[:, 0, :, :], mask)
    #     # mae_loss_psi = self.mae_psi(predicted, target, mask, input_image)
    #
    #     mask = (target != -1).float()
    #     mae_loss = self.mae_criterion(predicted[:, 0, :, :] * mask, target[:, 0, :, :] * mask)
    #     mse_loss = self.mse_criterion(predicted[:, 0, :, :] * mask, target[:, 0, :, :] * mask)
    #
    #     psi_diffs_mse, psi_diffs_mae = self.mae_psi(predicted, target, input_image, mask)
    #     delta = 0.02
    #     if mae_loss > delta:
    #         loss_final = 0.5 * mse_loss ** 2
    #     else:
    #         loss_final = 0.5 * mae_loss + (delta * (psi_diffs_mae - 0.5 * delta))
    #
    #     return loss_final, mae_loss
    #     # rule_loss = self.rule_wrt_saturated_unsaturated(predicted, mask, input_image)
    #     # return self.mae_weight * mae_loss + self.psi_penalty_weight * psi_diffs_mse , mae_loss

def load_station(q, d):
    station_path = "/s/chopin/f/proj/fineET/sm_predictions/input_datasets/station_data/split/5/"
    new_p = station_path + q + "/" + d.split(".")[0] + ".txt"
    lat_station, lon_station, sm_true = [], [], []
    if os.path.exists(new_p):
        with open(new_p, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                lat_station.append(float(row[0]))
                lon_station.append(float(row[1]))
                sm_true.append(float(row[2]))
        return lat_station, lon_station, sm_true
    else:
        return None, None, None

def perform_inferences_stations(folder, test_data=None, start_in=0, end_in=100):
    model2 = load_model_weights(folder)
    model2 = model2.to(device).float()
    model2 = model2.eval()
    inp_path = "/s/chopin/f/proj/fineET/sm_predictions/input_datasets/hru_station_only/split_14/"
    all_err, all_stations, all_out_vals = [], [], []

    if test_data is None:
        my_dataloader = Dataloader()
        _, test_data = my_dataloader.load_all_datasets_in_memory(start_in, end_in)
    print("Performing inference on folder: ", folder)
    batch_size = 128
    test_dataset = QuadhashDataset(test_data, training=False)

    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    with torch.no_grad():
        for (input_sample, target_img, allquads, dates) in dataloader_new:
            input_sample = input_sample.to(torch.float32).to(device)
            output_sample = model2(input_sample).cpu().numpy()

            for i in range(output_sample.shape[0]):
                d = dates[i]
                q = allquads[i]
                lat_s, lon_s, sm_s = load_station(q, d)
                if sm_s is not None:
                    # print("Opening hydroblocks", inp_path + q + "/" + d)
                    if os.path.exists(inp_path + q + "/" + d):

                        sm_hru = gdal.Open(inp_path + q + "/" + d)
                        if sm_hru is None:
                            continue

                        resized_out_image = cv2.resize(output_sample[i][0], (sm_hru.RasterXSize, sm_hru.RasterYSize))
                        resized_out_image[resized_out_image == -1] = np.nan

                        for t_s in range(len(sm_s)):
                            x_pixel = int((lon_s[t_s] - sm_hru.GetGeoTransform()[0]) / sm_hru.GetGeoTransform()[1])
                            y_pixel = int((lat_s[t_s] - sm_hru.GetGeoTransform()[3]) / sm_hru.GetGeoTransform()[5])
                            # print("predicted:", resized_out_image[y_pixel, x_pixel], "target: ", sm_s[t_s])
                            errs = abs(resized_out_image[y_pixel, x_pixel] - sm_s[t_s])
                            all_err.append(errs)

                        sm_hru = None
                    else:
                        continue

    print("Average error SM: ", np.mean(np.array(all_err)), " average std: ", np.std(np.array(all_err)), " on sample count: ", len(all_err))


def perform_inferences_station_smap_hru(folder, test_dataset, training=False, one_per_quad=False, iscorrected=False):
    output_geotiff_path = "/s/chopin/f/proj/fineET/sm_predictions/input_datasets/hru/split_14/"
    # output_geotiff_path = "/s/chopin/f/proj/fineET/sm_predictions/input_datasets/smap_9km/split_14/"

    batch_size = 32
    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_err = []
    with torch.no_grad():
        for (input_sample, target_img, allquads, dates) in dataloader_new:
            for i in range(input_sample.shape[0]):
                d = dates[i]
                q = allquads[i]
                lat_s, lon_s, sm_s = load_station(q, d)
                # q = q[:12]
                if sm_s is not None:
                    if os.path.exists(output_geotiff_path + q + "/" + d):
                        sm_hru = gdal.Open(output_geotiff_path + q + "/" + d)

                        if sm_hru is None:
                            continue

                        resized_out_image = sm_hru.ReadAsArray()
                        resized_out_image[resized_out_image == -9999] = np.nan

                        print("\n", q, d)
                        errs = get_closest_station_window(lon_s, lat_s, sm_s, sm_hru, resized_out_image)
                        all_err.extend(errs)
                        sm_hru = None
                    else:
                        continue
    print("Average error SM: ", np.mean(np.array(all_err)), " average std: ", np.std(np.array(all_err)), " on sample count: ", len(all_err))


def train_model(generator, num_epochs=200, batch_size=64, dirno=1, lr=0.0001, isModel1=False, start_in=0, end_in=500, isScienceGuided=False):
    my_dataloader = Dataloader()
    train_data, test_data = my_dataloader.load_all_datasets_in_memory(start_in, end_in)
    train_dl = get_input_batch(train_data, isTraining=True, batch_size=batch_size, isModel1=isModel1)
    test_dl = get_input_batch(test_data, isTraining=False, batch_size=64, isModel1=isModel1, shuffle=True)

    criterion1 = CustomLoss()
    criterion2 = CustomLoss_richards_psnr()
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    out_path = "/s/chopin/f/proj/fineET/sm_predictions/outputs_richards/" + str(dirno) + "/"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path + 'samples', exist_ok=True)
    all_loss_train, all_loss_test = [], []
    train_loss_file = out_path + 'train_loss.txt'
    test_loss_file = out_path + 'test_loss.txt'
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    for epoch in range(0, num_epochs + 1):
        print("Training model on epoch: ", epoch, "/", num_epochs, flush=True)
        generator = generator.to(device).train()
        epoch_loss = 0.0


        for (input_img, target_img, _, _) in train_dl:
            input_img = input_img.to(torch.float32)
            target_img = target_img.to(torch.float32)

            input_img = input_img.to(device)
            target_img = target_img.to(device)

            optimizer.zero_grad()
            outputs = generator(input_img)
            if not isScienceGuided:
                loss1, loss2 = criterion1(outputs, target_img) # mse, mae
                alpha = lr
            else:
                loss1, loss2, alpha = criterion2(predicted=outputs,target=target_img,land_cover=input_img[:, 18, :, :],
                                          input_image=input_img,epoch=epoch, num_epochs=num_epochs+1)

            loss1.backward()
            optimizer.step()
            epoch_loss += loss2.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}, New Learning Rate: {scheduler.get_last_lr()[0]}")

        # print("Training model with alpha: ", alpha)
        epoch_loss /= len(train_dl)
        # print(f"Epoch [{epoch}/{num_epochs}],Train Loss: {epoch_loss:.4f}")
        all_loss_train.append(epoch_loss)

        with open(train_loss_file, 'a') as f:
            f.write(f'{epoch_loss}\n')

        epoch_loss = 0.0
        generator = generator.eval()
        with torch.no_grad():
            for (input_img, target_img, _, _) in test_dl:
                input_img = input_img.to(torch.float32)
                target_img = target_img.to(torch.float32)

                input_img = input_img.to(device)
                target_img = target_img.to(device)

                outputs = generator(input_img)
                loss1, loss2 = criterion1(outputs[:, 0, :, :], target_img[:, 0, :, :])
                epoch_loss += loss2.item()

        epoch_loss /= len(test_dl)
        print(f"Epoch [{epoch}/{num_epochs}], Test Loss: {epoch_loss:.4f}")
        all_loss_test.append(epoch_loss)

        plt.plot(range(0, epoch + 1), all_loss_train, marker='o', color='#984ea3', markersize=2, linestyle='dotted',
                 label='Training Loss')
        plt.plot(range(0, epoch + 1), all_loss_test, marker='x', color='#999999', markersize=2, linestyle='solid',
                 label='Testing Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(out_path + 'training_test_loss.png')
        plt.close()

        with open(test_loss_file, 'a') as f:
            f.write(f'{epoch_loss}\n')

        if epoch % 10 == 0:
            print("Saving sample test data and loss")
            generator = generator.eval()
            with torch.no_grad():
                input_sample_daily, target_img, allquads, _ = next(iter(test_dl))
                input_sample_daily = input_sample_daily.to(torch.float32)
                input_sample_daily = input_sample_daily.to(device)
                output_sample = generator(input_sample_daily).cpu()

                l1_error = generator_loss(output_sample[0][0], target_img[0][0])
                PSNR_acc = generator_acc(output_sample[0][0], target_img[0][0])
                plot_targed_inferred_only_samples(output_sample[0], target_img[0], l1_error, epoch, out_path,
                                                  input_sample=input_sample_daily[0],
                                                  training=True, PSNR_acc=PSNR_acc)

                model_path = out_path + "model_weights.pth"
                x = torch.randn(1, 30, 64, 64).to('cpu')
                traced_cell = torch.jit.trace(generator.to('cpu'), (x))
            torch.jit.save(traced_cell, model_path)

    print("Training complete.")
    return test_data



def plot_distibution(sample_data, name = '', color='red', max_data = 99999.0):
    flattened_data = sample_data.view(-1).numpy()
    flattened_data = flattened_data[~np.isnan(flattened_data)]
    flattened_data = flattened_data[flattened_data <= max_data]
    plt.figure(figsize=(8, 6))
    parts = plt.violinplot(flattened_data, vert=True, showmeans=True,
                  showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    mean_value = np.mean(flattened_data)

    plt.title('Boxplot of predicted \n' + name + "\n Average Value: " + str(np.round(mean_value, 4)))
    plt.grid(True)
    plt.savefig("./" + name + ".png")
    plt.close()

def plot_scatter(pred_model1, pred_model2):
    pred_model1 = pred_model1[~torch.isnan(pred_model1)]
    pred_model2 = pred_model2[~torch.isnan(pred_model2)]
    x_values_1 = range(1, len(pred_model1) + 1)
    x_values_2 = range(1, len(pred_model2) + 1)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values_1, pred_model1, c='cyan', s=0.4, label='Using Soil Matric Potential')
    plt.scatter(x_values_2, pred_model2, c='orange', s=0.4, edgecolor='black', label='Using SE')

    plt.xlabel('Using Soil Matric Potential')
    plt.ylabel('Using SE')
    plt.title('Comparison of Predictions from Two Input Variables')
    plt.grid(True)

    plt.legend()
    plt.savefig("./scatter.png")
    plt.close()


if __name__ == '__main__':
    # 4 is huber based with some of mae in huber part
    # 5 is liner alpha
    # 6 is step alpha
    # 7 is mse based only + 80/20 spatial regions
    # 8 is mse based only
    # 9 again mae based
    # 10 is mae with lr scheduler

    # 11 is mae based only + 80/20 spatial regions
    # 12 is psi based only + 80/20 spatial regions + step
    # 13 is psi based only + use station sm in psi and vmc + 80/20 spatial regions + step
    start_in, end_in = 0, 206
    folder = 13
    print("Training Model on folder: ", folder, "start_in: ", start_in, "end_in: ", end_in)
    model = UnetGenerator(input_nc=26, output_nc=1, nf=64, use_dropout=False, final_out=1).to(device).float()
    # test_data = train_model(model, num_epochs=1501, batch_size=64, dirno=folder, lr=0.0001, isModel1=False,
    #             start_in=start_in, end_in=end_in, isScienceGuided=True)
    test_data = None
    loss_tot, psnr_tot = perform_inferences(test_data=test_data, folder=folder, one_per_quad=True, isModel1=False, start_in=start_in,
                                            end_in=end_in)
    # perform_inferences_stations(folder, test_data=test_data, start_in=start_in, end_in=end_in)

    # print("Finished training folder mae_only station: {} -- Final testing loss: {}, Testing PSNR: {}".format(folder, loss_tot, psnr_tot))

    # my_loader = Dataloader()
    # train_data, _ = my_loader.load_all_datasets_in_memory(0, 5)
    # train_dataset = QuadhashDataset(train_data, training=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=3)

