import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import QuadhashDataset, QuadhashDatasetModel1, Dataloader
import torch
import torch.nn as nn
import gdal
np.set_printoptions(suppress=True)
import cv2
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OUT_PATH = "/s/lovelace/f/nobackup/shrideep/sustain"
OUT_PATH = "/s/chopin/f/proj/fineET/"

def get_input_batch(data, isTraining=True, batch_size=64, isModel1=False):
    batch_size = batch_size

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

    dataloader_new = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    return dataloader_new

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

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        mask = (target != -1).float()
        loss = self.criterion(input * mask, target * mask)
        return loss

class CustomLoss_science_psnr(nn.Module):
    def __init__(self, mae_weight=1.0, psnr_weight=0.6, lc_weight=0.4, ssim_weight=0.5, rule_sm_weight=10):
        super(CustomLoss_science_psnr, self).__init__()
        self.mae_criterion = nn.L1Loss()
        self.mae_weight = mae_weight
        self.psnr_penalty_weight = psnr_weight
        self.land_cover_penalty_weight = lc_weight
        self.ssim_loss_weight = ssim_weight
        self.rule_sm_weight = rule_sm_weight

    def psnr_loss(self, input, target, mask):
        input = input * mask
        target = target * mask
        mse = nn.functional.mse_loss(input, target)
        psnr = 20 * torch.log10(1 / torch.sqrt(mse))
        return 52-psnr

    def ssim_based_loss(self, input, target, mask):
        input = input * mask
        target = target * mask
        return 1 - pytorch_ssim.ssim(input, target)

    def rule_wrt_saturated_unsaturated(self, generated_image, mask, input_img):
        generated_image = generated_image * mask
        theta_r = torch.unsqueeze(input_img[:, -11, :, :], 1)
        theta_s = torch.unsqueeze(input_img[:, -12, :, :], 1)
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

    def forward(self, input, target, land_cover=None, input_image=None):
        mask = (target != -1).float()
        psnr_loss_val = self.psnr_loss(input, target, mask)
        mae_loss = self.mae_criterion(input * mask, target * mask)
        rule_loss = self.rule_wrt_saturated_unsaturated(input, mask, input_image)
        return self.mae_weight * mae_loss + self.psnr_penalty_weight * psnr_loss_val + self.rule_sm_weight * rule_loss, mae_loss

def plot_targed_inferred_only_samples(output_sample, target_img, l1_error, epoch, out_path, input_sample, training=True, PSNR_acc=0):
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 4, 1)
    plt.title("Land Cover")
    input_sample1 = input_sample[18, :, :].cpu()
    input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.imshow(input_sample1)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 4, 2)
    plt.title("Saturated SWC")
    input_sample1 = input_sample[23, :, :].cpu()
    input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.imshow(input_sample1, cmap='cividis_r')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 4, 3)
    plt.title("Residual SWC")
    input_sample1 = input_sample[24, :, :].cpu()
    input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.imshow(input_sample1, cmap='cividis_r')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 4, 4)
    plt.title("Polaris - Alpha")
    input_sample1 = input_sample[-4, :, :].cpu()
    # input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.imshow(input_sample1)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 4, 5)
    plt.title("Polaris - n")
    input_sample1 = input_sample[-7, :, :].cpu()
    # input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.imshow(input_sample1)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 4, 6)
    input_sample1 = input_sample[0, :, :].cpu()
    input_sample1 = np.ma.masked_equal(input_sample1.numpy(), -1)
    plt.title("SMAP: " + str(input_sample1[0][0]))
    plt.imshow(input_sample1, cmap='cividis_r')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 4, 7)
    target_img = target_img.numpy()
    target_img[target_img < 0] = np.nan

    output_sample = output_sample.numpy()

    sm_vmin = np.nanmin((output_sample, target_img))
    sm_vmax = np.max((output_sample, target_img))

    plt.title("Target Image")
    target_img = np.ma.masked_equal(target_img.transpose(1, 2, 0), -1)
    plt.imshow(target_img, cmap='cividis_r', vmin=sm_vmin, vmax=sm_vmax)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 4, 8)
    output_sample = np.ma.masked_equal(output_sample.transpose(1, 2, 0), -1)
    plt.title("Generated Image \n L1 loss: " + str(round(l1_error.item(), 2)) + " PSNR: " + str(
        round(PSNR_acc.item(), 1)) + "dB")
    plt.imshow(output_sample, cmap='cividis_r', vmin=sm_vmin, vmax=sm_vmax)
    plt.axis('off')
    plt.colorbar()

    if training:
        plt.savefig(out_path + 'samples/' + str(epoch) + '.png')
    else:
        os.makedirs(out_path + 'testing', exist_ok=True)
        plt.savefig(out_path + 'testing/' + str(epoch) + '.png')
    plt.close()

def train_model(generator, num_epochs=200, batch_size=64, dirno=1, lr=0.0001, isModel1=False, start_in=0, end_in=500):
    my_dataloader = Dataloader()
    train_data, test_data = my_dataloader.load_all_datasets_in_memory(start_in, end_in)
    train_dl = get_input_batch(train_data, isTraining=True, batch_size=batch_size, isModel1=isModel1)
    test_dl = get_input_batch(test_data, isTraining=False, batch_size=64, isModel1=isModel1)

    criterion1 = CustomLoss()
    criterion2 = CustomLoss_science_psnr()

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    out_path = OUT_PATH + "/sm_predictions/outputs/" + str(dirno) + "/"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path + 'samples', exist_ok=True)
    all_loss_train, all_loss_test = [], []
    train_loss_file = out_path + 'train_loss.txt'
    test_loss_file = out_path + 'test_loss.txt'
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
            loss1, loss2 = criterion2(outputs, target_img, input_img[:, 18, :, :], input_img)

            loss1.backward()
            optimizer.step()
            epoch_loss += loss2.item()

        epoch_loss /= len(train_dl)
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
                loss = criterion1(outputs, target_img)
                epoch_loss += loss.item()

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

        if epoch % 15 == 0:
            print("Saving sample test data and loss")
            generator = generator.eval()
            with torch.no_grad():
                input_sample_daily, target_img, allquads, _ = next(iter(test_dl))
                input_sample_daily = input_sample_daily.to(torch.float32)
                input_sample_daily = input_sample_daily.to(device)
                output_sample = generator(input_sample_daily).cpu()

                l1_error = generator_loss(output_sample[0], target_img[0])
                PSNR_acc = generator_acc(output_sample[0], target_img[0])
                plot_targed_inferred_only_samples(output_sample[0], target_img[0], l1_error, epoch, out_path, input_sample=input_sample_daily[0],
                                     training=True, PSNR_acc=PSNR_acc)

                model_path = out_path + "model_weights.pth"
                x = torch.randn(1, 35, 64, 64).to('cpu')
                traced_cell = torch.jit.trace(generator.to('cpu'), (x))
            torch.jit.save(traced_cell, model_path)

    print("Training complete.")

def load_model_weights(folder):
    out_path = OUT_PATH + "/sm_predictions/outputs/" + str(folder) + "/"
    model_path = out_path + "model_weights.pth"
    loaded_model = torch.jit.load(model_path)
    return loaded_model

def perform_inferences(folder, one_per_quad=False, isModel1=False, start_in=0, end_in=500):
    model2 = load_model_weights(folder)
    model2 = model2.to(device).float()
    model2 = model2.eval()
    my_dataloader_2 = Dataloader()

    print("Performing inference on folder: ", folder)
    batch_size = 128
    out_path = OUT_PATH + "/sm_predictions/outputs/" + str(folder) + "/"
    os.makedirs(out_path, exist_ok=True)
    if isModel1:
        test_dataset = QuadhashDatasetModel1(training=False, one_per_quad=one_per_quad, corrected_hru=False)
    else:
        _, test_data = my_dataloader_2.load_all_datasets_in_memory(start_in=start_in, end_in=end_in)
        test_dataset = QuadhashDataset(test_data, training=False)

    total_loss,total_psnr =[], []
    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    count = 0
    with torch.no_grad():
        for (input_sample, target_img, allquads, months) in dataloader_new:
            input_sample = input_sample.to(torch.float32).to(device)
            output_sample = model2(input_sample).cpu()
            target_img = target_img.cpu()

            for i in range(input_sample.shape[0]):
                count += 1
                l1_error = generator_loss(output_sample[i], target_img[i])
                psnr_acc = generator_acc(output_sample[i], target_img[i])
                total_loss.append(round(l1_error.item(), 4))
                psna = psnr_acc.item()
                if psna == float('inf'):
                    psna = 48
                total_psnr.append(round(psna, 1))
                if count <= 50:
                    plot_targed_inferred_only_samples(output_sample[i], target_img[i], l1_error, count, out_path, input_sample[i], training=False, PSNR_acc=psnr_acc)
    loss_tot = np.round(np.mean(np.array(total_loss)), 3)
    psnr_tot = np.round(np.nanmean(np.array(total_psnr)), 5)
    print("Average loss on test data: ", loss_tot, "\nAverage PSNR accuracy: ", psnr_tot, flush=True)
    return loss_tot, psnr_tot

def calculate_no_of_parameters(model):
    numel_list = [p.numel() for p in model.parameters() if p.requires_grad == True]
    print("Total number of trainable parameters in the model: ", sum(numel_list))
    return sum(numel_list)

import sys
if __name__ == '__main__':
    min_folder = int(sys.argv[1])
    # min_folder = int(input())
    # min_folder = 71 and 81

    loss_tots, psnr_tots = [], []
    total = math.ceil(len(os.listdir("/s/lovelace/f/nobackup/shrideep/sustain/sm_predictions/input_datasets/hru/split_14/"))/1000)
    print("Total models need to be trained: ", total)
    max_folder = min_folder + 10
    
    #model = load_model_weights(1000).to(device).float()
    #train_model(model, num_epochs=2011, batch_size=64, dirno=1, lr=0.0001, isModel1=False, start_in=0, end_in=4)
    #sys.exit()
    
    for folder in range(1, total+1):
        start_in = (folder - 1) * 1000
        end_in = folder * 1000
        if folder >= min_folder and folder < max_folder:
            print("Training Model on folder: ", folder, "start_in: ", start_in,"end_in: ", end_in)
            model = load_model_weights(1000).to(device).float()
            train_model(model, num_epochs=2011, batch_size=64, dirno=folder, lr=0.0001, isModel1=False, start_in=start_in, end_in=end_in)

            loss_tot, psnr_tot = perform_inferences(folder=folder, one_per_quad=True, isModel1=False, start_in=start_in, end_in=end_in)
            loss_tots.append(loss_tot)
            psnr_tots.append(psnr_tot)
            print("Finished training on folder: ", folder)
            
            cmd = ['setfacl', '-Rm', 'g:sustain:rwx', '/s/chopin/f/proj/fineET/sm_predictions/outputs/' + str(folder)]
            subprocess.call(cmd)

    print("Average PSNRs for each directory {}".format(psnr_tots))
    print("Average MSEs for each directory {}".format(loss_tots))
