import torch
import torch.optim as optim
import lpips
from torch.utils.data import DataLoader
from tqdm import tqdm
from DatasetFromFolder import DatasetFromFolder_train, DatasetFromFolder_test
import util
import torch.nn as nn
import torchvision.transforms as transform
from basicsr.metrics import calculate_niqe
import Experiments.experiment1.utils_image as util_
from NetG import TD_IDES
from ChannelSplit_arch4_2 import TDSR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
totensor = transform.ToTensor()

zoom_factor = 4
BATCH_SIZE = 32
BATCH_SIZE_ = 2
NUM_WORKERS = 8
NB_EPOCHS = 800

train_hr = '../Dataset/train/DF2K_OST/'
test_hr = '../Dataset/test/DIV2K/HR/'
test_lr = '../Dataset/test/DIV2K/LR/'

trainset = DatasetFromFolder_train(train_hr)
testset = DatasetFromFolder_test(test_hr, test_lr, zoom_factor)

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE_, shuffle=False, num_workers=NUM_WORKERS)

net = TDSR().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

netG = TD_IDES().to(device)
pre_file = torch.load('TD_IDES.pth')
netG.load_state_dict(pre_file)

rec_loss = nn.L1Loss()
lpips_m = lpips.LPIPS().to(device)

for name, parameter in netG.named_parameters():
    parameter.requires_grad = False

for epoch in range(NB_EPOCHS):
    # train
    net.train()
    train_bar = tqdm(trainloader)
    running_results = {'batch_sizes': 0, 'l1_loss': 0, 'PSNR': 0, 'NIQE': 0, 'LPIPS': 0}
    for data, target in train_bar:
        batch_size = BATCH_SIZE
        running_results['batch_sizes'] += batch_size
        avg_psnr_train = 0
        avg_niqe_train = 0
        avg_lpips_train = 0

        hr_img = target.cuda()
        lr_img = data.cuda()

        degfea_out_, deg_fea_, degfea_in_ = netG(lr_img)
        fake_img_up = net(lr_img, degfea_out_)

        l1_loss = rec_loss(fake_img_up, hr_img)
        
        l1_loss.backward()
        optimizer.step()
        net.zero_grad()

        syn_hr_y = util_.rgb2ycbcr(util_.tensor2uint(fake_img_up).transpose(0, 2, 3, 1))
        hr_img_y = util_.rgb2ycbcr(util_.tensor2uint(hr_img).transpose(0, 2, 3, 1))
        b, c, h, w = hr_img.size()

        for i in range(0, b):
            psnr_train_0 = util.calculate_psnr_train(syn_hr_y[i], hr_img_y[i])
            avg_psnr_train += psnr_train_0

            # HR的PSNR
        for i in range(0, b):
            niqe_0 = calculate_niqe(syn_hr_y[i], 0, input_order='HW')
            avg_niqe_train += niqe_0

        with torch.no_grad():
            for i in range(0, b):
                lpips_0 = lpips_m(totensor(hr_img_y[i]).cuda(), totensor(syn_hr_y[i]).cuda()).item()
                avg_lpips_train += lpips_0

        running_results['PSNR'] += avg_psnr_train * batch_size
        running_results['NIQE'] += avg_niqe_train * batch_size
        running_results['LPIPS'] += avg_lpips_train * batch_size

        train_bar.set_description(
            desc='[%d/%d] PSNR: %.4f, NIQE: %.4f, LPIPS: %.4f  ' % (
                epoch, NB_EPOCHS,
                running_results['PSNR'] / (batch_size * running_results['batch_sizes']),
                running_results['NIQE'] / (batch_size * running_results['batch_sizes']),
                running_results['LPIPS'] / (batch_size * running_results['batch_sizes'])))

    # val
    avg_psnr_test = 0
    avg_niqe_test = 0
    avg_lpips_test = 0
    with torch.no_grad():
        for batch in testloader:
            lr_img, hr_img = batch[0].to(device), batch[1].to(device)

            degfea_out_, deg_fea_, degfea_in_ = netG(lr_img)
            fake_img = net(lr_img, degfea_out_)

            syn_hr_y = util_.rgb2ycbcr(util_.tensor2uint(fake_img).transpose(0, 2, 3, 1))
            hr_img_y = util_.rgb2ycbcr(util_.tensor2uint(hr_img).transpose(0, 2, 3, 1))
            b, c, h, w = hr_img.size()

            for i in range(0, b):
                psnr_train_0 = util.calculate_psnr_train(syn_hr_y[i], hr_img_y[i])
                avg_psnr_test += psnr_train_0

            for i in range(0, b):
                niqe_0 = calculate_niqe(syn_hr_y[i], 0, input_order='HW')
                avg_niqe_test += niqe_0

            with torch.no_grad():
                for i in range(0, b):
                    lpips_0 = lpips_m(totensor(hr_img_y[i]).cuda(), totensor(syn_hr_y[i]).cuda()).item()
                    avg_lpips_test += lpips_0

            running_results['NIQE'] += avg_niqe_test
            running_results['LPIPS'] += avg_lpips_test
            running_results['PSNR'] += avg_psnr_test

        print(f"Average test PSNR: {running_results['PSNR'] / (BATCH_SIZE_ * len(testloader))}.")
        print(f"Average test NIQE: {running_results['NIQE'] / (BATCH_SIZE_ * len(testloader))}.")
        print(f"Average test LPIPS: {running_results['LPIPS'] / (BATCH_SIZE_ * len(testloader))}")

torch.save(net.state_dict(), 'TDSR_sr.pth')
