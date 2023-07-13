import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from DatasetFromFolder import DatasetFromFolder_train, DatasetFromFolder_test
from New_idea.DW_degnet import TD_IDET
from Affine.archs.NetG import TD_IDES
import torch.nn.functional as F
from torch.optim import lr_scheduler
from Affine.archs.common import transfer_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

zoom_factor = 4
BATCH_SIZE = 32
BATCH_SIZE_ = 10
NUM_WORKERS = 8
NB_EPOCHS = 200

train_hr = '../../Dataset/train/DF2K_OST/HR/'
test_hr = '../../Dataset/test/DIV2K/HR/'
test_lr = '../../Dataset/test/DIV2K/LR/'

trainset = DatasetFromFolder_train(train_hr)
testset = DatasetFromFolder_test(test_hr, test_lr, zoom_factor)

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE_, shuffle=False, num_workers=NUM_WORKERS)

net_deg = TD_IDET().to(device)
pre_file = torch.load('../pretrain_weight/3D_IDET.pth')
net_deg.load_state_dict(pre_file)

#----------------Ours-------------------
netG = TD_IDES().to(device)
netG = transfer_model('../pretrain_weight/3D_IDET.pth', netG)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002)
exp_lr_scheduler = lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.5)

rec_loss = nn.L1Loss()

for name, parameter in net_deg.named_parameters():
    parameter.requires_grad = False

for epoch in range(NB_EPOCHS):
    # train
    netG.train()
    epoch_loss_train = 0
    avg_psnr_train = 0
    avg_ssim_train = 0
    avg_lpips_train = 0
    train_bar = tqdm(trainloader)
    running_results = {'batch_sizes': 0, 'kl_loss': 0, 'code_loss': 0, 'code_loss2': 0, 'code_loss3': 0, 'total_loss': 0, 'PSNR': 0, 'SSIM': 0,
                       'LPIPS': 0, 'PSNR_LR': 0, 'd_loss': 0, 'g_loss': 0}
    for data, target in train_bar:
        batch_size = BATCH_SIZE
        running_results['batch_sizes'] += batch_size

        hr_img = target.cuda()
        lr_img = data.cuda()

        syn_lr, deg_fea, degfea_in, degfea_out = net_deg(hr_img, lr_img)
        degfea_out_, deg_fea_, degfea_in_ = netG(lr_img)

        net_deg.zero_grad()
        netG.zero_grad()

        # ----------------calculate loss------------------------------------------
        l_rec = rec_loss(degfea_out_, lr_img-F.interpolate(hr_img, scale_factor=1/4, mode='bilinear'))
        l_rec2 = rec_loss(degfea_in_, degfea_in)
        l_rec3 = 0
        for i in range(0, 23):
            l_rec3_ = ((23-i)/276)*rec_loss(deg_fea_[i], deg_fea[i])
            l_rec3 += l_rec3_

        l_total = 5*l_rec + l_rec2 + l_rec3

        l_total.backward()
        optimizerG.step()
        running_results['code_loss'] += l_rec.item() * batch_size
        running_results['code_loss2'] += l_rec2.item() * batch_size
        running_results['code_loss3'] += l_rec3.item() * batch_size
        running_results['total_loss'] += l_total.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_total: %.4f, Loss_code: %.4f, Loss_code2: %.4f, Loss_code3: %.4f' % (
            epoch+1, NB_EPOCHS, running_results['total_loss'] / running_results['batch_sizes'],
            running_results['code_loss'] / running_results['batch_sizes'],
            running_results['code_loss2'] / running_results['batch_sizes'],
            running_results['code_loss3'] / running_results['batch_sizes'],
            )
                            )

    # val
    avg_psnr_test = 0
    avg_ssim_test = 0
    avg_lpips_test = 0
    l = 0
    with torch.no_grad():
        for batch in testloader:
            lr_img, hr_img = batch[0].to(device), batch[1].to(device)
            degfea_out_, deg_fea_, degfea_in_ = netG(lr_img)

            l_code = rec_loss(degfea_out_, lr_img-F.interpolate(hr_img, scale_factor=1/4, mode='bilinear'))
            l += l_code.item()
        l = l / 50
        print(l)
    exp_lr_scheduler.step()

torch.save(netG.state_dict(), '../pretrain_weight/TD_IDES.pth')
