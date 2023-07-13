# input LR images，output SR results
import os.path
import torch
from PIL import Image
import torchvision.transforms as transform
import Experiments.experiment1.utils_image as util_
from archs.NetG import TD_IDES
from archs.ChannelSplit_arch4_2 import TDSR

totensor = transform.ToTensor()

ori_path = '../../../Dataset/Experiments/experiment1/BSD100/SVLR'       # LR images
E_path = '../../../Dataset/Experiments/experiment1/BSD100/results'      # SR results

checkpoint = torch.load('../../pretrain_weight/TDSR.pth.tar')
net_sr = TDSR().cuda()
device_ids = [0, 1]
net_sr = torch.nn.DataParallel(net_sr, device_ids=device_ids)
net_sr.load_state_dict(checkpoint['state_dict'])

netG = TD_IDES().cuda()
pre_file_2 = torch.load('../../pretrain_weight/TD_IDES.pth')
netG.load_state_dict(pre_file_2)
epoch = 0

with torch.no_grad():
        for img in util_.get_image_paths(ori_path):
                print(epoch)
                epoch += 1
                torch.cuda.empty_cache()
                img_name, ext = os.path.splitext(os.path.basename(img))

                torch.cuda.empty_cache()
                lr_img = totensor(Image.open(img)).unsqueeze(0).cuda()

                degfea_out_, deg_fea_, degfea_in_ = netG(lr_img.cuda())
                syn_hr = net_sr(lr_img.cuda(), degfea_out_)

                syn_hr = util_.tensor2uint(syn_hr)
                util_.imsave(syn_hr, os.path.join(E_path, img_name + '_SR_' + 'TDSR' + '.png'))

