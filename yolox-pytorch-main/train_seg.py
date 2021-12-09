# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch import nn
from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def transfer_model(pretrained_file, model):
    pretrained_dict = torch.load(pretrained_file, 'cuda')  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrained_dict, model_dict):
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print('pretrained length:', len(pretrained_dict))
    print('model length:', len(model_dict))
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # print(model_dict[k])
            # state_dict.setdefault(k, v)
            # print(k)
            if np.shape(v) == np.shape(model_dict[k]):
                state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros(in_channels, out_channels,kernel_size, kernel_size)
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i,j,:,:] = filt
    return weight


def getsegloss(label,output):
    # print(output.shape,label.shape)
    return nn.CrossEntropyLoss(reduction='mean')(output,label)
    # return F.cross_entropy(output, label, reduction='none').mean(1).mean(1)

class Dataseg(Dataset):
    def __init__(self):
        super(Dataseg, self).__init__()
        self.json_file = 'D:/BaiduNetdiskDownload/train/landslide_train_google_20191115.json'
        self.img_file = 'D:/BaiduNetdiskDownload/train/JPEGImages/'
        self.coco = COCO(self.json_file)
        self.catIds = self.coco.getCatIds(catNms=['str'])  # 获取指定类别 id
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)  # 获取图片i

    def __len__(self):
        return len(os.listdir(self.img_file))
    def __getitem__(self, item):
        img = self.coco.loadImgs(self.imgIds[item - 1])[0]  # 加载图片,loadImgs() 返回的是只有一个内嵌字典元素的list, 使用[0]来访问这个元素
        image = plt.imread(self.img_file + img['file_name'])

        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        # image = np.array(Image.fromarray(image).resize((640, 640),Image.ANTIALIAS))
        mask = np.zeros((int(img['height']), int(img['width'])))
        for i in range(len(anns)):
            mask += self.coco.annToMask(anns[i])

        mask = np.clip(mask,0,1)
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(image)
        # plt.subplot(122)
        # plt.imshow(mask)
        # plt.show()
        return torch.cuda.FloatTensor(image[680:680+640,680:680+640,:]).permute(2,0,1),torch.cuda.LongTensor(mask[680:680+640,680:680+640])

if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # --------------------------------------------------------#
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # --------------------------------------------------------#
    classes_path = 'model_data/new_classes.txt'
    # -------------------------------------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   预训练权重对于99%的情况都必须要用，不用的话权值太过随机，特征提取效果不明显
    #   网络训练的结果也不会好，数据的预训练权重对不同数据集是通用的，因为特征是通用的
    # ------------------------------------------------------------------------------------#
    # model_path      = 'logs/ep100-loss2.860-val_loss2.638.pth'
    model_path = 'model_data/val_loss2.824_fpn_fssd.pth'
    # ---------------------------------------------------------------------#
    #   所使用的YoloX的版本。s、m、l、x
    # ---------------------------------------------------------------------#
    phi = 's'
    # ------------------------------------------------------#
    #   输入的shape大小，一定要是32的倍数
    # ------------------------------------------------------#
    input_shape = [640, 640]
    # ------------------------------------------------------------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False 
    #   YOLOX作者强调要在训练结束前的N个epoch关掉Mosaic。因为Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #   并且Mosaic大量的crop操作会带来很多不准确的标注框，本代码自动会在前90%个epoch使用mosaic，后面不使用。
    #   Cosine_scheduler 余弦退火学习率 True or False
    # ------------------------------------------------------------------------------------------------------------#
    mosaic = False
    Cosine_scheduler = False

    # ----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    # ----------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 70
    Freeze_batch_size = 6
    Freeze_lr = 1e-3
    # ----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    # ----------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 3
    Unfreeze_lr = 1e-6
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = True
    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0  
    # ------------------------------------------------------#
    num_workers = 4
    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    model = YoloBody(num_classes, phi)
    weights_init(model)
    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    # ------------------------------------------------------#
    print('Load weights {}.'.format(model_path))
    # device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dict      = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location = device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    model = transfer_model(model_path, model)
    model = model.backbone

    class seg_head(nn.Module):
        def __init__(self):
            super(seg_head, self).__init__()
            self.conv1 = nn.Conv2d(896,128,3,1,1)
            self.tranconv1 = nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1)
            self.tranconv1.weight.data.copy_(bilinear_kernel(128,64,4))
            self.tranconv2 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
            self.tranconv2.weight.data.copy_(bilinear_kernel(64, 16, 4))
            self.tranconv3 = nn.ConvTranspose2d(16, 2, kernel_size=4, stride=2, padding=1)
            self.tranconv3.weight.data.copy_(bilinear_kernel(16, 2, 4))
            self.up1 = nn.Upsample(scale_factor=2)
            self.up2 = nn.Upsample(scale_factor=4)
        def forward(self,x):
            x1,x2,x3 = x[0],x[1],x[2]
            x = torch.cat((self.up2(x3),self.up1(x2),x1),dim=1)
            x = self.conv1(x)
            x = self.tranconv1(x)
            x = self.tranconv2(x)
            x = self.tranconv3(x)
            # x = self.tranconv4(x)
            # return x
            return F.softmax(x,dim=1)

    class Segmodel(nn.Module):
        def __init__(self,backbone):
            super(Segmodel, self).__init__()
            self.backbone = backbone
            self.head = seg_head()

        def forward(self,x):
            x = self.backbone(x)
            return self.head(x)

    seg = Segmodel(model).cuda()
    # input = model(torch.ones(1,3,640,640))
    # plt.imshow(seg(input).detach().numpy()[0,0,:,:])
    # plt.show()
    # print(getsegloss(torch.LongTensor(torch.ones(1,640,640).numpy()),seg(input)))

    dataset = Dataseg()
    dataset = DataLoader(dataset,batch_size=2,shuffle=True,drop_last=True)
    for param in model.backbone.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(seg.parameters(),lr=1e-3,weight_decay=1e-3)
    for i in range(50):
        for image, mask in dataset:
            # print(set(mask.view(-1,).cpu().numpy()))
            output = seg(image)
            loss = getsegloss(mask, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())
    for param in model.backbone.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(seg.parameters(), lr=1e-4, weight_decay=1e-3)
    for i in range(50):
        for image, mask in dataset:
            # print(set(mask.view(-1,).cpu().numpy()))
            output = seg(image)
            loss = getsegloss(mask, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())
    state_dict = seg.state_dict()
    torch.save(state_dict,'seg.pth')

