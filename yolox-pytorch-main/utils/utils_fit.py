import torch
from tqdm import tqdm
import numpy as np
from utils.utils import get_lr


x_ema_sig = np.arange(5,7.5,0.025)
x_ema_act = np.arange(500,1300,8)
def sigmoid(x):
    s1 = np.exp(x)-np.exp(-x)
    s2 = np.exp(x)+np.exp(-x)
    return s1/s2

def arctan(x):
    return np.arctan(x)/np.pi*2
y_ema_sig = [sigmoid(i) for i in x_ema_sig]
y_ema_act = [arctan(i) for i in x_ema_act]

def ema(dict_old,dict_new,n=0.9990):
    dict_ = {k:None for k in dict_old.keys()}
    for i in dict_.keys():
        dict_[i] = (1-n)*dict_old[i] + n*dict_new[i]
    return dict_

def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    loss        = 0
    val_loss    = 0

    model_dict_old = model.state_dict()
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)

            #----------------------#
            #   计算损失
            #----------------------#
            loss_value = yolo_loss(outputs, targets)

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)

                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = yolo_loss(outputs, targets)

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    model_dict_new = model.state_dict()
    model_dict_new.update(ema(model_dict_old, model_dict_new, y_ema_sig[epoch]))
    #model_dict_new.update(ema(model_dict_old, model_dict_new, y_ema_act[epoch]))
    #model_dict_new.update(ema(model_dict_old, model_dict_new, n=0.9998))
    model.load_state_dict(model_dict_new)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
