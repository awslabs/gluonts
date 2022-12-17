

import os
import time
from torch import optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import warnings
from data import *
from utils import *

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

plt.switch_backend('agg')

class Exp_Basic(object):
    def __init__(self, args, model):
        self.args = args
        self.device = self._acquire_device()
        self.model = model

    def _acquire_device(self):
       
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) 
        device = torch.device('cuda:{}'.format(self.args.gpu))
        print('Use GPU: cuda:{}'.format(self.args.gpu))
        return device

    def _get_data(self, *args, **kwargs):
        pass

    def vali(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass
    
class Exp_Main(Exp_Basic):
    def __init__(self, args, model):
        super(Exp_Main, self).__init__(args, model)
        self.output_attention = args.output_attention
        self.model_type = args.version
        self.model = model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, mae_criterion):
        total_loss = []
        total_mae = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
               
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                elif self.args.output_stl:
                    outputs, _,_,_,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                mae_loss = mae_criterion(pred, true)

                total_loss.append(loss)
                total_mae.append(mae_loss)
        total_loss = np.average(total_loss)
        total_mae = np.average(total_mae)
        self.model.train()
        return total_loss, total_mae

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join('./checkpoints/')
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, des=self.args.des, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        mae_criterion = nn.L1Loss()
        
        for epoch in range(1000):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.output_attention:
                    outputs, A = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                elif self.args.output_stl:
                    outputs, _,_,_,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
       
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((1000 - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
               
                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae = self.vali(vali_data, vali_loader, criterion, mae_criterion)
            test_loss, test_mae = self.vali(test_data, test_loader, criterion, mae_criterion)
            f = open('./log/' + self.args.des + '.txt', 'a')
            f.write('\n')
            f.write('epoch:{}, train_loss:{}, vali_loss:{}, test_mse:{}, test_mae:{}'.format(epoch+1, train_loss, vali_loss, test_loss, test_mae))
            f.close()
        
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test MSE: {4:.7f} Test MAE: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, test_mae))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if self.args.adjust:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + self.args.des + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.args.des + 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + self.args.des + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
               
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                elif self.args.output_stl:
                    outputs, tr_ori, se_ori, tr_pred, se_pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    tr_ori = tr_ori.detach().cpu().numpy()
                    se_ori = se_ori.detach().cpu().numpy()
                    tr_pred = tr_pred.detach().cpu().numpy()
                    se_pred = se_pred.detach().cpu().numpy()
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
           
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                ch = -1
                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, ch], true[0, :, ch]), axis=0)
                    pd = pred[0, :, ch]
                    visual(gt, pd, self.args.seq_len, os.path.join(folder_path, str(i) + '.pdf'))     
                    if self.args.output_stl:
                        tr_ori = tr_ori[0, :, -1]
                        se_ori = se_ori[0, :, -1]
                        tr_pred = tr_pred[0, :, -1]
                        se_pred = se_pred[0, :, -1]
                        visual_stl(gt, pd, tr_ori, se_ori, tr_pred, se_pred, self.args.seq_len, os.path.join(folder_path, str(i) + 'stl.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(self.args.des)
        f.write('\n')
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return