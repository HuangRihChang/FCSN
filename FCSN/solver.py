import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from tqdm.notebook import tqdm, trange
import h5py
from prettytable import PrettyTable

from fcsn import FCSN
import eval

def get_device():
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")
    return device



def get_device():
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")
    return device

class Solver(object):
    """Class that Builds, Trains FCSN model"""

    def __init__(self, config=None, train_loader=None, test_dataset=None, 
                 device=None, optimizer="adam", loss_reduction=None,
                 eval_method="max"
                 ):
        self.config = config
        self.train_loader = train_loader
        self.test_dataset = test_dataset
        self.device = device
        self.loss_reduction = loss_reduction
        self.eval_method = eval_method
        # model
        self.model = FCSN(self.config.n_class)

        # optimizer
        if self.config.mode == 'train':
            if optimizer.lower() == "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
            else:
                self.optimizer = optim.SGD(self.model.parameters(), 
                                            lr=config.lr,
                                            momentum=self.config.momentum)
            self.model.train()

        self.model.to(self.device)

        if not os.path.exists(self.config.score_dir):
            os.mkdir(self.config.score_dir)

        if not os.path.exists(self.config.save_dir):
            os.mkdir(self.config.save_dir)

        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

    @staticmethod
    def sum_loss(pred_score, gt_labels, weight=None, loss_reduction=None):
        batch, seq = gt_labels.shape
        _, _, n_class = pred_score.shape
        pred_score = pred_score[:,:seq,:]
        gt_labels = gt_labels.reshape(-1)
        log_p = torch.log_softmax(pred_score, dim=-1).reshape(-1, n_class)
        criterion = torch.nn.NLLLoss(weight, reduction=loss_reduction)
        loss = criterion(log_p, gt_labels)
        return loss, log_p.shape

    def to_device(self, frame_features, label):
        return frame_features.to(self.device), label.to(self.device)

    def train(self, writer, train_batch=20, use_weight=False):
        assert train_batch <= len(self.train_loader), f"train_batch can not great than {len(self.train_val_loader)}"
        self.model.train()
        t = trange(self.config.n_epochs, desc='Epoch')
        mean_loss, eval_mean, mean_train_f1 = 0.0, 0.0, 0.0
        for epoch_i in t:
            self.model.train()
            sum_loss_history = []
            for i, (feature, label, _) in enumerate(tqdm(self.train_loader, desc='Batch', leave=False)):
                # [batch_size, 1024, seq_len]
                feature, label = self.to_device(feature,label)

                label =label.to(torch.long)
                feature = feature.permute(0,2,1)
                feature.requires_grad_()

                # ---- Train ---- #
                pred_score = self.model(feature).permute(0,2,1)

                if use_weight:
                    label_1 = label.sum()
                    label_0 = label.shape[1] - label_1
                    weight = torch.tensor([label_1, label_0], dtype=torch.float).to(self.device)
                else:
                    weight = None

                loss, predict_shape = self.sum_loss(pred_score, label, weight, self.loss_reduction)
                loss.backward()

                sum_loss_history.append(loss)
                mean_loss = torch.stack(sum_loss_history).mean().item()
                t.set_postfix(mean_loss=mean_loss, eval_mean_f1=eval_mean, mean_train_f1=mean_train_f1, prediction_shape = predict_shape)
                if i%train_batch or i+1 == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            mean_train_f1 = self.evaluate_train()        
            eval_mean = self.evaluate()
            writer.add_scalar('Loss', mean_loss, epoch_i)
            writer.add_scalar('F1 eval', eval_mean, epoch_i)
            writer.add_scalar('F1 train', mean_train_f1, epoch_i)
            writer.close()
            t.set_postfix(mean_loss=mean_loss, eval_mean_f1=eval_mean, mean_train_f1=mean_train_f1)

            if (epoch_i+1) % 30 == 0:
                continue
            #     ckpt_path = self.config.save_dir + '/epoch-{}.pt]'.format(epoch_i)
            #     tqdm.write('Save parameters at {}'.format(ckpt_path))
            #     torch.save(self.model.state_dict(), ckpt_path)

    def evaluate_train(self):
        self.model.eval()
        eval_arr = []
        with h5py.File(self.config.data_path) as data_file:
            for feature, label, idx in tqdm(self.train_loader, desc='Calculing F1', leave=False):
                idx = str(idx[0].split("_")[-1])
                feature, label = self.to_device(feature,label)
                label =label.to(torch.long)
                feature = feature.permute(0,2,1)
                pred_score = self.model(feature).permute(0,2,1).squeeze(0)
                pred_score = torch.softmax(pred_score, dim=-1)[:,1]
                video_info = data_file["video_"+str(idx)]                
                n_frames = video_info['length'][()]
                sb = video_info['change_points'][()]
                positions = video_info['picks'][()]
                pred_summary = eval.generate_summary([sb], [pred_score.detach().numpy()], [n_frames], [positions])[0]
                true_summary_arr = np.array(video_info['user_summary'][()])
                eval_res = eval.evaluate_summary(pred_summary, true_summary_arr, self.eval_method)
                eval_arr.append(eval_res)
        eval_mean = np.mean(eval_arr, axis=0).tolist()
        return eval_mean


    def evaluate(self):
        self.model.eval()
        eval_arr = []
        with h5py.File(self.config.data_path) as data_file:
            for feature, label, idx in tqdm(self.test_dataset, desc='Evaluate', leave=False):
                idx = str(idx[0].split("_")[-1])
                feature, label = self.to_device(feature,label)
                label =label.to(torch.long)
                feature = feature.permute(0,2,1)
                pred_score = self.model(feature).permute(0,2,1).squeeze(0)
                pred_score = torch.softmax(pred_score, dim=-1)[:,1]
                video_info = data_file["video_"+str(idx)]                
                n_frames = video_info['length'][()]
                sb = video_info['change_points'][()]
                positions = video_info['picks'][()]
                pred_summary = eval.generate_summary([sb], [pred_score.detach().numpy()], [n_frames], [positions])[0]
                true_summary_arr = np.array(video_info['user_summary'][()])
                eval_res = eval.evaluate_summary(pred_summary, true_summary_arr, self.eval_method)
                eval_arr.append(eval_res)
        eval_mean = np.mean(eval_arr, axis=0).tolist()
        return eval_mean