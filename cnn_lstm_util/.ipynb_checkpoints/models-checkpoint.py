
import pytorch_lightning as pl
from torch import nn
from torchvision import models
import torch, torch.nn.functional as F
from .metrics import METRICS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_cnn_output(model, input_shape:tuple=(3,224,224)):
    test_inputs = torch.randn((1,) +input_shape)
    outputs = model(test_inputs.to(device))
    return outputs.shape

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        return x
    
class CNN_head_custom(nn.Module):
    def __init__(self, model_name:str, pretrain:bool=True, trainable:bool=False):
        super(CNN_head_custom, self).__init__()
        avail_models =  ['mobilenetv3','resnext101','efficientnetb7','resnet50']
        #assert model_name in avail_models, f"Must be one of {avail_models}"       
        # Step 1: Initialize model with the weights
        model_registry = {
            'mobilenetv3': models.mobilenet_v3_small(pretrained=pretrain),
            'mobilenetv2': models.mobilenet_v2(pretrained=pretrain),
            'resnext101': models.resnext101_32x8d(pretrained=pretrain),    
            'efficientnetb7': models.efficientnet_b7(pretrained=pretrain),
            'resnet50': models.resnet50(pretrained=pretrain)
        }
        model = model_registry[model_name]
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = trainable
        # replace/remove head
        removed = list(model.children())[:-1]
        self.model_base = nn.Sequential(*removed, nn.Flatten()) 
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model_base.to(device)
    def forward(self, x):
        return self.model_base(x)
        
class CNN_LSTM(pl.LightningModule):
    def __init__(self, 
                 cnn_head_model,
                 lstm_settings = {'hidden_size':None, 'num_layers':1}, 
                 mode:str='SEG',
                 n_seg_class:int = 2,
                 n_reg_var=8, loss:dict={'SEG':['Jaccard'], 'REG': ['MAE']}, **kwargs):
        """
        CNN LSTM model inherits from CNN
        param
            mode [str]: ['SEG', 'REG','SEG-REG']
            n_seg_class [int]
            n_reg_var [int]
            metrics [list]: see metrics.py for more
        """
        super(CNN_LSTM, self).__init__()
        self.mode = mode
        # inherent cnn head
        self.cnn = cnn_head_model
        lstm_settings['input_size'] = list(cnn_head_model.parameters())[-1].size()[-1]
        print(f"CNN head output size:{lstm_settings['input_size']}")
        self.rnn = nn.LSTM(**lstm_settings, bidirectional=True, batch_first=True)
        self.lin_out = nn.Linear(lstm_settings.get('hidden_size')*2, n_reg_var)
        # metrics
        self.loss = loss
        self.all_loss = METRICS(num_seg_class=n_seg_class, num_class=n_reg_var).all_metrics 
        #self.metrics = {k: METRICS(num_seg_class=n_seg_class, num_class=n_reg_var).flat_metrics[k] for k in loss_metrics}
    
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.lin_out(r_out[:, -1, :])
        #
        return F.log_softmax(r_out2, dim=1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def evaluate_metrics(self, stage:str='val'):
        for m, m_func in self.metrics.items(): 
            value = m_func(y_hat, y)
            self.log(f"{stage}+_{m}", value, on_step=True, on_epoch=True, logger=True)
    
    ## Step functions for train/val/test
    def training_step(self, batch, batch_idx):
        x, y = self._fetch_from_batch(batch)
        # training metrics
        y_hat = self(x)
        losses =self.compute_loss(y_hat, y, stage='train')
        return losses['MAE']
    
    def val_step(self, batch, batch_idx):
        x, y = self._fetch_from_batch(batch)
        y_hat = self(x)
        losses =self.compute_loss(y_hat, y, stage='val')
        return losses['MAE']

    def test_step(self, batch, batch_idx):
        x, y = self._fetch_from_batch(batch)
        y_hat = self(x)
        losses =self.compute_loss(y_hat, y, stage='val')
        return losses['MAE']
    
    def compute_loss(self, y_hat, y, stage:str='train'):
        """
        compute loss for all the desired loss stated at __init__, record and pass back
        """
        l = dict()
        for l_type in ['SEG', 'REG']:
            if l_type not in self.mode: continue   
            print(f"Computing loss for: {self.loss[l_type]}")
            for l_name in self.loss[l_type]:
                if l_type =='SEG':
                    loss = self.all_loss[l_type][l_name](y_hat, y['mask'])
                else:
                    loss = self.all_loss[l_type][l_name](y_hat, y['y'])
                l[l_name] = loss
                self.log(f'{stage}_loss-{l_name}', loss, on_step=True, on_epoch=True, logger=True)
        return l
        
    def _fetch_from_batch(self, batch):
        """ fetch x, y from batch """
        x = batch.pop('image')
        return x, batch