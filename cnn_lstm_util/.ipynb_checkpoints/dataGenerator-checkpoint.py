import pandas as pd, numpy as np
#
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import pytorch_lightning as pl
#
from sklearn.model_selection import train_test_split
#
import albumentations as A, cv2, albumentations.pytorch
#
from .dataIndex import dataIndexing
from .preProcess import filter_to_valid_seq_dataindex, mask_to_categorical, compute_im_stats, load_img_df


class TASK_args:
    mode = 'SEG-REG'
    seed = 101
    
    
class DATA_args:
    y_reg_cols = [
        'coord_p_true_x', 'coord_p_true_y',
        'radiusX_p_true', 'radiusY_p_true',
        'coord_i_true_x', 'coord_i_true_y',
        'radiusX_i_true', 'radiusY_i_true']
    sequence_settings = {
        'sequence_length':41, 
        'mode': 'middle', 
        'gap': 0}
    augment = True
    transform_settings = {
        'normalize': True,
    }
    
    batch_size = 8
    
class augmentations:
    def __init__(self, additional_targets:dict={}, augment:bool=True, normalize:bool=True,
                 pad_resize:tuple=(224,224), target_resize:tuple=None):
            
        self.augment = augment
        # augmentations
        p1 = 0.1
        p2 = 0.05
        p3 = 0.2
        geometric_transformation = [
            # A.RandomSizedCrop(
            #     min_max_height=(int(height / 1.2), height),
            #     height=height,
            #     width=width,
            #     p=p1,
            # ),
            # A.GridDistortion(p=p1),
            A.Affine(
                scale=(0.60, 1.60),
                interpolation=cv2.INTER_LINEAR,
                cval=0,
                cval_mask=0,
                mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                p=p1,
            ),
            A.Affine(
                translate_percent=(-0.2, 0.2),
                interpolation=cv2.INTER_LINEAR,
                cval=0,
                cval_mask=0,
                mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                p=p1,
            ),
            A.Affine(
                rotate=(-30, 30),
                interpolation=cv2.INTER_LINEAR,
                cval=0,
                cval_mask=0,
                mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                p=p1,
            ),
            A.Affine(
                shear=(-20, 20),
                interpolation=cv2.INTER_LINEAR,
                cval=0,
                cval_mask=0,
                mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                p=p1,
            ),
            # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=pt),
            A.HorizontalFlip(p=0.5),
        ]

        color_transformations = [
            # In-place transformations
            A.RandomBrightnessContrast(p=p2),
            A.RandomGamma(gamma_limit=(80, 200), p=p3),
            A.Blur(blur_limit=7, p=p2),
            A.ToGray(p=p2),
            A.CLAHE(p=p2),
            A.ChannelDropout(channel_drop_range=(1, 2), fill_value=0, p=p2),
            A.ChannelShuffle(p=p2),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
                always_apply=False,
                p=p2,
            ),
            A.Equalize(mode="cv", by_channels=True, mask=None, mask_params=(), p=p2),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=p2),
            A.Posterize(num_bits=4, p=p2),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=p2),
        ]
        process_transformation =  []
        if normalize: color_transformations += [A.Normalize(always_apply=True)]
        if pad_resize is not None: process_transformation += [A.PadIfNeeded(*pad_resize, always_apply=True)]
        if target_resize is not None: process_transformation += [A.Resize(*target_size)]
        process_transformation += [A.pytorch.ToTensorV2()]
        #
        pipe = []
        self.color_transformations = A.Compose(color_transformations, 
                                               additional_targets={k:v for k,v in additional_targets.items() if k!='mask'})
        self.geometric_transformation = A.Compose(geometric_transformation, additional_targets=additional_targets)
        self.process_transformation = A.Compose(process_transformation,
                                                additional_targets=additional_targets)
    
    def apply_full_augment(self, **img_inputs): 
        if self.augment:
            img_inputs = self.color_transformations(**img_inputs)
            img_inputs = self.geometric_transformation(**img_inputs)
            img_inputs = self.process_transformation(**img_inputs)
        return img_inputs
    
    def apply_reg_augment(self, **img_inputs): 
        if self.augment:
            img_inputs = self.color_transformations(**img_inputs)
            img_inputs = self.process_transformation(**img_inputs)
        return img_inputs
    
############
class VideoSequenceData(Dataset):
    def __init__(self, dataindex_df, seq_start_idx, mode:str,  
                 sequence_settings:dict={
                     'sequence_length':40, 'mode': 'middle','sequence_gap': 1, 'frames_gap':0
                 },
                 x_img_col:str='x_img-path', y_img_col:str='y_img-path', y_reg_cols:list=[],
                 transform_settings:dict={
                     'augment':True, 'target_resize':(224,224), 'normalize':True, 
                 },
                 debug:bool=False, **kwargs):
        """ 
        parameters
            dataindex_df: dataframe with columns
            mode: one of ['SEG', 'REG', 'SEG-REG']
            sequence_settings (dict)
                sequence_length(int): e.g. 40
                mode (str): forward: predict 1 forward. middle: predict middle of sequence.
                sequence_gap (int): gap between sequence
                frames_gap (int): gap between sampled frames
            y_reg_cols (list): e.g. ['coord_p_true_x', 'coord_p_true_y']
            mode (str):              
                SEG: output only image
                REG: output only regression variables
                SEG-REG: output both
        """
        assert all([c in sequence_settings for c in ['sequence_length','mode']]), f"Parameters required are missing"
        # set attribute
        self.dataindex_df = dataindex_df
        self.sequence_settings = sequence_settings
        self.augment = transform_settings.get('augment', False)
        self.mode = mode
        self.seq_mode=self.sequence_settings.get('mode', 'forward')
        self.debug = debug
        # Ensure correctness. & how much room to spare in sequence
        if self.seq_mode=='middle': 
            assert self.sequence_settings.get('sequence_length')%2==1, "Mode middle must use an odd number sequence number"
        # filter for legitamate sequence dataframe
        self.seq_dataindex_df = dataindex_df.loc[seq_start_idx]
        if self.debug: print(f"dataindex shape: {dataindex_df.shape}, legal dataindex shape:{self.seq_dataindex_df.shape}")
            # 
        self.x_img_df = dataindex_df[x_img_col]
        if self.mode=='SEG' or self.mode=='SEG-REG':
            self.y_img_df = dataindex_df[y_img_col]
            self.num_classes = len(np.unique(self.load_img(self.y_img_df.iloc[0:1])))
        if mode=='REG' or mode=='SEG-REG':
            self.y_reg_df = dataindex_df[y_reg_cols]
        # augmentation for all targets in a window
        if self.augment:
            # set additional aug targets
            self.im_aug_targets = ['image'+str(i) for i in range(1, sequence_settings.get('sequence_length'))]
            self.augs = augmentations(additional_targets={target: 'image' for target in self.im_aug_targets+['mask']},
                                      **transform_settings)
            if debug: print(f"Number of additional augmentation targets:{len(self.im_aug_targets)}")
        assert mode in ['REG', 'SEG', 'SEG-REG'],"Invalid mode"
        print(f"Task mode:{mode}")
        
    def __getitem__(self, idx):
        """ Get item from the seq_index_df but fetch from dataindex_df since seq_dataindex_df is the legitamate starting idx with range from 0 to length given by the __n__ method        
        """
        start_idx = self.dataindex_df.index.get_loc(self.seq_dataindex_df.iloc[
            idx*self.sequence_settings.get('sequence_gap', 1)].name)
        seq_idxs = list(range(start_idx, start_idx + self.sequence_settings.get('sequence_length') +
                              self.sequence_settings.get('frames_gap',0)*(self.sequence_settings.get('sequence_length')-1),
                              self.sequence_settings.get('frames_gap',0)+1))
        x_img_data = self.load_img(self.x_img_df.iloc[seq_idxs])   
        if self.debug: print(x_img_data.shape, x_img_data.dtype, x_img_data.max(), x_img_data.min(), )
        # target seq idx
        if self.seq_mode=='forward': target_idx = [seq_idxs[-1]+1]
        elif self.seq_mode=='middle': target_idx = [seq_idxs[len(seq_idxs)//2]]
            # SEG/ SEG-REG
        if self.mode == 'SEG'  or self.mode=='SEG-REG':
            y_mask_data = self.load_img(self.y_img_df.iloc[target_idx], is_mask=True)[0]
            if self.debug: print(f"y_mask shape: {y_mask_data.shape}")
            if self.mode == 'SEG' and self.augment:
                x_img_data, y_mask_data = self.apply_seq_transforms(**{'x_img_data':x_img_data, 'y_mask_data':y_mask_data})
                y_mask_data = np.squeeze(mask_to_categorical(y_mask_data, num_classes=self.num_classes))
            # SEG only
            if self.mode == 'SEG':
                return {'image': x_img_data, 'mask': y_mask_data}
                #return torch.tensor(x_img_data), torch.tensor(y_mask_data)
            # REG/ SEG-REG
        if self.mode == 'REG' or self.mode=='SEG-REG':  # REG or SEG-REG
            if self.augment:
                x_img_data = self.apply_seq_transforms(**{'x_img_data': x_img_data})
            y_reg_data = self.y_reg_df.iloc[target_idx].to_numpy().flatten()
            # REG only
            if self.mode == 'REG':
                return {'image':x_img_data, 'y': y_reg_data}
        if self.debug: print(f"Augmented shapes:{x_img_data.shape, y_mask_data.shape}")
        if self.debug: print(f"Post-aug stat: {x_img_data.min(), x_img_data.max()}, {y_mask_data.min(), y_mask_data.max()}")
        # SEG-REG
        return {'image':x_img_data, 'mask':y_mask_data, 'y': y_reg_data}

    def __len__(self):
        return len(self.seq_dataindex_df)//self.sequence_settings.get('sequence_gap', 1)
        
    def apply_seq_transforms(self, **img_inputs):
        aug_inputs = {target: img_inputs['x_img_data'][n] for (n, target) in enumerate(self.im_aug_targets, 1)}  
        aug_inputs.update({'image':img_inputs['x_img_data'][0].astype('uint8')})
        if self.mode=="SEG":
            # define transformation targets
             # Augment sequence image data
            img_data_transformed = self.augs.apply_full_augment(**aug_inputs, 
                                                                **{'mask': img_inputs['y_mask_data'].astype('uint8')})
        else:
            img_data_transformed = self.augs.apply_reg_augment(**aug_inputs)
        x_img_data = np.concatenate(
            [np.expand_dims(img_data_transformed[target], axis=0) for target in ['image']+self.im_aug_targets])
        if 'mask' in img_data_transformed:
            return x_img_data, img_data_transformed['mask']   #torch.tensor(y_mask_data)
        else:
            return x_img_data
    
    def load_img(self, img_paths: list, is_mask=False):
        """ load array from a list of image paths """
        if is_mask: flag = 0
        else: flag = -1
        return np.concatenate([np.expand_dims(cv2.imread(
            img_fp, flag), axis=0) for img_fp in img_paths.tolist()])
    
    @property
    def columns(self):
        return self.dataindex_df.columns

    
class VideoSequenceDataModule(pl.LightningDataModule):
    def __init__(self, dataindex_path, mode:str,  batch_size: int = 32,
                 sequence_settings:dict={
                     'sequence_length':40, 'mode': 'middle', 'sequence_gap': 1   #'frames_gap':1
                 },
                 x_img_col:str='x_img-path', y_img_col:str='y_img-path', y_reg_cols:list=[],
                 transform_settings:dict={
                     'augment':True, 'target_resize':(224,224), 'normalize':True
                 }, debug:bool=False, **kwargs):
        super().__init__()
        self.dataindex_df = pd.read_csv(dataindex_path, index_col=0)
        self.seq_dataindex_df = filter_to_valid_seq_dataindex(self.dataindex_df, vid_col='video', 
                                                              seq_len=sequence_settings.get('sequence_length'))
        print(f"Sequence settings:{sequence_settings}")
        self.batch_size = batch_size
        # pass all variables to each dataloader and remove class ones
        self.all_args = locals()
        keys = list(self.all_args.keys())
        [self.all_args.pop(k) for k in keys if (k in ['batch_size', 'dataindex_path']) or k.startswith('__')  or k.startswith('self')]
        print(f"Arguments to be passed to dataloader: {self.all_args.keys()}")

    def setup(self, stage:str = None):
        if not hasattr(self, 'datasets'):  self.datasets = dict()
        # Assign train/val datasets for use in dataloaders
        subset = self.dataindex_df['set'].unique().tolist() + ['predict']
        print(f"Dataset subsets:{subset}")
        #['train','val','test','predict']
        for dset in subset:
            if (stage == dset or stage is None) and dset not in self.datasets:  # 
                self.datasets[dset] = VideoSequenceData(self.dataindex_df,
                    self.dataindex_df[self.dataindex_df['set']==dset].index, **self.all_args)
        if stage == None:
            pass
        
    def train_dataloader(self):
        self.setup(stage='train')
        return DataLoader(self.datasets['train'], batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size)

    def test_dataloader(self):
        raise Exception('not implemented')
        return DataLoader(self.datasets['test'], batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.datasets['predict'], batch_size=self.batch_size)

    @property
    def full_loader(self):
        full_dataset = VideoSequenceData(self.dataindex_df, batch_size=self.batch_size, debug=self.debug)
        return DataLoader(full_dataset, batch_size=self.batch_size)
            
    def teardown(self, stage: str = None):
        # Used to clean-up when the run is finished
        self.finish()