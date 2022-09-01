from PIL import Image

import numpy as np, cv2, pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]  

def mask_to_categorical(y_mask_data, num_classes):
    """convert a label mask image to one-hot"""
    y_mask_data = np.apply_along_axis(to_categorical, 0, y_mask_data, num_classes)
    y_mask_data = np.moveaxis(y_mask_data, 0, -1)   # channel-last
    return y_mask_data
#############################
# utility funcs
def load_img_df(data_index, im_col:str,
               is_mask: bool = False):
    """
    data_indx: pd.DataFrame /pd.Series
    if_mask: shape of image is (H, W)
        return np.array with shape (batch, h, w, c), c >= 1
    """
    imgs = []
    for idm in data_index.index:
        try:
            if type(data_index)==pd.Series:
                im = np.array(Image.open(data_index.loc[idm]))
            else:
                im = np.array(Image.open(data_index.loc[idm][im_col]))
            if is_mask and im.ndim>2: im = im[:,:,0]  # duplicate channel (non-grayscale)
            imgs.append(im)
        except: pass
    imgs = np.asarray(imgs)
    if imgs.ndim <= 3: imgs = np.expand_dims(np.array(imgs), -1)
    print("load_img_df dtype:", imgs.dtype)
    return imgs

# filter dataindex to ensure legitemate index dataframe
def filter_to_valid_seq_dataindex(imgid_df:pd.DataFrame, seq_len:int, vid_col='video', frame_col='frames'):
    def filter_from_max(s):
        """filter frames such that sequence length will not exceed frame"""
        valid = s[frame_col] <= s[frame_col].max()-seq_len  
        return s[valid]
    return imgid_df.groupby(vid_col).apply(filter_from_max).droplevel(0)

def compute_im_stats(x_imgs):
    mean = np.mean(x_imgs, axis=tuple(range(len(x_imgs.shape)-1)))
    std = np.std(x_imgs, axis=tuple(range(len(x_imgs.shape)-1)))
    return mean, std

###############################
## Creating OH mask
def create_oh_mask(data_index,
                   y_col:str='y_img-path', y_input_col:str='y_input_img-path',):
                   #n_class:int=3)  -> None:
    """
    Use OHencode_masks to create input mask images
        parameters: 
            data_indx (pd.DataFrame): data index dataframe with paths of x-imgs and y-imgs given
                        by columns 'x_img-path' and 'y_img-path'
            n_class: number of classes (including bg)
    """
    imgs = load_img_df(data_index, y_col, is_mask=True)
    n_class=len(np.unique(imgs[0]))
    print(f"[INFO] Creating masks from image: {imgs.shape} with {n_class} classes.\n")
    oh_masks = OHencode_masks(imgs[:,:,:,0], n_class=n_class)
    print(f"[INFO] Created masks: {oh_masks.shape}.\n")
    Path(data_index.iloc[0][y_input_col]).parent.mkdir(parents=True, exist_ok=True)
    for n, idx in enumerate(data_index.index):
        Image.fromarray(oh_masks[n]).save(data_index.loc[idx][y_input_col])
        
def OHencode_masks(masks: np.array, 
                   n_class: int):
    """
    One-hot encoding of masks provided as array 
        parameters 
            masks (np.array((batch, height, width))): masks to be one-hot encoded
        reurn 
            OH-encoded masks (np.array((batch, height, width, class)))
    """
    labelencoder = LabelEncoder()
    n, h, w = masks.shape
    masks_reshaped_encoded = labelencoder.fit_transform(masks.reshape(-1,1))
    masks_encoded_original_shape = masks_reshaped_encoded.reshape(n, h, w)
    ## 
    print("Converting to categorical classes.")
    masks_input = np.expand_dims(masks_encoded_original_shape, axis=3)
    masks_cat = to_categorical(masks_input, num_classes=n_class)
    masks_cat = masks_cat.reshape((n, h, w, n_class))
    print(f"Encoded mask with {n_class} classes -> final shape {masks_cat.shape}")
    return masks_cat.astype('uint8')

def OHencode_masks_batchmode(masks: np.array, n_class: int, n_per_batch:int=1500):
    """ use OHencode_masks but perform batch-wise to reduce memory use
    # batch mode
    """
    n_masks = len(masks)
    batch_masks = []
    for i in range(int(np.ceil(n_masks/n_per_batch))):
        batch_masks.append(OHencode_masks(masks[i*n_per_batch: min(n_masks, (i+1)*n_per_batch)], n_class=n_class))
    return np.concatenate(batch_masks)

#### creaet OH contour masks

def create_contour_mask(data_index,
                        y_col:str='y_img-path', y_input_col:str='y_input_img-path',):
    """
    Use get_contours to create input contour mask images
        parameters: 
            data_index: data index dataframe with paths of x-imgs and y-imgs given
                        by columns 'x_img-path' and y_input_col
            n_class: number of classes (including bg)
    """
    imgs = load_img_df(data_index, y_col, is_mask=True)
    oh_contour_masks = get_contours_all(imgs[:,:,:,0])
    print(f"[INFO] Created contour masks: {oh_contour_masks.shape}.\n")
    for n, idx in enumerate(data_index.index):
        Path(data_index.loc[idx][y_input_col]).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(oh_contour_masks[n]).save(data_index.loc[idx][y_input_col])

def get_contours_all(label_imgs, contour_dilation:tuple=(3,3)):
    """ 
    label_imgs: np.array with shape (batch, h, w)
    return np.array with shape (batch, h, w)
    """
    _, x_dim, y_dim = label_imgs.shape
    label_imgs = label_imgs.reshape((len(label_imgs), -1))
    oh_imgs = np.apply_along_axis(lambda x: get_contours(x.reshape((x_dim, y_dim)), 
                                                         contour_dilation =contour_dilation), 
                                 axis=-1, arr=label_imgs)
    return oh_imgs
    
def get_contours(label_img, return_contours:bool = False, dilate:tuple=(3,3)):
    all_contours = []
    n_class = len(np.unique(label_img))
    im = np.zeros(label_img.shape+(n_class,), dtype='uint8')
    for c in np.unique(label_img):
        if c==0:continue
        contours_ = cv2.findContours(
            (label_img==c).astype('uint8'), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[0][0]
        contours_ = contours_.reshape((contours_.shape[0], contours_.shape[2]))
        im[..., c][contours_[:,1], contours_[:,0]] = 1   # x, y
        if not dilate is None:
            im[..., c] = cv2.dilate(im[..., c], np.ones(dilate))
    im[..., 0] = np.clip(np.subtract(np.ones(label_img.shape), np.sum(im, axis= -1)), 0, 1)
    if return_contours:
        return im, contours
    return im

### Blurring for pixel weighting
def blur_contour(bin_contour):
    """ return array (array.dtype = 'float')"""
    return cv2.GaussianBlur(bin_contour.astype('float32'), (7,7), 3)

def blur_contour_all(bin_contours):
    img_shape = bin_contours.shape[1:]
    bin_contours = bin_contours.reshape((len(bin_contours), -1))
    blurred_contours = np.apply_along_axis(lambda x: blur_contour(x.reshape(img_shape)),
                                           axis=-1, arr=bin_contours)
    return blurred_contours

def compute_im_stats(x_imgs):
    mean = np.mean(x_imgs, axis=tuple(range(len(x_imgs.shape)-1)))
    std = np.std(x_imgs, axis=tuple(range(len(x_imgs.shape)-1)))
    return mean, std