import os
from pathlib import Path
import pandas as pd

from .preProcess import create_oh_mask, create_contour_mask
from sklearn.model_selection import train_test_split


class dataIndexing:
    # Prepare data index for downstream tasks & create OH encoded mask images
    def main(self, 
             folder:Path = Path('./train/'),
             selected_folders: list = None,
             x_fn:str = "images", 
             mask_fns:list = ["masks","masks_polygon","masks_ellipse"],
             annot_fns:list = ["annotations","metadata"],
             lax: bool = False,
             stratify_cols: list = None,
             data_splits: dict = None,
             create_oh_masks: bool=True,
             create_contour_masks: bool=False
            ):
        """ Main pipeline to produce data index table. 
        Refer to the docstring for prep_data_index function for more
        """
        assert not (create_oh_masks and create_contour_masks), "Can either create regular mask or contour mask"
        print(f"Working in :{folder}")
        data_index_df = prep_all_dir(
            folder, x_fn=x_fn, mask_fns=mask_fns, annot_fns=annot_fns,
            selected_folders=selected_folders,
            lax=lax
        )
        # Add stratification columns
        if type(stratify_cols)==list and len(stratify_cols)>1: 
            data_index_df = self.add_multi_stratify_col(data_index_df, stratify_cols)
            stratify_cols='stratify'
        if data_splits:
            data_index_df = self.add_data_split(data_index_df, data_splits=data_splits, stratify_by=stratify_cols)
        # encode masks
        if create_oh_masks: create_oh_mask(data_index=data_index_df,)
        if create_contour_masks: create_contour_mask(data_index=data_index_df,)
        # save data table
        data_index_df.to_csv(folder/"data_index.csv", index=True)
        print(f"Data index saved to: {folder}")
        
    def add_multi_stratify_col(self, data_index_df, cols:list = []):
        data_index_df['stratify'] = data_index_df[cols].apply(lambda x: "-".join(x.astype(str)), 1)
        return data_index_df
    
    def add_data_split(self, data_index, data_splits:dict={'train':0.7,'val':0.1, 'test':0.2}, stratify_by:str=None):
        '''stratify train test split'''
        data_index_split = data_index.copy()
        if len([k for k in data_splits])>1:
            for sp_name in list(data_splits.keys())[:-1]:
                # ttsplit & stratify 
                (_, _, _, _, ind_train, ind_test)  = train_test_split(
                    data_index_split["x_img-path"], data_index_split["y_img-path"], 
                    data_index_split.index, stratify=data_index_split[stratify_by] if stratify_by else None, 
                    test_size=1-(data_splits[sp_name]/sum(data_splits.values())))
                # set defined （'train') set in data_index 
                data_index.loc[ind_train, 'set'] = sp_name
                # remove training set
                data_index_split = data_index_split.loc[ind_test]
                data_splits.pop(sp_name)
        # last type
        data_index.loc[ind_test, 'set'] = list(data_splits.keys())[-1]
        # show group stats
        print("Group stats:\n",data_index.groupby('set')['set'].apply(lambda x: len(x)))
        return data_index
        
def prep_data_index(data_folder: Path, 
                    x_fn:str = "images", 
                    mask_fns:list = ["masks","masks_polygon","masks_ellipse"],
                    annot_fns:list = ["annotations","metadata"],
                    meta_ext: str = ".csv",
                    lax: bool = False,
                    verbose: bool = True,
                    **kwargs) -> pd.DataFrame:
    """
    Returns Dataframe for input/output data paths
            Parameters:
                    x_fn, : input , output image & meta info folder name
                    y_fns, meta_fns (list): Possible input , output image & meta info folder names 
                    lax (bool) : relax strictness. If true, don't skip files that does not occur in all of the folders
            Returns:
                    dataframe (pandas.DataFrame): dataframe of all data paths where
                        column: ['x_img-path', 'y_img-path', 'meta-path']
                        index: x image file stem
    """
    # Define path for inputs & outputs
    x_path = data_folder/x_fn
    gt_path = data_folder/"groundtruth"
    for mask_fn in mask_fns:
        if (gt_path/mask_fn).exists(): gt_img_path = gt_path/mask_fn  
    for annot_fn in annot_fns:
        if (gt_path/annot_fn).exists(): annot_path = gt_path/annot_fn
    gt_input_img_path = gt_path/"masks-input" # processed input path
    if not gt_input_img_path.exists(): gt_input_img_path.mkdir(parents=False, exist_ok=True)
    
    # Create dataframe
    imgid_df = pd.DataFrame()
    imgid_df.index.name = 'name'
    
    # Indexing loop
        # Iterate videos
    for vid_dir in [d for d in x_path.iterdir() if d.is_dir()]:
        vid_name = vid_dir.name
        print(f"Processing video: {vid_name}")
            # annotation file
        annot_exist = (annot_path/(vid_name+'.csv')).exists()
        if annot_exist:
            annot_df = read_annot(annot_path/(vid_name+'.csv'), index_name='name')
            # Iterate directory of video frames
        for frame_fp in [im for im in vid_dir.iterdir() if im.is_file() and (im.suffix=='.jpg' or im.suffix=='.png')]:
            # Loop over x Image folder
            gt_img_fp = gt_img_path/vid_name/frame_fp.name
            gt_img_exist = os.path.isfile(gt_img_fp)
            if (gt_img_exist and annot_exist) or lax:  ## Match file against x or lax
                if verbose: print(f"FOUND: {frame_fp.name}")
                # path info (ADD index info here)
                info = pd.Series({
                    "video": vid_name,
                    "face_side": "right" if "right" in vid_name else ("left" if "left" in vid_name else "NA"),
                    "x_img-path": frame_fp, 
                    "y_img-path": gt_img_fp if gt_img_exist else None,
                    "y_input_img-path": gt_input_img_path/vid_name/frame_fp.name if gt_img_exist else None,   # where to put input mask
                    #"meta-path":annot_fn/vid_name if annot_exist else None
                })
                try:
                    info = info.append(annot_df.loc[frame_fp.name])
                    # Add information
                    info = add_annot_info(info)
                except Exception as e:
                    if verbose: print(f"For {frame_fp.name}: Error {e} encountered")
                # add to dataframe
                info.name=frame_fp.name
                imgid_df = imgid_df.append(info)
            else: # "Image not found"
                if verbose: print(f"MISSING: {x_img.name} is not found in '''{y_fns}''' images or '''{meta_fns}''' folder")
    # Put frame number
    def get_frame(x_img_name:str):
        """ get frame no. from name of image """
        return int(Path(x_img_name).stem.split('_')[-1])
    frame = pd.Series(imgid_df.index).apply(get_frame).set_axis(imgid_df.index)
    imgid_df.insert(1, 'frames', frame)
    imgid_df = imgid_df.sort_values(by=list(imgid_df.columns[:2]))   # Sort by videos then frame
    return imgid_df

def prep_all_dir(folder: Path= Path('./'), 
                 x_fn:str = "images", mask_fns:list = ["masks","masks_polygon","masks_ellipse"], annot_fns:list = ["annotations","metadata"],
                 selected_folders: list = None, lax: bool = False,
                 save=True):
    all_dfs = []
    if selected_folders is None:
        selected_folders = [f for f in folder.iterdir() if f.is_dir() and not f.name.startswith('.')]
    print(f"Indexing following folders: {selected_folders}")
    for fn in selected_folders:
        try:
            print(f"Working in directory: {str(folder/fn)}")
            data_index_df = prep_data_index(folder/fn, 
                                            x_fn=x_fn, mask_fns=mask_fns, annot_fns=annot_fns,
                                            lax=lax)
            all_dfs.append(data_index_df)
            print(f"... resulting directory index shape:{data_index_df.shape}")
        except Exception as e:
            print(f"An error occurred for directory: {str(folder/fn)}")
            print(f"Error : {e}")
    all_dfs = pd.concat(all_dfs)
    if save: all_dfs.to_csv(folder/"data_index.csv")
    return all_dfs

def add_annot_info(info: pd.Series):
    info['PIR'] = info['radiusY_p_true']/info['radiusY_i_true']
    return info

def read_annot(fp, index_name:str = None):
    if index_name is None: index_name=0
    if fp.suffix==".csv":
        return pd.read_csv(fp, index_col= index_name)
    else:
        raise Exception('No support', 'format not supported for the annotations')
    


if __name__ =="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_fp', '-data_fp', help='folder name of x_images', type=str, default=Path('./train'))
    ap.add_argument('--n_class', '-n_class', help='number of class o label', type=intt, default=3)
    ap.add_argument('--x_fn', '-img_fn', help='folder name of x_images', type=str, default='x_fn')
    ap.add_argument('--y_fn', '-mask_fn', help='folder name of mask images', type=str, default='y_fn')
    ap.add_argument('--meta_fn', '-meta_fn', help='folder name of meta info', type=str, default='groundtruth')
    args = ap.parse_args()
    
    run_dataindexing = dataIndexing()
    run_dataindexing.main(folder = args.data_fp,
                          selected_folders = None,
                          n_class = 3,
                          x_fn= args.x_fn,
                          y_fn = args.mask_fn, 
                          meta_fn= args.meta_fn
                         )