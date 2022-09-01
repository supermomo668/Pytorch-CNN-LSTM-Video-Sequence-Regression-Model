
from torchmetrics import (
    JaccardIndex, Accuracy, F1Score, Dice, AUROC, MeanAbsoluteError, 
    MeanAbsolutePercentageError, MeanSquaredLogError, CosineSimilarity
)
import pytorch_lightning as pl

class METRICS:
    def __init__(self, num_seg_class:int=None, num_class:int=None):
        self.num_seg_class = num_seg_class
        # metrics
        self.all_metrics = {
            'SEG':{
                'Jaccard': JaccardIndex(num_classes=self.num_seg_class),
                'Dice': Dice(num_class=self.num_seg_class),
            },
            'REG':{
                'MAE': MeanAbsoluteError(),
                'MAPE': MeanAbsolutePercentageError(),
                'MSLE': MeanSquaredLogError(),
                'CosineSimularity': CosineSimilarity(),
            }
        }
        
    @property
    def metrics(self, selected_metrics: list=[]):
        assert all([m in self.flat_metrics for m in selected_metrics]), f"Must be one of {list(self.flat_metrics.keys())}"
        metrics=dict()
        for m in selected_metrics:
            metrics[m] = self.flat_metrics[m]
        return metrics
    
    @property
    def flat_metrics(self):
        flat_metrics=dict()
        for _, v in self.all_metrics.items():
            flat_metrics.update(v)
        return flat_metrics
            
    @property
    def SEG_metrics(self):
        return  self.metrics.get('SEG')
    
    @property
    def REG_metrics(self):
        return  self.metrics.get('REG')
    
class CALLBACKS:
    def __init__(self, model_chkpt_dir:str):
        self.model_chkpt_dir = model_chkpt_dir
        pass
    
    @property
    def all_callbacks(self):
        all_callbacks = {
            'Model Checkpoint': pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=self.model_chkpt_dir,
                                                             filename='models-{epoch:02d}-{val_loss:.2f}', 
                                                             save_top_k=2, mode='min'),
            'EarlyStopping': pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-7, patience=8, mode="min"),
            'GradientAccumulationScheduler': pl.callbacks.GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
        }
        return all_callbacks
    
    def select_callbacks(self, callbacks:list):
        """list of callbacks to use"""
        assert all([l in self.all_callbacks for l in callbacks]), f"Must be in one of {self.all_callbacks}"
        cbs = []
        for cb in callbacks:
            cbs.append(self.all_callbacks[cb])
        return cbs