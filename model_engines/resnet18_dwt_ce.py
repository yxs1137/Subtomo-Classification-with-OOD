
import torch
import torch.nn as nn

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features
from dataloaders.factory import get_train_dataloader, get_id_dataloader, get_ood_dataloader
from LOSS import tvMFLoss

from models_dwt_3d_ce.resnet import resnet18,resnet34,resnet50
from LOSS import tvMFLoss

class ResNet18DWTModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ResNetDWT(args)
        state_dict = torch.load('./pretrained_model/resnet18_dwt_bior3.3_64_best.pth.tar', map_location='cpu')['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = 'feat.'+k[7:]
            else:
                new_k = 'feat.'+k
                
            # if k.startswith("module."):
            #     new_k = k[7:]
            # else:
            #     new_k = k
        
            new_state_dict[new_k] = v
        msg = self._model.load_state_dict(new_state_dict, strict=False)            

        self._model.to(self._device)
        self._model.eval()
    
    def set_dataloaders(self):
        self._dataloaders = {}
        self._dataloaders['train'] = get_train_dataloader(self._data_root_path, 
                                                         self._train_data_name,
                                                         self._batch_size, 
                                                         num_workers=self._num_workers)

        self._dataloaders['id'] = get_id_dataloader(self._data_root_path, 
                                                         self._id_data_name,
                                                         self._batch_size, 
                                                         num_workers=self._num_workers)
        self._dataloaders['ood'] = get_ood_dataloader(self._data_root_path, 
                                                         self._ood_data_name,
                                                         self._batch_size, 
                                                         num_workers=self._num_workers)
        
    def train_model(self):
        pass
    
    def get_model_outputs(self):
        model_outputs = {}
        for fold in self._folds:
            model_outputs[fold] = {}
            
            _dataloader = self._dataloaders[fold]
            _tensor_dict = extract_features(self._model, _dataloader, self._device)
            
            model_outputs[fold]["feas"] = _tensor_dict["feas"]
            model_outputs[fold]["logits"] = _tensor_dict["logits"]
            model_outputs[fold]["labels"] = _tensor_dict["labels"]
        
        return model_outputs['train'], model_outputs['id'], model_outputs['ood']


model_dict = {'resnet18_dwt':[resnet18,512],
              'resnet34_dwt':[resnet34,512],
              'resnet50_dwt':[resnet50,2048],
              }
class ResNetDWT(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18_dwt', head='linear', feat_dim=512, num_classes=12):
        super(ResNetDWT, self).__init__()
        # model_fun, dim_in = model_dict[name]
        self.feat = resnet18()
        self.fc_loss = tvMFLoss(feat_dim,num_classes)

    def forward(self, x):
        with torch.no_grad():
            rep,logits = self.feat(x)
        return rep,logits 
    
