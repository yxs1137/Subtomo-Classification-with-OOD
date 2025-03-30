import time

import random
import torch
import argparse
import numpy as np

from tabulate import tabulate

from misc.utils_python import mkdir, import_yaml_config, save_dict, load_dict

from model_engines.factory import create_model_engine
from model_engines.interface import verify_model_outputs
from ood_detectors.factory import create_ood_detector

from eval_assets import save_performance

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, 
                        default='resnet18_dwt',)
    parser.add_argument('--inner_dot',  type=str, 
                        default='cosine', choices=['cosine','t-vMF'])
    parser.add_argument('--seed', type=int, 
                        default=1, 
                        help='Seed number')

    parser.add_argument('--gpu_idx', '-g', type=int, 
                        default=0, 
                        help='gpu idx')
    parser.add_argument('--num_workers', '-nw', type=int, 
                        default=8, 
                        help='number of workers')
    parser.add_argument('--train_data_name', '-td', type=str,  
                        default='shrec19',
                        
                        help='The data name for the in-distribution')
    parser.add_argument('--id_data_name', '-id', type=str,  
                        default='shrec19',
                        
                        help='The data name for the in-distribution')
    parser.add_argument('--ood_data_name', '-ood', type=str, 
                        default='real', 
                        help=['inaturalist', 'sun', 'places', 'textures', 'openimage-o']
                        )
    
    parser.add_argument("--ood_detectors", type=str, nargs='+', 
                        default=['mahalanobis',],
                        help="List of OOD detectors")

    parser.add_argument('--batch_size', '-bs', type=int, 
                        default=32, 
                        help='Batch size for inference')

    parser.add_argument('--data_root_path', type=str, 
                        default='./filelists', 
                        help='Data root path')
    parser.add_argument('--save_root_path', type=str,
                        default='./saved_model_outputs')

    parser.add_argument('--n_cls', type=int, 
                    default=12, )

    args = parser.parse_args()
    args.device = torch.device('cuda:%d' % (args.gpu_idx) if torch.cuda.is_available() else 'cpu')

    args = import_yaml_config(args, f'./configs/model/{args.model_name}.yaml')
    
    args.log_dir_path = f"./logs/seed-{args.seed}/{args.model_name}/{args.train_data_name}/{args.id_data_name}"
    
    args.train_save_dir_path = f"{args.save_root_path}/seed-{args.seed}/{args.model_name}/{args.train_data_name}"
    args.id_save_dir_path = f"{args.save_root_path}/seed-{args.seed}/{args.model_name}/{args.id_data_name}"
    args.ood_save_dir_path = f"{args.save_root_path}/seed-{args.seed}/{args.model_name}/{args.ood_data_name}"

    args.detector_save_dir_path = f"{args.save_root_path}/seed-{args.seed}/{args.model_name}/{args.train_data_name}/detectors"
    
    mkdir(args.log_dir_path)
    
    mkdir(args.train_save_dir_path)
    mkdir(args.id_save_dir_path)
    mkdir(args.ood_save_dir_path)

    mkdir(args.detector_save_dir_path)

    print(tabulate(list(vars(args).items()), headers=['arguments', 'values']))

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():

    args = get_args()
    set_seed(args.seed)

    scores_set = {}
    accs = {}
    for oodd_name in args.ood_detectors:
        args = import_yaml_config(args, f"./configs/detector/{oodd_name}.yaml")
        scores_set[oodd_name], labels, accs[oodd_name] = evaluate(args, oodd_name)

    save_performance(scores_set, labels, accs, f"{args.log_dir_path}/ood-{args.ood_data_name}.csv")

def evaluate(args, ood_detector_name: str):
    
    '''
    Executing model engine
    '''
    print(f"[{args.model_name} / {ood_detector_name}]: running model...")

    model_engine = create_model_engine(args.inner_dot,args.model_name)
    model_engine.set_model(args)
    model_engine.set_dataloaders()
    
    
    save_dir_paths = {}
    save_dir_paths['train'] = args.train_save_dir_path
    save_dir_paths['id'] = args.id_save_dir_path
    save_dir_paths['ood'] = args.ood_save_dir_path

    model_outputs = {}
    labels = {}
    
    try:
        for fold in ['train', 'id', 'ood']:
            model_outputs[fold] = torch.load(f"{save_dir_paths[fold]}/model_outputs_{fold}.pt")
    except:
        model_engine.train_model()
        model_outputs = {}
        model_outputs['train'], model_outputs['id'], model_outputs['ood'] = model_engine.get_model_outputs()
        for fold in ['train', 'id', 'ood']:
            
            assert verify_model_outputs(model_outputs[fold])
            torch.save(model_outputs[fold], f"{save_dir_paths[fold]}/model_outputs_{fold}.pt")
    
    labels = {}
    labels['id'] = model_outputs['id']['labels']
    labels['ood'] = model_outputs['ood']['labels']

    '''
    Executing ood detector
    '''
    print(f"[{args.model_name} / {ood_detector_name}]: running detector...")

    saved_detector_path = f"{args.detector_save_dir_path}/{ood_detector_name}.pkl"
    try:
        ood_detector = load_dict(saved_detector_path)["detector"]
    except:
        ood_detector = create_ood_detector(ood_detector_name)
        ood_detector.setup(args, model_outputs['train'])
        
        print(f"[{args.model_name} / {ood_detector_name}]: saving detector...")
        save_dict({"detector": ood_detector}, saved_detector_path)
        print(f"[{args.model_name} / {ood_detector_name}]: detector saved!")

    '''
    Evaluating metrics
    '''
    print(f"[{args.model_name} / {ood_detector_name}]: evaluating metrics...")
    id_scores = ood_detector.infer(model_outputs['id'])
    ood_scores = ood_detector.infer(model_outputs['ood'])
    
    scores = torch.cat([id_scores, ood_scores], dim=0).numpy()
    
    id_logits = model_outputs['id']['logits']
    ood_logits = model_outputs['ood']['logits']
    detection_labels = torch.cat([torch.ones_like(labels['id']), torch.zeros_like(labels['ood'])], dim=0).numpy()
    all_logits = torch.cat([id_logits,ood_logits],dim=0)
    
    all_labels = torch.cat([labels['id'], -torch.ones_like(labels['ood'])], dim=0).numpy()
    
    preds_id = []
    threshold = np.percentile(id_scores, 5)

    for idx, score in enumerate(scores):
        if score >= threshold:
            logit = all_logits[idx]
            pred = np.argmax(logit)
        else:
            pred = -1  # OOD 类别
        preds_id.append(pred)
    preds_id = np.array(preds_id) 
    acc = np.mean(preds_id == all_labels)
    
    return scores, detection_labels, acc

if __name__ == '__main__':
    main()