
from model_engines.interface import ModelEngine

def create_model_engine(inner_dot,model_name='resnet_dwt') -> ModelEngine:
    if inner_dot == "cosine":
        from model_engines.resnet18_dwt_ce import ResNet18DWTModelEngine as ModelEngine

    return ModelEngine()