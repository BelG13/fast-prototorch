import torch
import torchvision
import numpy as np
import logging
from models.ProtoTest import ProtoTest
from metrics.accuracy import accuracy
from datasets.titanic.titanic import DataTitanic
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from tqdm import tqdm
from metrics.plot.save_results import plot_dict, plot_metrics

import hydra
from omegaconf import DictConfig, OmegaConf


def train(batch_size=128, shuffle=True, epochs = 20):
    """train all the models based on their configurations.

    Args:
        batch_size (int, optional): Defaults to 128.
        shuffle (bool, optional): Defaults to True.
        epochs (int, optional): Defaults to 20.

    Returns:
        _type_: they keys are model names and the values are lists that store the loss per epoch.
    """
    
    # logging.basicConfig(filename="train.py", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # key: model name , value: model object
    models = {}   
    
    # key: model name , value: criterion object  
    criterions = {}
    
    # key: model name , value: optimizer object
    optimizers = {}
    
    # key: metric name, value: metric object
    metrics = {}
    
    @hydra.main(version_base=None, config_path="configs", config_name="config")
    def _main(cfg: DictConfig):
        """Processes the information from the configuration files and
           update and set up the training hyperparameters.

        Args:
            cfg (DictConfig): configuration and specifications of the models.
        """
        
        # models init
        for _model in cfg.base_models:
            models[_model.name]     = instantiate(_model.parameters)
            criterions[_model.name] = instantiate(_model.criterion)
            optimizers[_model.name] = instantiate(
                _model.optimizer,
                params=models[_model.name].parameters()
            )
        
        # metrics init
        for _metric in cfg.metrics:
            metrics[_metric.name] = instantiate(_metric.origin_class)
            
    logger.info("Extraction of model configurations")
    _main()
    
    # data importation
    trainLoader = DataLoader(
        DataTitanic(mode="training", path="./datasets/titanic/csv/train.csv"),
        batch_size=batch_size,
        shuffle=shuffle
    ) 
    
    # trainLoader = torch.utils.data.DataLoader(
    #                 torchvision.datasets.MNIST('./datasets/MNIST', train=True, download=True,
    #                                             transform=torchvision.transforms.Compose([
    #                                             torchvision.transforms.ToTensor(),
    #                                             torchvision.transforms.Normalize(
    #                                                 (0.1307,), (0.3081,))
    #                                             ])),
    #                 batch_size=batch_size, shuffle=True)

    model_losses = {}
    scores = {}
    for name, model in models.items():
        logger.info(f"Training {name} :")  
         
        losses = []
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            
            temp = {}
            if scores.get(name) is None:
                scores[name] = {}
            
            if temp.get(name) is None:
                temp[name] = {}
                
                
            for i, (X, y) in enumerate(trainLoader):
                
                logits = model(X)
                # loss = criterions[name](preds, y.unsqueeze(-1).float())
                loss = criterions[name](logits, y.long())
                loss.backward()
                total_loss += loss.item()
                
                optimizers[name].step()
                optimizers[name].zero_grad()
                
                
                for metric_name, metric in metrics.items():
                    
                    if scores.get(name).get(metric_name) is None:
                        scores[name][metric_name] = []
                        
                    if temp.get(name).get(metric_name) is None:
                        temp[name][metric_name] = [metric(logits, y)]
                    else:
                        temp[name][metric_name].append(
                            metric(logits, y)
                        )

            for metric_name, metric in metrics.items():
                scores[name][metric_name].append(np.array(temp[name][metric_name]).mean())
                
            losses.append(total_loss / len(trainLoader))     
        
        model_losses[name] = losses
                
    return model_losses, scores
            
            
if __name__ == "__main__":
    
    model_losses, scores = train()
    # print(scores)
    plot_dict(model_losses, path="test_loss.png")
    plot_metrics(scores, "test_metric.png")
    
    
    