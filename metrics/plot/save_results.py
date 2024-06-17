import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

def plot_dict(model_losses: dict, path: str):
    """create a plot for the loss over the epoch

    Args:
        model_losses (dict): key-> model name , value-> model loss per epoch
        metrics (dict): key-> metric name , value: {model name : list metric values over batch}
        folder_path (str): 

    Returns:
        fig: the plot that has been generated
    """
    
    fig, ax = plt.subplots()
    for model_name, losses in model_losses.items():
        
        ax.plot(list(range(len(losses))), losses, '--', label=model_name)
        
    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    
    plt.savefig("artifacts/" + path)




def plot_metrics(scores, path: str):
    
    metric_names = []
    @hydra.main(config_path="/Users/belboss/Desktop/Coding/fast-prototorch/configs", config_name="config")
    def _main(cfg: DictConfig):
        for _metric in cfg.metrics:
            metric_names.append(_metric.name)
    
    _main()
    
    for metric_name in metric_names:
        
        fig, ax = plt.subplots()
        
        for model_name, values in scores.items():
            
            x = list(range(len(values[metric_name])))
            y = values[metric_name]
            
            ax.plot(x, y, '--', label=metric_name + "--" + model_name)
        ax.grid()
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric_name)
        ax.legend()
        plt.savefig("./artifacts/" + path)
        
    
    
    



        
        
    