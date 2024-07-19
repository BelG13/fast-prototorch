import matplotlib.pyplot as plt
import hydra
import os
from omegaconf import DictConfig, OmegaConf

def plot_dict(model_losses: dict):
    """create a plot for the loss over the epoch

    Args:
        model_losses (dict): key-> model name , value-> model loss per epoch
        metrics (dict): key-> metric name , value: {model name : list metric values over batch}
        folder_path (str): 

    Returns:
        fig: the plot that has been generated
    """
    
    run_config = dict()
    @hydra.main(config_path=f"{os.getcwd()}/configs", config_name="config", version_base="1.1")
    def _main(cfg: DictConfig):
        """Get the run parameters

        Args:
            cfg (DictConfig): main config
        """

        run_config.update(dict(cfg.run_config[0]))
    
    _main()
    
    # path of the artifacts 
    run_path = "./artifacts/" + run_config.get("run_name")
    path_exists = os.path.exists(run_path)
    is_directory = os.path.isdir(run_path)
    
    # check if the run folder already exist
    if not (path_exists and is_directory):
        os.mkdir(run_path)
    
    fig, ax = plt.subplots()
    for model_name, losses in model_losses.items():
        ax.plot(list(range(len(losses))), losses, '--', label=model_name)
        
    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    
    plt.savefig(f"{run_path}/loss")


def plot_metrics(scores: dict)-> None:
    """Creates the plots for all the metrics mentionned in the config file.

    Args:
        scores (dict[dict]): The first keys are the model name and the values are a dict.
                             In the inner dict, the keys are the metric names and the values
                             are the list of values for these metrics, those we plot.
        path (str): Path in which we want to save the metrics
    """
    
    metric_names = []
    run_config = dict()
    @hydra.main(config_path=f"{os.getcwd()}/configs", config_name="config", version_base="1.1")
    def _main(cfg: DictConfig):
        """Get the metrics names and the run parameters

        Args:
            cfg (DictConfig): main config
        """
        for _metric in cfg.metrics:
            metric_names.append(_metric.name)
        run_config.update(dict(cfg.run_config[0]))
    
    _main()
    
    # path of the artifacts
    run_path = "./artifacts/" + run_config.get("run_name")
    path_exists = os.path.exists(run_path)
    is_directory = os.path.isdir(run_path)
    
    # check if the run folder already exist
    if not (path_exists and is_directory):
        os.mkdir(run_path)
    
    for metric_name in metric_names:
        
        fig, ax = plt.subplots()
        # get the metrics values, create the plot
        for model_name, values in scores.items():
            
            x = list(range(len(values[metric_name])))
            y = values[metric_name]
            
            ax.plot(x, y, '--', label=model_name)
            
        # save the plot
        ax.grid()
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend()
        plt.savefig(f"{run_path}/{metric_name}")
        
    
    
    



        
        
    