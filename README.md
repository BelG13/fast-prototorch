# Fast Prototorch

![Fast Prototorch Logo](https://dummyimage.com/100x100/000/fff&text=FP)

## Introduction 

Welcome to Fast Prototorch! This project aims to facilitate rapid prototyping of models using PyTorch. The goal is to provide an efficient, hands-on approach to quickly compare models and select the best one for your custom dataset.

## Setup

To get started, clone the repository and follow the instructions below on how to use it.

```bash
git clone https://github.com/BelG13/fast-prototorch.git
cd fast-prototorch
```

## Getting Started

### Training One Model

To see the basic use case of Fast Prototorch, navigate to the `./models` directory and create a new PyTorch model in a new Python file. Implement your model inside this file and ensure the model class has a `name` attribute. Once done, update the `./configs/config.yaml` file to add your model to the base models using the following structure:

```yaml
models:
  - name: YourModelName
    parameters:
        module: models.your_model_file.YourModelClass
        param1: value1
        param2: value2
    optimizer:
        _target_: torch.optim.<your_optimizer>
        param1: param of your optimizer
        param2: other param of your optimizer
    criterion:
        _target_: <path to your loss function>
```

After this setup, simply run:

```bash
python train.py
```

The model will be trained automatically, and you can view the training results in the `./artifacts` directory.

### Adding Multiple Models

To compare multiple models, add them to the config file using the same structure as above:

```yaml
models:
  - name: FirstModel
    parameters:
        _target_: models.your_model_file.YourModelClass
        param1: value1
        param2: value2
    optimizer:
        _target_: torch.optim.<your_optimizer>
        param1: param of your optimizer
        param2: other param of your optimizer
    criterion:
        _target_: <path to your loss function>
  - name: SecondModel
    parameters:
        _target_: models.your_model_file.YourModelClass
        param1: value1
        param2: value2
    optimizer:
        _target_: torch.optim.<your_optimizer>
        param1: param of your optimizer
        param2: other param of your optimizer
    criterion:
        _target_: <path to your loss function>
```

Now, when you run:

```bash
python train.py
```

The results will compare all the models listed in your config file.

### Using Multiple Metrics

You can also compare models using various metrics instead of just loss. To do this, create a new metric file under `./metrics` with the following structure:

- A function that computes the metric (`logits`, `y_true`, `*args`, `**kwargs`).
- A callable singleton object that uses the metric function when called.

Here's an example for accuracy:

```python
class AccuracyMetric(object):
    
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AccuracyMetric, cls).__new__(cls)
        return cls.instance
    
    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Any:
        return accuracy(logits, y)
```

Finally, include the metric(s) in your config file:

```yaml
metrics:
  - name: accuracy
    origin_class:
      _target_: metrics.accuracy.AccuracyMetric

  - name: recall
    origin_class:
      _target_: metrics.recall.RecallMetric
```

## Future Work and Updates

Stay tuned for future updates and enhancements! We plan to add more features and improve the usability of Fast Prototorch. Feel free to contribute or suggest improvements. Happy prototyping! 😊

