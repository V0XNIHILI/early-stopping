# Early stopping for neural network training

Early stopping is a form of regularization used to avoid overfitting on the training dataset. Early stopping keeps track of the validation loss, if the loss stops decreasing for several epochs in a row the training stops. The [`EarlyStopping`](./early_stopping/EarlyStopping.py) class is used to create an object to keep track of the validation loss while training a model. It is possible to set the `patience` argument in the [`EarlyStopping`](./early_stopping/EarlyStopping.py) class to indicate how many epochs to wait after the last time the validation loss improved before breaking the training loop.

Underneath is a plot which shows the last epoch allowed by the `EarlyStopping`` object, right before the model started to overfit. It had patience set to 20.

![Loss plot](assets/loss_plot.png?raw=true)


## Installation

```bash
pip install git+ssh://git@github.com/V0XNIHILI/early-stopping.git
```

## Usage

```python
from early_stopping import EarlyStopping

def save_model():
    # Save model code

# With optional new_best_callback argument
stop_early = EarlyStopping(patience=20, verbose=True, new_best_callback=save_model)

# Set up remainder of training loop as usual

for epoch in range(epochs):
    # Training code
    # ...
    # Validation code

    if stop_early(val_loss):
        break
```

## References

The ```EarlyStopping``` class in ```pytorchtool.py``` is inspired by the [Ignite EarlyStopping class](https://github.com/pytorch/ignite/blob/master/ignite/handlers/early_stopping.py). The code in this repository is originally based on [this](https://github.com/Bjarten/early-stopping-pytorch) project from [Bjarte Mehus Sunde](https://github.com/Bjarten).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


