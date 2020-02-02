"""Expose a classifier."""
import torch.nn as nn


class MlpClassifier(nn.Module):
    """A multi-layred perceptron.

    References
    ----------
    https://developers.google.com/machine-learning/guides/text-classification/step-4

    """

    def __init__(self,
                 input_shape,
                 num_classes,
                 units=64,
                 dropout_rate=0.2):
        """Construct a multi layer perceptron.

        Parameters
        ----------
        input_shape: int
            shape of input to the model.

        num_classes: int
            number of output classes.

        dropout_rate: float
            Percentage of input to drop at Dropout layers.

        """
        super(MlpClassifier, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc0 = nn.Linear(input_shape, units)
        self.fc1 = nn.Linear(units, num_classes)
        if num_classes == 2:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.LogSoftmax(dim=num_classes)

    def forward(self, x):
        """Define the computation performed at every call."""
        x = nn.Dropout(self.dropout_rate)(x)
        x = self.fc0(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(self.dropout_rate)(x)
        x = self.fc1(x)
        return self.activation(x)


class Classifier:
    """
    """

    def __init__(self, classifier: MlpClassifier):
        """
        """
        self.classifier = classifier

    def train(self, dataloader):
        """
        """
        raise NotImplementedError
