"""Expose a classifier."""
from logging import getLogger
import torch.nn as nn
import torch.utils.data as tud
import torch.optim as to
from greentea.text import Texts
from .vectorizer import Vectorizer
from .theme import Themes


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


class PreTrainedTextVecMlpClassifier:
    """Use a pre-trained text vectorizer."""

    LOGGER = getLogger(__name__)

    def __init__(self, vectorizer: Vectorizer, classifier: MlpClassifier):
        """Take a trained vectorizer a :py:class:`MlpClassifier` to train."""
        self.vectorizer = vectorizer
        self.classifier = classifier

    def train(self,
              dataloader: tud.DataLoader,
              epochs=1000,
              learning_rate=1e-3):
        """Fit :py:attr:`classifier` on `dataloader`.

        Parameters
        ----------
        dataloader: DataLoader

        """
        parameters = self.classifier.parameters()
        criterion = nn.CrossEntropyLoss()
        optimizer = to.Adam(parameters, lr=learning_rate)
        for epoch in range(epochs):
            self.LOGGER.info(f'epoch {epoch + 1}')
            self._epoch_train(dataloader, criterion, optimizer, epoch + 1)

    def _epoch_train(self,
                     dataloader: tud.DataLoader,
                     criterion,
                     optimizer,
                     epoch,
                     log_loss_period=2000):
        running_loss = 0.0
        for batch_index, dataset in enumerate(dataloader):
            self.LOGGER.verbose(f'batch {batch_index + 1}')
            texts, themes = dataset
            running_loss += self._batch_train(Texts(texts),
                                              Themes(themes),
                                              criterion,
                                              optimizer)
            if batch_index % log_loss_period == log_loss_period - 1:
                self.LOGGER.info(
                    '[%d, %5d] loss: %.3f' %
                    (epoch, batch_index + 1, running_loss / log_loss_period))
                running_loss = 0.0

    def _batch_train(self, texts: Texts, themes: Themes, criterion, optimizer):
        # zero the parameter grandients.
        text_vectors = self.vectorizer.transform(texts)
        features = text_vectors.as_torch_tensor()
        optimizer.zero_grad()
        outputs = self.classifier(features)
        labels = themes.get_index()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()
