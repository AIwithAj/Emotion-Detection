import os
from box.exceptions import BoxValueError
import yaml
from src.Emotion_Detection import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import wandb
from tensorflow.keras.callbacks import Callback
from wandb.keras import WandbCallback

from src.Emotion_Detection import logger

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_training_history(history):
    logger.info("logging training history mmatrix ...")

    """
    Plots the training and validation accuracy and loss.

    Parameters:
    - history: A Keras History object. Contains the logs from the training process.

    Returns:
    - None. Displays the matplotlib plots for training/validation accuracy and loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    # Plot training and validation accuracy
    axes[0].plot(epochs_range, acc, label='Training Accuracy')
    axes[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].set_title('Training and Validation Accuracy')

    # Plot training and validation loss
    axes[1].plot(epochs_range, loss, label='Training Loss')
    axes[1].plot(epochs_range, val_loss, label='Validation Loss')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Training and Validation Loss')

    plt.close()  # Close the figure to prevent it from being displayed

    return fig  # Return the figure object



class LogConfMatrix(Callback):
  def __init__(self,test_generator,model):
      logger.info("logging confusion mmatrix ...")
      
      self.model=model
      self.test_generator = test_generator
  def on_epoch_end(self, epoch, logs):
        true_classes = self.test_generator.classes
        predicted_classes = np.argmax(self.model.predict(self.test_generator, steps=int(np.ceil(self.test_generator.samples/self.test_generator.batch_size))), axis=1)
        class_labels = list(self.test_generator.class_indices.keys())
        try:
            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(true_classes, predicted_classes)

            # Plot precision-recall curve
            plt.figure(figsize=(10, 5))
            plt.plot(recall, precision, lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            plt.legend(loc="best")

            # Log the plot to wandb
            wandb.log({"precision_recall_curve": plt})

            # Show the plot (optional)
            plt.show()
        except Exception as e:
            print("Failed to log precision recall curve: ",e)

        cm = wandb.plot.confusion_matrix(
            y_true=true_classes, preds=predicted_classes, class_names=class_labels
            )


        wandb.log({"conf_mat": cm})


        # Convert true and predicted labels to one-hot encoded format
        y_encoded = pd.get_dummies(true_classes).astype(int).values
        preds_encoded = pd.get_dummies(predicted_classes).astype(int).values

        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(class_labels)):
            fpr[i], tpr[i], _ = roc_curve(y_encoded[:, i], preds_encoded[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves
        plt.figure(figsize=(10, 5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        for i, color in enumerate(colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"ROC curve for {class_labels[i]} (area = {roc_auc[i]:0.2f})")

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')

        # Log the plot to wandb
        wandb.log({"roc_curve": plt})

        # Show the plot (optional)
        plt.show()


class LogResultsTable(Callback):
    def __init__(self,test_generator,model):
      logger.info("logging  ResultTable ...")
      self.model=model
      self.test_generator = test_generator
    def on_epoch_end(self, epoch, logs):
        columns = ["image", "Predicted", "Label"]
        val_table = wandb.Table(columns=columns)
        class_labels = list(self.test_generator.class_indices.keys())

        for i in range(25):
            im, label = next(self.test_generator)

            # Predict the class probabilities for the input image
            predictions = self.model.predict(im)

            # Get the index of the predicted class with the highest probability
            pred_index = np.argmax(predictions, axis=-1)

            # Convert the predicted index to the corresponding class label
            pred_label = class_labels[pred_index[0]]

            # Convert the true label to the corresponding class label
            true_label_index = np.argmax(label, axis=-1)
            true_label = class_labels[true_label_index[0]]

            row = [wandb.Image(im[0]), pred_label, true_label]
            val_table.add_data(*row)

        wandb.log({"Model Results": val_table})
