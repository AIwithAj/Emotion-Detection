import tensorflow as tf
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.Emotion_Detection.entity.config_entity import EvaluationConfig
from src.Emotion_Detection.utils.common import save_json

import wandb


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):


        datagenerator_kwargs = dict(
            rescale = 1./255,
            
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
         
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            class_mode="categorical",
            color_mode="rgb",
            shuffle=False,
            **dataflow_kwargs
        )



    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_wandb(self):
        wandb.login(relogin=True)
        run = wandb.init(
            project = "Emotion",
            name="Evaluation"
        )
        wandb.log(
            {"loss": self.score[0], "accuracy": self.score[1]}
        )
        true_classes = self.valid_generator.classes
        predicted_classes = np.argmax(self.model.predict(self.valid_generator, steps=np.ceil(self.valid_generator.samples/self.valid_generator.batch_size)), axis=1)
        class_labels = list(self.valid_generator.class_indices.keys())
        try:

            run.log({"pr": wandb.plots.precision_recall(y_true=true_classes, y_probas=predicted_classes, labels=class_labels)})
        except Exception as e:
            print("Failde to login precision recall curve ")

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
        wandb.finish()
