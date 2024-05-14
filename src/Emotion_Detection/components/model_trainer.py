
from src.Emotion_Detection import logger
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import time
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import wandb
from src.Emotion_Detection.utils.common import LogConfMatrix,LogResultsTable,plot_training_history
import  os
from wandb.keras import WandbCallback,WandbMetricsLogger,WandbModelCheckpoint
from pathlib import Path
from src.Emotion_Detection.config.configuration import TrainingConfig



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        train_datagenerator_kwargs = dict(
            rescale = 1./255,
        )



        train_datagenerator_ = tf.keras.preprocessing.image.ImageDataGenerator(
            **train_datagenerator_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.2,
                **train_datagenerator_kwargs
            )
        else:
            train_datagenerator = train_datagenerator_



        self.train_generator = train_datagenerator.flow_from_directory(
                                                    self.config.training_data,  # Directory containing training data
                                                    class_mode="categorical",  # Classification mode for categorical labels
                                                    target_size=(224, 224),  # Resize input images to (224,224)
                                                    color_mode='rgb',  # Color mode for images (RGB)
                                                    shuffle=True,  # Shuffle training data
                                                    batch_size=self.config.params_batch_size,  # Batch size for training
                                                    subset='training'  # Subset of data (training)
                                                   )

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                  rescale=1 / 255.  # Rescale pixel values to [0,1]
                                 )
        self.test_generator = test_datagen.flow_from_directory(
                                                  self.config.testing_data,  # Directory containing testing data
                                                  class_mode="categorical",  # Classification mode for categorical labels
                                                  target_size=(224, 224),  # Resize input images to (224,224)
                                                  color_mode="rgb",  # Color mode for images (RGB)
                                                  shuffle=False,  # Do not shuffle testing data
                                                  batch_size=self.config.params_batch_size  # Batch size for testing
                                                 )
        
                # Extract class labels for all instances in the training dataset
        classes = np.array(self.train_generator.classes)

        # Calculate class weights to handle imbalances in the training data
        # 'balanced' mode automatically adjusts weights inversely proportional to class frequencies
        class_weights = compute_class_weight(
            class_weight='balanced',  # Strategy to balance classes
            classes=np.unique(classes),  # Unique class labels
            y=classes  # Class labels for each instance in the training dataset
        )

        # Create a dictionary mapping class indices to their calculated weights
        self.class_weights_dict = dict(enumerate(class_weights))

        # Output the class weights dictionary
        print("Class Weights Dictionary:", self.class_weights_dict)



                # File path for the model checkpoint
        cnn_path =self.config.model_chkpt
        name = 'best_weights'
        chk_path = os.path.join(cnn_path, name)

        # Callback to save the model checkpoint
        checkpoint = ModelCheckpoint(filepath=chk_path,
                                    save_best_only=True,
                                    verbose=1,
                                    monitor='val_accuracy',
                                    mode = 'max')

        # Callback for early stopping
        earlystop = EarlyStopping(monitor = 'val_accuracy',
                                patience = 7,
                                restore_best_weights = True,
                                verbose=1)

        # Callback to reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.2,
                                    patience=2,
        #                             min_lr=0.00005,
                                    verbose=1)

        # Callback to log training data to a CSV file
        csv_logger = CSVLogger(os.path.join(cnn_path,'training.log'))
        wandb.login(relogin=True)
        run = wandb.init(
            project = "Emotion",
            name="Trainer"
        )

        # model_checkpoint = WandbModelCheckpoint(
        #     filepath=r'model_artifiact/best_model',
        #     monitor='val_accuracy',  # Monitor validation accuracy
        #     save_weights_only=False,  # Save entire model
        #     mode='max',  # Save when validation accuracy is maximized
        #     save_best_only=True,  # Save only the best model
        #     verbose=1,  # Verbosity level
        #     filename='best_model.h5',  # Filename for saved model
        #     log_weights=True  # Log histograms of the model's layer weights
        # )

        logger.info("wandb login succefully by the name Emotion ")
        wandb_callback = WandbCallback(
                    monitor='val_accuracy',  # Monitor validation accuracy
                    save_model='best',  # Save only the best model
                    save_model_path='best_model.h5'  # Custom name for the saved model
                )
        
        # Aggregating all callbacks into a list
        self.callbacks = [wandb_callback, earlystop, csv_logger,LogResultsTable(self.test_generator,self.model),LogConfMatrix(self.test_generator,self.model)]  # Adjusted as per your use-case
     
        logger.info("callbacks list succesfully inititated")
   
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        # Log in with a different account

        self.steps_per_epoch = (self.train_generator.samples // self.train_generator.batch_size)+1
        self.test_steps_epoch = (self.test_generator.samples // self.test_generator.batch_size)+1
        logger.info("training initated ....")
        history=self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=2,
            validation_steps=1,
            validation_data=self.test_generator,
            class_weight=self.class_weights_dict,
            callbacks=self.callbacks,
        )
        logger.info("saving model....")

        # Call the function to get the figure
        fig = plot_training_history(history)

        # Log the figure to Weights & Biases
        wandb.log({"training_history": fig})
        logger.info("wandb training history logged in ")

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )