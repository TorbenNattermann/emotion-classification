from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
import wandb
from sklearn.metrics import f1_score
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.preprocessing import label_binarize

class Sweeper:

    def __init__(self, X_train, Y_train, X_val, Y_val, config):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.config = config
        self.run_sweep()

    def f1_metric(self, y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        f1 = 2 * precision * recall / (precision + recall + K.epsilon())
        return K.mean(f1)


    def train_model(self):
        class_mapping = {class_label: idx for idx, class_label in enumerate(sorted(self.Y_train.emotion.unique()))}
        Y_train_indices = self.Y_train.emotion.map(class_mapping)
        Y_val_indices = self.Y_val.emotion.map(class_mapping)
        Y_train_binary = to_categorical(Y_train_indices, num_classes=self.config['num_classes'])
        Y_val_binary = to_categorical(Y_val_indices, num_classes=self.config['num_classes'])
        # Initialize WandB
        run = wandb.init(project='emotion_classification')
        wandb.config.update(self.config)

        # Model architecture
        model = models.Sequential([
            layers.Dense(self.config['hl1_size'], activation='relu', input_shape=(1566,)),
            layers.Dropout(0.2),
            layers.Dense(self.config['hl2_size'], activation='relu'),
            layers.Dropout(0.2)])
        if self.config['use_third_hidden_layer']:
                model.add(layers.Dense(self.config['hl3_size'], activation='relu'))
                model.add(layers.Dropout(0.2))

        model.add(layers.Dense(self.config['num_classes'], activation='softmax'))


        # Compile the model
        model.compile(optimizer=self.config['optimizer'], loss='categorical_crossentropy', metrics=['accuracy', self.f1_metric])
        #checkpoint_filepath = 'Results/checkpoints/model_checkpoint.h5'

        # Print the model summary
        #model.summary()

        # Callbacks for logging to WandB and F1 score
        wandb_callback = wandb.keras.WandbCallback()
        early_stopping_callback = EarlyStopping(monitor='val_f1_metric', mode='max', start_from_epoch=20, patience=5)

        # Train the model
        model.fit(self.X_train, Y_train_binary, epochs=self.config['epochs'], batch_size=self.config['batch_size'],
                  validation_data=(self.X_val, Y_val_binary), callbacks=[wandb_callback, early_stopping_callback])

        run.finish()

    def run_sweep(self):
        # Now you can define your sweep configuration and initiate the sweep
        sweep_config = {
            'method': 'bayes',  # random, bayes, grid
            'metric': {'goal': 'minimize', 'name': 'val_f1_metric'},
            'parameters': {
                'hl1_size': {'values': [1024, 512]},
                'hl2_size': {'values': [1024, 512, 256]},
                'hl3_size': {'values': [128, 64]},
                'lr': {'values': [0.0001, 0.0005, 0.001]},
                'use_third_hidden_layer': {'values': [True, False]},
                'batch_size': {'values': [32, 64]}
            }
        }

        # Now you can initiate the sweep
        sweep_id = wandb.sweep(sweep_config, project='emotion_classification')

        # Then you can run the sweep
        wandb.agent(sweep_id, function=self.train_model, count=30)
