from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
import wandb
from sklearn.metrics import f1_score
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.preprocessing import label_binarize


# Define a custom callback for logging F1 score
class F1ScoreCallback(Callback):
    def __init__(self, model, validation_data, training_data):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.training_data = training_data

    def on_epoch_end(self, epoch, logs=None):
        # Calculate validation F1 score
        val_predictions = self.model.predict(self.validation_data[0])
        val_f1 = f1_score(np.argmax(self.validation_data[1], axis=1), np.argmax(val_predictions, axis=1),
                          average='weighted')
        wandb.log({'val_f1': val_f1}, step=epoch + 1)

        # Calculate training F1 score
        train_predictions = self.model.predict(self.training_data[0])
        train_f1 = f1_score(np.argmax(self.training_data[1], axis=1), np.argmax(train_predictions, axis=1),
                            average='weighted')
        wandb.log({'train_f1': train_f1}, step=epoch + 1)


# Custom macro F1 score function for multiclass classification
def f1_metric(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)


def train_nn(X_train, Y_train, X_val, Y_val, configs, language, model_id):
    class_mapping = {class_label: idx for idx, class_label in enumerate(sorted(Y_train.emotion.unique()))}
    Y_train_indices = Y_train.emotion.map(class_mapping)
    Y_val_indices = Y_val.emotion.map(class_mapping)
    Y_train_binary = to_categorical(Y_train_indices, num_classes=configs['num_classes'])
    Y_val_binary = to_categorical(Y_val_indices, num_classes=configs['num_classes'])
    run = wandb.init(project='emotion_classification', config=configs)
    # Model architecture
    model = models.Sequential([
        layers.Dense(configs['hl1_size'], activation='relu', input_shape=(1566,)),
        layers.Dropout(0.2),
        layers.Dense(configs['hl2_size'], activation='relu'),
        layers.Dropout(0.2)])
    if configs['use_third_hidden_layer']:
        model.add(layers.Dense(configs['hl3_size'], activation='relu'))
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(configs['num_classes'], activation='softmax'))

    # Compile the model
    model.compile(optimizer=configs['optimizer'], loss='categorical_crossentropy', metrics=['accuracy', f1_metric])
    checkpoint_filepath = f'Results/checkpoints/{language}/ckpt_model_{model_id}.h5'

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_f1_metric',  # Use the custom F1 metric as the monitor
        mode='max',
        save_best_only=True
    )
    # Print the model summary
    model.summary()

    # Callbacks for logging to WandB and F1 score
    wandb_callback = wandb.keras.WandbCallback()
    f1_callback = F1ScoreCallback(model, validation_data=(X_val, Y_val_binary), training_data=(X_train, Y_train_binary))

    # Train the model
    model.fit(X_train, Y_train_binary, epochs=configs['epochs'], batch_size=configs['batch_size'],
              validation_data=(X_val, Y_val_binary), callbacks=[wandb_callback, f1_callback, model_checkpoint_callback])
    run.finish()
    model.load_weights(checkpoint_filepath)
    train_proba = model.predict(X_train)
    val_proba = model.predict(X_val)
    train_pred = pd.DataFrame(train_proba).idxmax(axis=1).map({v: k for k, v in class_mapping.items()})
    val_pred = pd.DataFrame(val_proba).idxmax(axis=1).map({v: k for k, v in class_mapping.items()})
    return train_pred, val_pred, train_proba, val_proba, list(class_mapping.keys())
