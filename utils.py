import json
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback
from scipy.signal import savgol_filter


# TRAIN

# Reduce learning rate linearity every batch update (from GitHub)
class LinearDecayLR(Callback):
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None, verbose=0):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.verbose = verbose

    # Compute the learning rate
    def linear_decay(self):
        r = self.iteration / self.total_iterations
        return self.max_lr - (self.max_lr - self.min_lr) * r

    # Initialize the learning rate
    def on_train_begin(self, logs=None):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    # Update the learning rate every batch update
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, self.linear_decay())

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, K.get_value(self.model.optimizer.lr)))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


# Save the model parameters in json file
def save_model_parms(data, fname='./model_parms.json'):
    with open(fname, 'w') as fp:
        json.dump(data, fp)


# Save training history in npy file
def save_training_history(history_dict, fname='./training_history.npy'):
    np.save(fname, history_dict)
    return


# Plot training history of accuracy & loss
def plot_training_history(history_dict):
    epochs = np.arange(1, len(history_dict['acc']) + 1)
    fig = plt.figure(figsize=(12, 6))

    # Plot train/val accuracy
    plt.subplot(2, 1, 1)

    yhat_a = savgol_filter(history_dict['acc'], 13, 3)
    yhat_va = savgol_filter(history_dict['val_acc'], 13, 3)
    plt.plot(epochs, yhat_a, color='blue', label='Train Acc')
    plt.plot(epochs, yhat_va, color='red', label='Val Acc')

    plt.legend()
    plt.title('Training History')
    plt.ylabel('Accuracy')
    plt.xlim(0, len(history_dict['acc']) + 1)

    # Plot train/val loss
    plt.subplot(2, 1, 2)

    yhat_l = savgol_filter(history_dict['loss'], 13, 3)
    yhat_vl = savgol_filter(history_dict['val_loss'], 13, 3)
    plt.plot(epochs, yhat_l, color='blue', label='Train Acc')
    plt.plot(epochs, yhat_vl, color='red', label='Val Acc')

    plt.legend()
    plt.ylabel('Loss')
    plt.xlim(0, len(history_dict['acc']) + 1)

    plt.show()
    return


# PREDICT

# Load model parameters from json file
def load_model_parms(fname='./model_parms.json'):
    with open(fname) as data_file:
        data = json.load(data_file)
    return data


# Show the highlighted image & plot the predictions
def plot_predict_history(image, history_arr):
    dpi = 50  # 50
    fig_w = image.shape[1] / dpi
    fig_h = 2 * (image.shape[0] / dpi)
    fig = plt.figure(figsize=(fig_w, fig_h))

    plt.subplot(2, 1, 1)
    plt.imshow(image, aspect='equal')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.margins(0)
    xaxis = np.concatenate((history_arr, np.repeat(0., 14)))
    xaxis = np.concatenate((np.repeat(0., 13), xaxis))
    plt.plot(np.arange(1, len(xaxis) + 1), xaxis, color='black')
    plt.ylim(-1., 1.)
    plt.xticks([], [])
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', color='#666666', linewidth=0.5)
    plt.grid(which='minor', linestyle=':', color='#999999', alpha=0.4)
    plt.axhspan(0, 1., alpha=0.2, color='blue')
    plt.axhspan(-1., 0, alpha=0.2, color='red')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    plt.show()
    return
