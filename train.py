import os
import logging
import argparse
import utils
import squeezenet
import time
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import SGD

# Remove the TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

# ImageNet weights
WEIGHTS_PATH = "./weights/squeezenet_imagenet_weights_notop.h5"


# Train with given batchsize & epochs number
def train(train_dir, mean_image_path, batchsize, num_epochs,
          lr, weight_decay_l2, img_height, img_width, weights):

    # Make a './model' directory to store trained model and model parameters
    if not os.path.exists('./model'):
        os.makedirs('./model')

    # Data augmentation
    datagen = ImageDataGenerator(featurewise_center=True, samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False, zca_epsilon=1e-06,
                                 rotation_range=20, width_shift_range=0.1,
                                 height_shift_range=0.1, brightness_range=None,
                                 shear_range=0.01, zoom_range=0.1,
                                 channel_shift_range=0.0, fill_mode='nearest',
                                 cval=0.0, horizontal_flip=True, vertical_flip=False,
                                 rescale=None, preprocessing_function=None,
                                 data_format="channels_last", validation_split=0.25, dtype=None)

    # Dataset image mean to center the data
    img_mean_array = img_to_array(load_img(mean_image_path, target_size=(img_height, img_width)))
    datagen.mean = img_mean_array

    # Train generator
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        color_mode='rgb',
        batch_size=batchsize,
        class_mode='categorical',
        subset='training',  # set as training data
        interpolation='bilinear')

    # Validation generator
    validation_generator = datagen.flow_from_directory(
        train_dir,  # same directory as training data
        target_size=(img_height, img_width),
        color_mode='rgb',
        batch_size=batchsize,
        class_mode='categorical',
        subset='validation',  # set as validation data
        interpolation='bilinear')

    classes = list(train_generator.class_indices.keys())
    num_classes = len(classes)

    # SGD Optimizer
    opt = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False)

    # Generate and compile the model
    model = squeezenet.SqueezeNet(num_classes, weight_decay_l2, inputs=(img_height, img_width, 3))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Steps per epoch
    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = validation_generator.n // validation_generator.batch_size

    step_size_train = 5
    step_size_valid = 2

    # Linear LR decay after each batch update
    linearDecayLR = utils.LinearDecayLR(min_lr=1e-5, max_lr=lr,
                                        steps_per_epoch=step_size_train,
                                        epochs=num_epochs, verbose=1)

    # Add ImageNet weights if needed
    if weights == 'imagenet':
        model.load_weights(WEIGHTS_PATH, by_name=True)

    # Train the model
    print("Start training the model")
    training = model.fit_generator(
        train_generator,
        steps_per_epoch=step_size_train,
        validation_data=validation_generator,
        validation_steps=step_size_valid,
        epochs=num_epochs,
        verbose=1,
        callbacks=[linearDecayLR])
    print("Model training finished")

    # Make model parameters to be used for prediction
    model_parms = {'num_classes': num_classes,
                   'classes': classes,
                   'height': img_height,
                   'width': img_width}

    # Create the training history
    train_history = training.history

    return model, model_parms, train_history


if __name__ == "__main__":
    # Parse the arguements
    parser = argparse.ArgumentParser(description="SqueezeNet Training.")

    parser.add_argument("--dir", type=str, default='./train')
    parser.add_argument("--mean-image", type=str, default='./images/mean_image.png', dest='mean_image')
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay-l2", type=float, default=0.0001, dest='weight_decay_l2')
    parser.add_argument("--img-width", type=int, default=128, dest='width')
    parser.add_argument("--img-height", type=int, default=128, dest='height')
    parser.add_argument("--weights", type=str, default='')
    args = parser.parse_args()

    # Execution time
    start_time = time.time()

    # Train the model
    model, model_parms, training_history = train(args.dir, args.mean_image,
                                                 args.batchsize, args.epochs,
                                                 args.lr, args.weight_decay_l2,
                                                 args.width, args.height, args.weights)
    # Save the trained model
    model.save('./model/squeezenet_model.h5')
    print("Trained Squeezenet Keras model is saved")

    # Save the model parms for prediction
    utils.save_model_parms(model_parms, fname='./model/model_parms.json')
    print("Model parameters (classes, image size) are saved")

    # Save the training history for train/val loss and accuracy
    utils.save_training_history(training_history, fname='./model/training_history.npy')
    print("Training history of loss, accuracy and learning rate is saved")

    # Execution time
    duration = time.time() - start_time
    print("Execution time: " + time.strftime("%H:%M:%S", time.gmtime(duration)))

    # Plot the training history
    utils.plot_training_history(training_history)
