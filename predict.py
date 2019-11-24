import cv2
import argparse
import utils
import os
import logging
import time
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Remove the TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


# Predict the class of an image using the model
def predict(test_image_object, mean_image_path, model, model_parms):
    K.clear_session()

    # Load trained model
    model = load_model(args.saved_model)

    # Load model parms
    model_parms = utils.load_model_parms(args.model_parms)

    # Mean image
    img_mean_array = img_to_array(load_img(args.mean_image,
                                           target_size=(model_parms['height'], model_parms['width'])))
    # Test image (either a single image or from a list of images)
    if type(test_image_object) is str:
        img_test_array = img_to_array(
            load_img(args.test_image, target_size=(model_parms['height'], model_parms['width'])))
        img_test_array -= img_mean_array
        img_test_batch = np.expand_dims(img_test_array, axis=0)
    else:
        img_test_array = img_to_array(test_image_object)
        img_test_array -= img_mean_array
        img_test_batch = np.expand_dims(img_test_array, axis=0)

    # Predict the class of the test image
    prob = model.predict(x=img_test_batch, batch_size=1, verbose=1, steps=None)
    prediction = np.argmax(prob, axis=1)[0]
    return np.amax(prob), model_parms['classes'][prediction]


if __name__ == "__main__":
    # Parse arguements
    parser = argparse.ArgumentParser(description="SqueezeNet Prediction.")
    parser.add_argument("--test-image", type=str, default='./images/test_image.jpg', dest='test_image')
    parser.add_argument("--mean-image", type=str, default='./images/mean_image.png', dest='mean_image')
    parser.add_argument("--saved-model", type=str, default='./model/squeezenet_model.h5', dest='saved_model')
    parser.add_argument("--model-parms", type=str, default='./model/model_parms.json', dest='model_parms')
    args = parser.parse_args()

    # Execution time
    start_time = time.time()

    name = os.path.realpath(args.test_image).rsplit(".", 1)[0]
    ext = os.path.splitext(os.path.basename(args.test_image))[1]

    src = cv2.imread(args.test_image)
    epochs = -(-(src.shape[1] - 50) // 2)

    # Predict the class of a single image
    if src.shape[0] + src.shape[1] < 101:
        predicted_prob, predicted_class = predict(args.test_image, args.mean_image, args.saved_model, args.model_parms)
        print(predicted_class + " (" + str(round(predicted_prob, 2)) + ")")
        duration = time.time() - start_time
        print("Execution time: " + time.strftime("%H:%M:%S", time.gmtime(duration)))
    # Predict the class of each image contained in a large image of a handwritten line
    else:
        img = src.copy()
        y, x, _ = img.shape
        prob_history = []
        var = 1
        e_list = [[]]
        e_i = 0
        margin = 2
        e_m = margin
        for i in range(0, x - 50, 2):
            frame = src[0:50, i:i + 50]
            model_parms = utils.load_model_parms(args.model_parms)
            frame = cv2.resize(frame, (model_parms['height'], model_parms['width']), interpolation=cv2.INTER_NEAREST)
            frame = (frame[..., ::-1])
            predicted_prob, predicted_class = predict(frame, args.mean_image, args.saved_model, args.model_parms)
            print(str(var) + "/" + str(epochs) + " - " + predicted_class + " (" + str(round(predicted_prob, 2)) + ")")
            var += 1
            # Set the prediction as positive if "e" or negative if "!e" for plotting
            if predicted_class == "e" and predicted_prob >= 0.9:
                prob_history = np.append(prob_history, predicted_prob)
                e_list[e_i].append(i)
                e_m = margin
            else:
                prob_history = np.append(prob_history, -predicted_prob)
                if e_m <= 0:
                    e_i += 1
                    e_list.append([])
                e_m -= 1
        # Smooth the highlights (not the plot)
        for j in range(len(e_list)):
            if len(e_list[j]) >= 5:
                e_min = e_list[j][0]
                e_max = e_list[j][-1]
                cv2.rectangle(img, (e_min + 22, 0), (e_max + 27, 49), (0, 0, 255), -1)
        alpha = 0.1
        img = cv2.addWeighted(img, alpha, src, 1 - alpha, 0)

        # Save highlighted image
        cv2.imwrite(name + "_p_4" + ext, img)

        # Execution time
        duration = time.time() - start_time
        print("Execution time: " + time.strftime("%H:%M:%S", time.gmtime(duration)))

        # Show the highlighted image & plot the predictions
        utils.plot_predict_history(img, prob_history)
