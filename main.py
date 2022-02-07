import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from face_detection_operation import get_detected_face


class FaceRecognition:

    print(tf.version, " tf ver")

    # if use CPU
    # config = tf.compat.v1.ConfigProto(
    #     device_count={'GPU': 0}
    # )
    # sess = tf.compat.v1.Session(config=config)

    #tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    #inserted
    TRAINING_DATA_DIRECTORY = "./dataset" #
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    BATCH_SIZE = 32
    training_generator = None
    NumberOfPersons = 4 # configurable

    EPOCHS = 10 # for compile2

    model_path = "model"
    model_name = "fine_tuning.h5"

    # @staticmethod
    # def data_generator():
    #     img_data_generator = ImageDataGenerator(
    #         rescale=1. / 255,
    #         # horizontal_flip=True,
    #         fill_mode="nearest",
    #         # zoom_range=0.3,
    #         # width_shift_range=0.3,
    #         # height_shift_range=0.3,
    #         rotation_range=30
    #     )
    #     return img_data_generator

    @staticmethod
    def load_saved_model(model_path):
        model = load_model(model_path)
        return model

    def print_hi(name):
        # Use a breakpoint in the code line below to debug your script.
        print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    # Press the green button in the gutter to run the script.
    if __name__ == '__main__':


        #print_hi('PyCharm')
        base_dir = "./dataset" #
        IMAGE_SIZE = 224
        BATCH_SIZE = 5

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.1, # 90/10
            fill_mode="nearest",
            rotation_range=30,
            # shear_range = 0.2,
            # zoom_range = 0.2,
            # horizontal_flip=True,
            # zoom_range=0.3,
            # width_shift_range=0.3,
            # height_shift_range=0.3,
        )

        #training data
        train_generator = data_generator.flow_from_directory(
            base_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            subset='training')
        #validation data
        val_generator = data_generator.flow_from_directory(
            base_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            subset='validation')

        # Triggering a training generator for all the batches
        for image_batch, label_batch in train_generator:
            break

        # This will print all classification labels in the console
        print(train_generator.class_indices)

        # Creating a file which will contain all names in the format of next lines
        labels = '\n'.join(sorted(train_generator.class_indices.keys()))

        # Writing it out to the file which will be named 'labels.txt'
        with open('labels.txt', 'w') as f:
            f.write(labels)

        # Resolution of images (Width , Height, Array of size 3 to accommodate RGB Colors)
        IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

        # creating a model with excluding the top layer
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(NumberOfPersons, activation='softmax')  # anzahl der Personen
        ])

        # Adam Algorithm
        # model.compile(optimizer=tf.keras.optimizers.Adam(),
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])

        # DSG Algorithm
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-2 / EPOCHS),
            metrics=["accuracy"]
        )

        # To see the model summary in a tabular structure
        model.summary()

        # Printing some statistics
        print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

        # Train the model
        # We will do it in 10 Iterations
        epochs = 20

        # Fitting / Training the model
        history = model.fit(train_generator,
                            epochs=epochs,
                            validation_data=val_generator)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

        # Setting back to trainable
        base_model.trainable = True

        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(base_model.layers))

        # Fine tune from this layer onwards
        fine_tune_at = 100

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

            #inserted
            # img_data_generator = ImageDataGenerator(
            #     rescale=1. / 255,
            #     # horizontal_flip=True,
            #     fill_mode="nearest",
            #     # zoom_range=0.3,
            #     # width_shift_range=0.3,
            #     # height_shift_range=0.3,
            #     rotation_range=30
            # )
            #inserted
            # training_generator = img_data_generator.flow_from_directory(
            #     TRAINING_DATA_DIRECTORY,
            #     target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            #     batch_size=BATCH_SIZE,
            #     class_mode='categorical'
            # )
            #training_generator - for testing.py

            # names from Dataset
            class_names = train_generator.class_indices
            #
            class_names_file_reverse = model_name[:-3] + "_class_names_reverse.npy"
            class_names_file = model_name[:-3] + "_class_names.npy"
            #####
            np.save(os.path.join(model_path, class_names_file_reverse), class_names)
            class_names_reversed = np.load(os.path.join(model_path, class_names_file_reverse), allow_pickle=True).item()
            class_names = dict([(value, key) for key, value in class_names_reversed.items()])
            np.save(os.path.join(model_path, class_names_file), class_names)
            #print("items from train_generator saved to /model")

            #var 1
            # model.compile(
            #     loss='categorical_crossentropy',
            #     optimizer=tf.keras.optimizers.Adam(1e-5),
            #     metrics=["accuracy"]
            # )

            # var 2
            model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-2 / 15),
                metrics=["accuracy"]
            )

        # Compile the model

        # Getting the summary of the final model
        model.summary()
        # Printing Training Variables
        print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

        # Continue Train the model
        history_fine = model.fit(train_generator,
                                 epochs=15, # config
                                 validation_data=val_generator
                                 )

        saved_model_dir = 'model/fine_tuning.h5'
        model.save(saved_model_dir)
        class_names = train_generator.class_indices
        print("Model Saved to model/fine_tuning.h5")

        # optional
        # tf.saved_model.model(model, saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # //Use this if 238 fails
        # tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)

        '''
        '''
        acc = history_fine.history['accuracy']
        val_acc = history_fine.history['val_accuracy']
        loss = history_fine.history['loss']
        val_loss = history_fine.history['val_loss']
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()



        # def save_model(self, model_name):
        #     model_path = "./model"
        #     # if not os.path.exists(model_name):
        #     #     os.mkdir(model_name)
        #     if not os.path.exists(model_path):
        #         os.mkdir(model_path)












