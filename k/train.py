from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets.cifar10 import load_data
import keras
import json
import numpy as np


# todo: Supects
# todo: 1. Wrong blocks type
# todo: 2. Kernel size
# todo: 3. Data augment
# todo: 4. Learning rate decay
# todo: Optional, ResnetC using batch size 32 has archived more than 93.5%


def lr_reduce(epoch):
    if epoch < 100:
        return 1e-2
    elif 100 <= epoch < 175:
        return 1e-3
    else:
        return 1e-4


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


if __name__ == '__main__':
    args = DotDict({
        'batch_size': 32,
        'val_batch_size': 10,
        'cuda': True,
        'model': '',
        'train_plot': False,
        'logdir': 'logs/k',
        'epochs': 180,
        'try_no': '1_resnet50',
        'imsize': 224,
        'traindir': '/root/palm/DATA/plant/train',
        'valdir': '/root/palm/DATA/plant/validate',
        'workers': 16,
        'resume': False,
    })
    model = keras.applications.resnet50.ResNet50()
    count = 0
    print(model.summary())
    model.compile(optimizer=sgd(lr=0.01, momentum=0.9, nesterov=1),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    reduce_lr = LearningRateScheduler(lr_reduce, verbose=1)
    checkpoint = ModelCheckpoint(f'k/weights/try-{args.try_no}.h5',
                                 monitor='val_acc',
                                 mode='max',
                                 save_best_only=1,
                                 save_weights_only=1)
    tensorboard = TensorBoard(log_dir=args.logdir,
                              histogram_freq=1,
                              write_graph=True,
                              write_images=False)

    train_generator = ImageDataGenerator(width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.15,
                                         zoom_range=0,
                                         horizontal_flip=True,
                                         vertical_flip=False,
                                         preprocessing_function=keras.applications.resnet50.preprocess_input
                                         )
    test_generator = ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)
    train_datagen = train_generator.flow_from_directory(args.traindir,
                                                        batch_size=args.batch_size,
                                                        target_size=args.imsize
                                                        )
    test_datagen = test_generator.flow_from_directory(args.traindir,
                                                      batch_size=args.val_batch_size,
                                                      target_size=args.imsize)

    f = model.fit_generator(train_datagen,
                            epochs=args.epoch,
                            validation_data=test_datagen,
                            callbacks=[checkpoint, reduce_lr, tensorboard])
