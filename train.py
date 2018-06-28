import sys
import argparse
import os, os.path
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import print_summary
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', default=None)
parser.add_argument('--tensor_board', default='/tmp/cats_dogs')
parser.add_argument('--optimizer', default='adam')
#parser.add_argument('--trainset', default='kaggle')
parser.add_argument('--fc_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--conv_a', type=int, default=32)
parser.add_argument('--conv_b', type=int, default=64)
parser.add_argument('--conv_c', type=int, default=128)
parser.add_argument('--conv_d', type=int, default=128)
args = parser.parse_args()

print args

image_size = 8
for i in xrange(4):
	image_size = image_size * 2 + 2
print "image_size:%s" % image_size


def file_count_rec(d):
    return sum(len(files) for path, dirs, files in os.walk(d))

# dimensions of our images.
img_width, img_height = image_size, image_size

dataset = 'data_small.flickr'
nb_samples = file_count_rec(dataset)
validation_split = 0.2

nb_validation_samples = int(validation_split * nb_samples)
nb_train_samples = nb_samples - nb_validation_samples

epochs = args.epochs
batch_size = args.batch_size

kernel_init = 'glorot_normal'
classes = 2

if K.image_data_format() == 'channels_first':
    axis = 1
    input_shape = (3, img_width, img_height)
else:
    axis = 3
    input_shape = (img_width, img_height, 3)

model = Sequential()

model.add(Conv2D(32, (3, 3), kernel_initializer=kernel_init, input_shape=input_shape))
model.add(BatchNormalization(axis=axis, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer=kernel_init))
model.add(BatchNormalization(axis=axis, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), kernel_initializer=kernel_init))
model.add(BatchNormalization(axis=axis, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), kernel_initializer=kernel_init))
model.add(BatchNormalization(axis=axis, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), kernel_initializer=kernel_init))
model.add(BatchNormalization(axis=axis, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), kernel_initializer=kernel_init))
model.add(BatchNormalization(axis=axis, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))
  
model.add(Flatten())
if classes == 1:
	model.add(Dense(1, activation='sigmoid'))
	class_mode = 'binary'
else:
	model.add(Dense(classes, activation='softmax'))
	class_mode = 'categorical'

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
print sgd

optimizer=args.optimizer # or sgd

model.compile(loss=class_mode+'_crossentropy', optimizer=optimizer, metrics=['accuracy'])

if args.load_model is not None:
    # TODO load entire model
    model.load_weights(args.load_model)

model_id = 1
while os.path.isfile("model_%s.txt" % model_id):
    model_id += 1

# this is the augmentation configuration we will use for training
augmentation = {'shear_range':0.2, 'zoom_range':0.2, 'horizontal_flip':True}
  #rotation_range=10,
train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=validation_split, **augmentation)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=validation_split)

train_generator = train_datagen.flow_from_directory(
    dataset,
    #save_to_dir="train_out",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=class_mode,
	subset="training")

validation_generator = test_datagen.flow_from_directory(
    dataset,
    #save_to_dir="validation_out",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=class_mode,
	subset="validation")

callbacks = [
  #ModelCheckpoint('models/' + model_id + '.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', save_best_only=True, monitor='val_acc'),
  TensorBoard(log_dir="%s/model_%s" % (args.tensor_board, model_id))
]

def output(f, s):
    f.write(s)
    sys.stdout.write(s)

print "model_%s" % model_id
with open("model_%s.txt" % model_id, "w") as f:
    output(f, "dataset: %s\n" % dataset)
    output(f, "optimizer: %s\n" % args.optimizer)
    output(f, "%s\n" % augmentation)
    print_summary(model, line_length=None, positions=None, print_fn=lambda s: output(f, s + "\n"))

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks)
