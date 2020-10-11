!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

!mkdir -p drive
!google-drive-ocamlfuse drive

import sys
sys.path.insert(0, 'drive/')

# Commented out IPython magic to ensure Python compatibility.
# Importing libraries
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import CSVLogger
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import utils as np_utils
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import numpy
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, Activation,BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical
from tqdm import tqdm
import pandas as pd
import keras
import numpy as np
import scipy
import sklearn
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.layers import BatchNormalization
from tensorflow.keras.layers import Embedding
#from keras.utils import np_utils
#from tensorflow.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.layers import LeakyReLU
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from __future__ import print_function
import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input
from tensorflow.keras import backend as K
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, ReLU
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
import seaborn as sns
from  tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.models import Model
import numpy
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
GOOGLE_COLAB = True
TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
TEST_FILE = "test_file.txt"
MODEL_FILE = "model.h0"
if GOOGLE_COLAB:
    !pip install livelossplot
from livelossplot import PlotLossesKeras

!unzip drive/c_b_covid_fold5

training_data_dir ="c_b_covid_fold5/train" #80%
validation_data_dir = "c_b_covid_fold5/test" #20%

# Hyperparams
IMAGE_SIZE = 224
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS =30
BATCH_SIZE =3
TEST_SIZE = 3
learning = 0.00001

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

#InceptionV3

#import libraries
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
#import numpy
#from sklearn.metrics import confusion_matrix
#from tensorflow.keras.layers import Dense, Dropout, Activation,BatchNormalization
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#import matplotlib.pyplot as plt
#import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
#%matplotlib inline

#load pre trained Xception model
base_model=tf.keras.applications.inception_v3.InceptionV3(weights='imagenet',include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
 #and a logistic layer -- let's say we have 200 classes

predictions = Dense(2, activation='softmax')(x)

#x=Dense(4)(x)
#Activation(tf.nn.softmax)
#predictions = x 
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=learning, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy',metrics=['accuracy'])
#Summary of Xception Model

import warnings
warnings.filterwarnings("ignore")

model.summary()

# Commented out IPython magic to ensure Python compatibility.
#ResNet50

#import libraries

#import libraries
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
#import numpy
#from sklearn.metrics import confusion_matrix
#from tensorflow.keras.layers import Dense, Dropout, Activation,BatchNormalization
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#import matplotlib.pyplot as plt
#import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
# %matplotlib inline

#load pre trained Xception model
base_model=tf.keras.applications.ResNet50(weights='imagenet',include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
 #and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

#x=Dense(4)(x)
#Activation(tf.nn.softmax)(x)
#predictions = x 

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=learning, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy',metrics=['accuracy'])
#Summary of Xception Model

import warnings
warnings.filterwarnings("ignore")

model.summary()

# Commented out IPython magic to ensure Python compatibility.
#ResNet101

#import libraries
from tensorflow.python.keras.applications.resnet import ResNet101
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
# %matplotlib inline

#load pre trained Xception model
base_model=tf.keras.applications.resnet.ResNet101(include_top=True, weights='imagenet')

# add a global spatial average pooling layer
x = base_model.output
x = Flatten()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
 #and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

#x=Dense(4)(x)
#Activation(tf.nn.softmax)(x)
#predictions = x 

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=learning, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy',metrics=['accuracy'])
#Summary of Xception Model

import warnings
warnings.filterwarnings("ignore")

model.summary()

# Commented out IPython magic to ensure Python compatibility.
#ResNet152

#import libraries
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
# %matplotlib inline

#load pre trained Xception model
base_model=tf.keras.applications.resnet.ResNet152(include_top=True, weights='imagenet')

# add a global spatial average pooling layer
x = base_model.output
x = Flatten()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
 #and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

#x=Dense(4)(x)
#Activation(tf.nn.softmax)(x)
#predictions = x 

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=learning, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy',metrics=['accuracy'])
#Summary of Xception Model

import warnings
warnings.filterwarnings("ignore")

model.summary()

# Commented out IPython magic to ensure Python compatibility.
#InceptionResNetV2

#import libraries
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
# %matplotlib inline

#load pre trained Xception model
base_model=InceptionResNetV2(weights='imagenet',include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
 #and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=learning, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy',metrics=['accuracy'])
#Summary of Xception Model

import warnings
warnings.filterwarnings("ignore")

model.summary()

with open(MODEL_SUMMARY_FILE,"w") as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))

# Data augmentation
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

# Data preparation
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical")
validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False)

from tensorflow.keras.callbacks import TensorBoard
  csv_logger = CSVLogger('training_log.csv')
  import warnings
warnings.filterwarnings("ignore")

# Training
H = model.fit_generator(
    training_generator,
    steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // BATCH_SIZE,
    callbacks=[csv_logger])

#model.save_weights(MODEL_FILE)

N = EPOCHS
plt.style.use("seaborn-white")
plt.figure(dpi=600)
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="test_loss")
plt.title("Training Loss",fontsize=14)
plt.xlabel("Epoch",fontsize=14)
plt.ylabel("Loss",fontsize=14)
plt.legend(loc="upper right",fontsize=14)
plt.savefig("plot.png")

plt.style.use("seaborn-white")
plt.figure(dpi=600)
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="test_acc")
plt.title("Training Accuracy",fontsize=14)
plt.xlabel("Epoch",fontsize=14)
plt.ylabel("Accuracy",fontsize=14)
plt.legend(loc="lower right",fontsize=14)
plt.savefig("plot.png")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

LABELS = ["Bacterial","Covid19"]

def show_confusion_matrix(validations, predictions):
    matrix = confusion_matrix(validations, predictions)
    plt.figure(figsize=(8, 8), dpi=600)
    sn.set(font_scale=1.6)#for label size
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False)    

filenames = validation_generator.filenames
nb_samples = len(filenames)

Y_pred = model.predict_generator(validation_generator,(nb_samples//BATCH_SIZE)+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
show_confusion_matrix(validation_generator.classes, y_pred)
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names =["covid","Bacterial"]
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
sn.set(font_scale=1.4)#for label size

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from itertools import cycle
plt.style.use("seaborn-white")
y_test = label_binarize(validation_generator.classes, classes=[0, 1])
y_pred= label_binarize(y_pred, classes=[0, 1])
# Plot linewidth.
lw = 2
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(1):
   fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
   roc_auc[i] = auc(fpr[i], tpr[i])

plt.style.use("seaborn-white")

plt.figure(dpi=600)
lw = 2
plt.plot(fpr[i], tpr[i], color='darkorange',
       lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.legend(loc="lower right",fontsize=14)
plt.show()


plt.figure(2)
colors = cycle(['darkmagenta', 'darkorange', 'darkblue'])
for i, color in zip(range(1), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.01, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
