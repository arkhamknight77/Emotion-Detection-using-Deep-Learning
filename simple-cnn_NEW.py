#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
import cv2
import math
import numpy as np
from collections import OrderedDict
import scikitplot
import seaborn as sns
from matplotlib import pyplot

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

from keras import backend as K
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


np.random.seed(42)


# In[3]:


INPUT_PATH = "../input/ck48-5-emotions/CK+48/"

total_images = 0
for dir_ in os.listdir(INPUT_PATH):
    count = 0
    for f in os.listdir(INPUT_PATH + dir_ + "/"):
        count += 1
    total_images += count
    print(f"{dir_} has {count} number of images")
    
print(f"\ntotal images: {total_images}")


# `sadness` and `fear` has very low number of images as compared to other classes

# In[4]:


TOP_EMOTIONS = ["happy", "surprise", "anger", "sadness", "fear"]


# In[5]:


img_arr = np.empty(shape=(total_images, 48, 48, 1))
img_label = np.empty(shape=(total_images))
label_to_text = {}

idx = 0
label = 0
for dir_ in os.listdir(INPUT_PATH):
    if dir_ in  TOP_EMOTIONS:
        for f in os.listdir(INPUT_PATH + dir_ + "/"):
            img_arr[idx] = np.expand_dims(cv2.imread(INPUT_PATH + dir_ + "/" + f, 0), axis=2)
            img_label[idx] = label
            idx += 1
        label_to_text[label] = dir_
        label += 1

img_label = np_utils.to_categorical(img_label)

img_arr.shape, img_label.shape, label_to_text


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label, train_size=0.7, stratify=img_label, shuffle=True, random_state=42)
X_train.shape, X_test.shape


# In[7]:


fig = pyplot.figure(1, (10,10))

idx = 0
for k in label_to_text:
    sample_indices = np.random.choice(np.where(y_train[:,k]==1)[0], size=5, replace=False)
    sample_images = X_train[sample_indices]
    for img in sample_images:
        idx += 1
        ax = pyplot.subplot(5,5,idx)
        ax.imshow(img.reshape(48,48), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label_to_text[k])
        pyplot.tight_layout()


# In[8]:


# data normalization
X_train = X_train / 255.
X_test = X_test / 255.


# In[9]:


def build_dcnn(input_shape, num_classes):
    model_in = Input(shape=input_shape, name="input")
    
    conv2d_1 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_1'
    )(model_in)
    batchnorm_1 = BatchNormalization(name='batchnorm_1')(conv2d_1)
    conv2d_2 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_2'
    )(batchnorm_1)
    batchnorm_2 = BatchNormalization(name='batchnorm_2')(conv2d_2)
    
    maxpool2d_1 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_1')(batchnorm_2)
    dropout_1 = Dropout(0.3, name='dropout_1')(maxpool2d_1)

    conv2d_3 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_3'
    )(dropout_1)
    batchnorm_3 = BatchNormalization(name='batchnorm_3')(conv2d_3)
    conv2d_4 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_4'
    )(batchnorm_3)
    batchnorm_4 = BatchNormalization(name='batchnorm_4')(conv2d_4)
    
    maxpool2d_2 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_2')(batchnorm_4)
    dropout_2 = Dropout(0.3, name='dropout_2')(maxpool2d_2)

    conv2d_5 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_5'
    )(dropout_2)
    batchnorm_5 = BatchNormalization(name='batchnorm_5')(conv2d_5)
    conv2d_6 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_6'
    )(batchnorm_5)
    batchnorm_6 = BatchNormalization(name='batchnorm_6')(conv2d_6)
    
    maxpool2d_3 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_3')(batchnorm_6)
    dropout_3 = Dropout(0.3, name='dropout_3')(maxpool2d_3)

    flatten = Flatten(name='flatten')(dropout_3)
    
    dense_1 = Dense(
        128,
        activation='elu',
        kernel_initializer='he_normal',
        name='dense1'
    )(flatten)
    batchnorm_7 = BatchNormalization(name='batchnorm_7')(dense_1)
    dropout_4 = Dropout(0.4, name='dropout_4')(batchnorm_7)

    model_out = Dense(
        num_classes,
        activation='softmax',
        name='out_layer'
    )(dropout_4)

    model = Model(inputs=model_in, outputs=model_out, name="DCNN")
    
    return model


# In[10]:


INPUT_SHAPE = (48, 48, 1)
optim = optimizers.Adam(0.001)

model = build_dcnn(input_shape=(48,48,1), num_classes=len(label_to_text))
model.compile(
        loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy']
)

plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True, dpi=50, to_file='model.png')


# In[11]:


early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00008,
    patience=12,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    min_delta=0.0001,
    factor=0.4,
    patience=6,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]

batch_size = 10
epochs = 60


# In[12]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
)
train_datagen.fit(X_train)


# In[13]:


history = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    use_multiprocessing=True
)


# In[14]:


sns.set()
fig = pyplot.figure(0, (12, 4))

ax = pyplot.subplot(1, 2, 1)
sns.lineplot(history.epoch, history.history['accuracy'], label='train')
sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
pyplot.title('Accuracy')
pyplot.tight_layout()

ax = pyplot.subplot(1, 2, 2)
sns.lineplot(history.epoch, history.history['loss'], label='train')
sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
pyplot.title('Loss')
pyplot.tight_layout()

pyplot.savefig('epoch_history.png')
pyplot.show()


# In[15]:


label_to_text


# In[16]:


text_to_label = dict((v,k) for k,v in label_to_text.items())
text_to_label


# In[17]:


yhat_test = model.predict(X_test)
yhat_test = np.argmax(yhat_test, axis=1)
ytest_ = np.argmax(y_test, axis=1)

scikitplot.metrics.plot_confusion_matrix(ytest_, yhat_test, figsize=(7,7))
pyplot.savefig("confusion_matrix_model3pipes.png")

test_accu = np.sum(ytest_ == yhat_test) / len(ytest_) * 100
print(f"test accuracy: {round(test_accu, 4)} %\n\n")

print(classification_report(ytest_, yhat_test))


# In[18]:


layer_list = [(layer.name, layer) for layer in model.layers if "conv" in layer.name]
layer_list


# In[23]:


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# In[24]:


# Saving Model weights and json
model_json_file = './model.json'
model_weights_file = './model.h5'
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)


# In[ ]:


# default harcascade 
face_cascade = cv2.CascadeClassifier('../input/haarcascade-frontalface-defaultxml/haarcascade_frontalface_default.xml')


# In[ ]:


get_ipython().run_line_magic('run', "-i './scripts/enable_widgets.py'")


# In[ ]:





# ## Let's Visualize What our CNN Learned

# #### Plot Filters

# In[ ]:


INTERESTED_CONV_LAYERS = ["conv2d_1", "conv2d_4"]


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nidx = 1\nfor layer in layer_list:\n    if layer[0] in INTERESTED_CONV_LAYERS:        \n        layer_output = layer[1].output\n        filters, bias = layer[1].get_weights()     \n        filters = (filters - filters.min()) / (filters.max() - filters.min())\n    \n        cols = 20\n        rows = math.ceil(filters.shape[-1] / cols)\n        fig = pyplot.figure(i, (20, rows))\n\n        idx += 1\n        for i,f in enumerate(np.rollaxis(filters, 3)):\n            ax = pyplot.subplot(rows, cols, i+1)\n            f = np.mean(f, axis=2)\n            ax.imshow(f, cmap="viridis")\n            ax.set_xticks([])\n            ax.set_yticks([])\n            pyplot.suptitle(f"layer name: {layer[0]}, {filters.shape[3]} filters of shape {filters.shape[:-1]}", fontsize=20, y=1.1)\n            pyplot.tight_layout()')


# #### Plot Feature Maps

# In[ ]:


sns.reset_orig()
sample_img = X_test[0] # select a random image
pyplot.imshow(sample_img.reshape(48,48), cmap="gray")
pyplot.show()


# In[ ]:


sample_img = np.expand_dims(sample_img, axis=0)
sample_img.shape


# In[ ]:


INTERESTED_CONV_LAYERS = ["conv2d_1", "conv2d_4"]


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ni = 1\nfor layer in layer_list:\n    if layer[0] in INTERESTED_CONV_LAYERS:    \n        model_conv2d = Model(inputs=model.inputs, outputs=layer[1].output)\n        featuremaps_conv2d = model_conv2d.predict(sample_img)\n\n        cols = 20\n        rows = math.ceil(featuremaps_conv2d.shape[-1] / cols)\n        fig = pyplot.figure(i, (20, rows))        \n        i += 1\n        \n        for idx, feature_map in enumerate(np.rollaxis(featuremaps_conv2d, axis=3)):\n            ax = pyplot.subplot(rows, cols ,idx+1)\n            ax.imshow(feature_map[0], cmap="viridis")\n            ax.set_xticks([])\n            ax.set_yticks([])\n            pyplot.suptitle(f"layer name: {layer[0]}, feature map shape: {featuremaps_conv2d.shape}", fontsize=20, y=1.1)\n            pyplot.tight_layout()')


# We can see that the first convolutional layer has detected many edges in the image and indeed this is what we expected.

# #### Plot Class Activation Map (CAM) 
# The following CAM is taken from [here](https://github.com/himanshurawlani/convnet-interpretability-keras/blob/master/Visualizing%20heatmaps/heatmap_visualization_using_gradcam.ipynb).

# In[ ]:


INTERESTED_CONV_LAYERS = ["conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5", "conv2d_6"]


# In[ ]:


preds = model.predict(sample_img)
label_to_text[np.argmax(preds[0])]


# In[ ]:


pred_vector_output = model.output[:, np.argmax(preds[0])]
pred_vector_output


# In[ ]:


heatmaps = []

for layer in layer_list:
    if layer[0] in INTERESTED_CONV_LAYERS:
        some_conv_layer = model.get_layer(layer[0])
        grads = K.gradients(pred_vector_output, some_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, some_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([sample_img])

        for i in range(model.get_layer(layer[0]).output_shape[-1]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmaps.append(np.mean(conv_layer_output_value, axis=-1))


# In[ ]:


fig = pyplot.figure(figsize=(14, 3))

for i, (name,hm) in enumerate(zip(INTERESTED_CONV_LAYERS, heatmaps)):
    ax = pyplot.subplot(1, 6, i+1)
    img_heatmap = np.maximum(hm, 0)
    img_heatmap /= np.max(img_heatmap)
    ax.imshow(img_heatmap, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.title(name)
    pyplot.tight_layout()


# We can see that our model focuses on important aspects of the image i.e., `lips`, `eyes` and `eyebrows`.

# In[ ]:


fig = pyplot.figure(figsize=(14, 3))

for i, (name,hm) in enumerate(zip(INTERESTED_CONV_LAYERS, heatmaps)):
    img_heatmap = np.maximum(hm, 0)
    img_heatmap /= np.max(img_heatmap)
    
    img_hm = cv2.resize(img_heatmap, (48,48))
    img_hm = np.uint8(255 * img_hm)

    img_hm = cv2.applyColorMap(img_hm, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = img_hm * 0.4 + sample_img

    ax = pyplot.subplot(1, 6, i+1)
    ax.imshow(superimposed_img[0,:,:])
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.title(name)
    pyplot.tight_layout()


# Infact this all can be done just using few lines of code using some interesting libraries like `keract`, `keras-viz`, `lucid`. Below I showed keract

# In[ ]:


get_ipython().system(' pip install keract')


# In[ ]:


from keract import get_activations, display_activations, display_heatmaps


# In[ ]:


activations = get_activations(model, sample_img)
activations.keys()


# As there are many layers in the model and visualizing all of them is not very informative, so I just plotted the interested ones i.e., conv layers

# In[ ]:


INTERESTED_CONV_LAYERS = ["conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5", "conv2d_6", "dense1", "out_layer"]

activations_convs = OrderedDict()

for k,v in activations.items():
    if k in INTERESTED_CONV_LAYERS:
        activations_convs[k] = v
        
activations_convs.keys()


# In[ ]:


display_activations(activations_convs, save=False)


# In[ ]:


display_heatmaps(activations_convs, sample_img, save=False)


# **If you like this notebook then please upvote and share with others and also see my other kernels on [EDA](https://www.kaggle.com/gauravsharma99/eda-on-mpg-data), [Statistical Analysis](https://www.kaggle.com/gauravsharma99/statistical-analysis-on-mpg-data), [NSL](https://www.kaggle.com/gauravsharma99/neural-structured-learning), [etc.](https://www.kaggle.com/notebooks)**

# In[ ]:




