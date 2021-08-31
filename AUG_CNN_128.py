
"""
Created on Wed Jan 13 01:20:21 2021

@author: adham
"""
import os

#print("The number of MRI Images labelled 'yes':",len(os.listdir('yes')))
#print("The number of MRI Images labelled 'no':",len(os.listdir('no')))

import tensorflow as tf
from zipfile import ZipFile
import os,glob
import cv2
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense,MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import matplotlib.pyplot as plt
#print(os.getcwd())




# print("X_train Shape: ", X_train.shape)
# print("X_test Shape: ", X_test.shape) 
# print("y_train Shape: ", y_train.shape)
# print("y_test Shape: ", y_test.shape)

## build CNN Model 
m1=Sequential()
m1.add(BatchNormalization(input_shape = (128,128,3)))
m1.add(Convolution2D(32, (3,3), activation ='relu', input_shape = (128, 128, 3))) 
m1.add(MaxPooling2D(pool_size=2))
m1.add(Convolution2D(filters=64, kernel_size=4, padding='same', activation='relu'))
m1.add(MaxPooling2D(pool_size=2))
m1.add(Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'))
m1.add(MaxPooling2D(pool_size=2))
m1.add(Convolution2D(filters=128, kernel_size=2, padding='same', activation='relu'))
m1.add(MaxPooling2D(pool_size=2))
m1.add(Dropout(0.25))
m1.add(Flatten()) 

m1.add(Dense(units=128,activation = 'relu'))
m1.add(Dense(units = 64, activation = 'relu'))
m1.add(Dense(units = 32, activation = 'relu'))
m1.add(Dense(units = 2, activation = 'softmax'))


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory('data/train',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical'
                                            ,seed=123
                                            )


test_set = test_datagen.flow_from_directory( 'data/test',
                                            target_size=(128, 128),
                                            batch_size=16,
                                            class_mode='categorical'
                                            ,seed=123
                                            )


# compile model


m1.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])


    
m1.fit_generator(
        training_set,
        steps_per_epoch=100,
        nb_epoch=50,
        validation_data=test_set,
        nb_val_samples=25)



m1.save("cnn_128_modelSaved.h5")

import os
print(os.getcwd())


os.chdir('test/yes')

X_test = []
y_test = []
for i in tqdm(os.listdir()):
      img = cv2.imread(i)   
      img = cv2.resize(img,(128,128))
      X_test.append(img)
      y_test.append((i[0:1]))

os.chdir("../")

print(os.getcwd())

os.chdir('no')

for i in tqdm(os.listdir()):
      img = cv2.imread(i)   
      img = cv2.resize(img,(128,128))
      X_test.append(img)
      y_test.append('N')

os.chdir("../")      
os.chdir("../")      

X_test = np.array(X_test)


le = preprocessing.LabelEncoder()
y_test = le.fit_transform(y_test)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
y_test = np.array(y_test)

y_test=np.argmax(y_test,axis=1)

from sklearn.metrics import accuracy_score, confusion_matrix

# validate on val set
predictions = m1.predict(X_test)


predictions=np.argmax(predictions,axis=1)

accuracy = accuracy_score(y_test, predictions)
print('Val Accuracy = %.2f' % accuracy)

confusion_mtx = confusion_matrix(y_test, predictions) 




# # store images in list

# os.chdir('yes')
# X_32 = []
# y_32 = []
# for i in tqdm(os.listdir()):
#       img = cv2.imread(i)   
#       img = cv2.resize(img,(32,32))
#       X_32.append(img)
#       y_32.append((i[0:1]))

# os.chdir("../")      

# #print(os.getcwd())

# os.chdir('no')
# for i in tqdm(os.listdir()):
#       img = cv2.imread(i)   
#       img = cv2.resize(img,(32,32))
#       X_32.append(img)
# for i in range(1,99):
#     y_32.append('N')



############# show images stored in plots
    

# # split into train and test
# X_train_32, X_test_32, y_train_32, y_test_32 = train_test_split(X_32, y_32, test_size=0.2, random_state=42)
# print ("Shape of an image in X_train: ", X_train_32[0].shape)
# print ("Shape of an image in X_test: ", X_test_32[0].shape)
    

# # convert them into categorical[0,1] 0->no , 1->yes , and transform into numpy arrays
    
# le = preprocessing.LabelEncoder()
# y_train_32 = le.fit_transform(y_train_32)
# y_test_32 = le.fit_transform(y_test_32)
# y_train_32 = tf.keras.utils.to_categorical(y_train_32, num_classes=2)
# y_test_32 = tf.keras.utils.to_categorical(y_test_32, num_classes=2)
# y_train_32 = np.array(y_train_32)
# X_train_32 = np.array(X_train_32)
# y_test_32 = np.array(y_test_32)
# X_test_32 = np.array(X_test_32) 
    





# y_predicted_32=m1.predict_classes(X_test_32)
# y_actual_32=np.argmax(y_test_32,axis=1)

# from sklearn.metrics import confusion_matrix
# cm_32 = confusion_matrix(y_actual_32, y_predicted_32)
 
# acc_32 = ((cm_32[0][0]+cm_32[1][1])/y_predicted_32.size)*100


#128&128 plots


# L = 2
# W = 2
# fig, axes = plt.subplots(L, W, figsize = (12,12))
# axes = axes.ravel()
# for i in np.arange(0, L * W):  
#     axes[i].imshow(X_test[i])
#     axes[i].set_title(f"Prediction Class = {y_predicted[i]:0.1f}\n Actual Label = {y_actual[i]:0.1f}")
#     axes[i].axis('off')
# plt.subplots_adjust(wspace=0.5)

#%matplotlib inline
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))
# for i in range(4):
#     plt.subplot(1, 4, i+1)
#     plt.imshow(X[i], cmap="gray")
#     plt.axis('off')
# plt.show()



# 32x32


# L = 2
# W = 2
# fig, axes = plt.subplots(L, W, figsize = (12,12))
# axes = axes.ravel()
# for i in np.arange(0, L * W):  
#     axes[i].imshow(X_test[i])
#     axes[i].set_title(f"Prediction Class = {y_predicted[i]:0.1f}\n Actual Label = {y_actual[i]:0.1f}")
#     axes[i].axis('off')
# plt.subplots_adjust(wspace=0.5)


#%matplotlib inline
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))
# for i in range(4):
#     plt.subplot(1, 4, i+1)
#     plt.imshow(X[i], cmap="gray")
#     plt.axis('off')
# plt.show()



















