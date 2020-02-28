# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:32:20 2020

@author: fang
"""

# In[]
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator ,load_img
from sklearn.model_selection import train_test_split
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2                                
import pandas as pd                    
                    
                     
# In[]
file_path ='C:\\MyProject\\Tensorlfow\\Cat Dog Images Classificaiton\\data\\train\\'
file_names_train = os.listdir('C:\\MyProject\\Tensorlfow\\Cat Dog Images Classificaiton\\data\\train')

# In[]
def get_categories(file_names):
    categories = []
    for filename in file_names:
        category = filename.split('.')[0]
        if category =='dog':
            categories.append(1)
        else:
            categories.append(0)  
    df = pd.DataFrame({
        'filename' : file_names,
        'category' : categories
        })
    return df

df_train = get_categories(file_names_train)

# In[]
'''
參數設定
'''
batch_size = 128
epochs = 15
IMG_HEIGHT = 227
IMG_WIDTH = 227

# In[]
sample = random.choice(file_names_train)
image = load_img(file_path+sample)
plt.imshow(image)
# In[]


# In[]
df_train['category'] = df_train['category'].replace({0:'cat',1:'dog'})

train_df, validate_df = train_test_split(df_train, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

# In[]

'''
Alex-module
'''

 
model = Sequential()
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

# In[]
train_image_generator =ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
train_generator  = train_image_generator.flow_from_dataframe(
    train_df,
    file_path,
    x_col='filename',
    y_col='category',
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    class_mode='categorical',
    batch_size=batch_size
    )
# In[]
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    file_path, 
    x_col='filename',
    y_col='category',
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    class_mode='categorical',
    batch_size=batch_size
)
# In[]
history_alex = model.fit(
    train_generator,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size
    
)

# In[]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history_alex.history['loss'], color='b', label="Training loss")
ax1.plot(history_alex.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history_alex.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history_alex.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
# In[]
test_file_path ='C:\\MyProject\\Tensorlfow\\Cat Dog Images Classificaiton\\data\\test1\\'
test_filenames = os.listdir("C:\\MyProject\\Tensorlfow\\Cat Dog Images Classificaiton\\data\\test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
# In[]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    test_file_path, 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=batch_size,
    shuffle=False
)
# In[]
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

# In[]
test_df['category'].value_counts().plot.bar()

# In[]
'''
show predict plots
'''
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(test_file_path+filename, target_size=(IMG_HEIGHT,IMG_WIDTH))
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()

# In[]
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission0228.csv', index=False)



