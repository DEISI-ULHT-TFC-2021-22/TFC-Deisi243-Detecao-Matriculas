# https:\\\\youtu.be\\2miw-69Xb0g

"""
@author: Sreenivas Bhattiprolu
IMAGE CLASSIFICATION USING XGBOOST by extracting features using VGG16 imagenet pretrained weights.
This code explains the process of using XGBoost for image classification
using pretrained weights (VGG16) as feature extractors.
Code last tested on: 
    Tensorflow: 2.2.0
    Keras: 2.3.1
    Python: 3.7
pip install xgboost  
    
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)   
    
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16


# Read input images and assign labels based on folder names
print(os.listdir("C:\\Users\\Diogo Pereira\\Desktop\\TFC\\modelos\\xgboost\\dataset\\"))

SIZE = 48  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for img_path in glob.glob("C:\\Users\\Diogo Pereira\\Desktop\\TFC\\modelos\\xgboost\\dataset\\train\\*.png"):
    label = img_path.split("\\")[-1].split("__")[0]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
    img = cv2.resize(img, (SIZE, SIZE))
    train_images.append(img)
    train_labels.append(label)

#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Capture test\\validation data and labels into respective lists

test_images = []
test_labels = [] 
for img_path in glob.glob("C:\\Users\\Diogo Pereira\\Desktop\\TFC\\modelos\\xgboost\\dataset\\test\\*"):
    label = img_path.split("\\")[-1].split(".png")[0]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE, SIZE))
    test_images.append(img)
    test_labels.append(label)

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)


#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

#############################
#Load model without classifier\\fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0

import pickle
with open("./config/xgboost/encoder.p", "wb") as f:
    pickle.dump(le, f)
with open("./config/xgboost/extrator.p", "wb") as f:
    pickle.dump(VGG_model, f)

#Now, let us use features from convolutional network for RF
feature_extractor=VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_training = features #This is our X input to RF

#XGBOOST
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_for_training, y_train) #For sklearn no one hot encoding

model.save_model("./config/model.txt")

#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

#Now predict using the trained RF model. 
prediction = model.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)
real = le.inverse_transform(y_test)

for i in range(len(prediction)):
    if (r := real[i]) != (p := prediction[i]):
        print(r, p)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction)
#print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction = model.predict(input_img_features)[0] 
prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])
#plt.show()