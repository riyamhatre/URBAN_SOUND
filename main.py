# Imports
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

audio_dataset_path = 'UrbanSound8K/audio/'
metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')


def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

extracted_features = []

for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
    final_class_labels = row["class"]
    final_class_folds = row["fold"]
    data = features_extractor(file_name)
    extracted_features.append([data, final_class_labels, final_class_folds])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class', 'fold'])
extracted_features_df.head()

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())
folds = np.array(extracted_features_df['fold'].tolist())

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

num_labels = y.shape[1]

# Model
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

num_epochs = 500 #100 originally
num_batch_size = 256 #32 originally

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', verbose=1, save_best_only=True)

group_kfold = GroupKFold(n_splits=10)
group_kfold.get_n_splits(X, y, folds)

accuracies = []

high_acc = 0
high_X_train = []
high_X_test = []
high_y_train = []
high_y_test = []

for train_index, test_index in group_kfold.split(X, y, folds):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

    test_accuracy = model.evaluate(X_test,y_test,verbose=0)
    print(test_accuracy[1])
    accuracies.append(test_accuracy[1])

    if test_accuracy[1] > high_acc:
        high_acc = test_accuracy[1]
        high_X_train = X_train
        high_X_test = X_test
        high_y_train = y_train
        high_y_test = y_test

model.fit(high_X_train, high_y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(high_X_test, high_y_test), callbacks=[checkpointer], verbose=1)


print("accuracies of 10-fold cross validation: " + str(accuracies))
print("highest accuracy: " + str(high_acc))