import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from tensorflow.keras import backend as K

video_processing_pipeline('21Sarah.mkv','final_Sarah.mp4')
video_processing_pipeline('Abid.mkv','final_Abid.mp4')
video_processing_pipeline('Aby.mkv','final_Aby.mp4')
video_processing_pipeline('Armando.mkv','final_Armando.mp4')
video_processing_pipeline('Brandon.mkv','final_Brandon.mp4')
video_processing_pipeline('Brian.mkv','final_Brian.mp4')
video_processing_pipeline('Caleb.mkv','final_Caleb.mp4')
video_processing_pipeline('Calvin.mkv','final_Calvin.mp4')
video_processing_pipeline('Cameron.mkv','final_Cameron.mp4')
video_processing_pipeline('Criscan.mkv','final_Criscan.mp4')
video_processing_pipeline('Darean.mkv','final_Darean.mp4')
video_processing_pipeline('Elena.mkv','final_Elena.mp4')
video_processing_pipeline('Ely.mkv','final_Ely.mp4')
video_processing_pipeline('Eric.mkv','final_Eric.mp4')
video_processing_pipeline('Gabriella.mkv','final_Gabriella.mp4')


sarah = feature_extract('final_Sarah.mp4')
abid = feature_extract('final_Abid.mp4')
aby = feature_extract('final_Aby.mp4')
armando = feature_extract('final_Armando.mp4')
brandon = feature_extract('final_Brandon.mp4')
brian = feature_extract('final_Brian.mp4')
caleb = feature_extract('final_Caleb.mp4')
calvin = feature_extract('final_Calvin.mp4')
cameron = feature_extract('final_Cameron.mp4')
criscan = feature_extract('final_Criscan.mp4')
darean = feature_extract('final_Darean.mp4')
elena = feature_extract('final_Elena.mp4')
ely = feature_extract('final_Ely.mp4')
eric = feature_extract('final_Eric.mp4')
gabriella = feature_extract('final_Gabriella.mp4')


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.cast(y_pred, 'float32') - K.cast(y_true, 'float32'))))
reshaped_feature_data = combined_dataset.reshape((13500, 5, 5 * 2048))

# Define the number of folds
k_folds = 5
target_data = pd.read_csv("target_regression.csv")
target_data = target_data.to_numpy()
# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
# Initialize lists to store results
test_losses = []
train_losses = []
val_losses = []
test_accuracies = []

# Iterate over the folds
for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(reshaped_feature_data, target_data)):
    print(f'Fold {fold + 1}/{k_folds}')
    
    X_train, X_test = reshaped_feature_data[train_indices], reshaped_feature_data[test_indices]
    y_train, y_test = target_data[train_indices], target_data[test_indices]
    print(y_train)
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(5, 5 * 2048), return_sequences=True))
    model.add(Dropout(0.6))
    model.add(BatchNormalization())
    model.add(LSTM(units=50))
    model.add(Dropout(0.6))
    model.add(BatchNormalization())
    model.add(Dense(units=1, activation='linear'))
    # Train the model with dropout and early stopping
    model.compile(optimizer='adam', loss=root_mean_squared_error,metrics=['mae', 'mse'])

    # Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # Append results to lists
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    test_losses.append(test_loss)
    
    # Store training and validation losses from the training history
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

# Calculate and print the average test loss and accuracy across folds
average_test_loss = np.mean(test_losses)
print(f'Average Test Loss: {average_test_loss}')

# Plot training, validation, and test losses
plt.figure(figsize=(12, 6))
for i in range(k_folds):
    plt.plot(train_losses[i], label=f'Training Fold {i+1}')
    plt.plot(val_losses[i], label=f'Validation Fold {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Across Folds')
plt.legend()
plt.show()
