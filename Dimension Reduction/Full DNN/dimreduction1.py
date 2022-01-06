import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import netron
from sklearn import preprocessing
from tensorflow.python.keras.backend import dot
from tqdm import tqdm
import pickle


def norm(data):
    a= preprocessing.StandardScaler().fit(data)
    d=a.transform(data)
    m=a.mean_
    s=a.scale_
    return d,m,s,a

def rmse(x,y):
    return np.sqrt(np.mean((x-y)**2))

d = layers.Input(shape = (500,4))
conv1 = layers.Conv1D(128, 13, padding='same', activation=tf.nn.relu)(d)
conv2 = layers.Conv1D(128, 13, padding='same', activation=tf.nn.relu)(conv1)
maxpooling1d1_1 = layers.MaxPooling1D(pool_size=2, strides=2)(conv2)
conv3 = layers.Conv1D(384, 5, padding='same', activation=tf.nn.relu)(conv2)
conv4 = layers.Conv1D(384, 5, padding='same', activation=tf.nn.relu)(conv3)
maxpooling1d1_2 = layers.MaxPooling1D(pool_size=2, strides=2)(conv4)
concat1 = layers.Concatenate(axis=2)([maxpooling1d1_1, maxpooling1d1_2])
maxpooling1d2 = layers.MaxPooling1D(pool_size=2, strides=2)(concat1)
spatialdropout1d1 = layers.SpatialDropout1D(0.16)(maxpooling1d2)
conv5 = layers.Conv1D(64, 17, padding='same', activation=tf.nn.relu)(spatialdropout1d1)
conv6 = layers.Conv1D(64, 17, padding='same', activation=tf.nn.relu)(conv5)
avgpooling1d1_1 = layers.AveragePooling1D(pool_size=2, strides=2)(conv6)
conv7 = layers.Conv1D(192, 3, padding='same', activation=tf.nn.relu)(conv6)
conv8 = layers.Conv1D(192, 3, padding='same', activation=tf.nn.relu)(conv7)
avgpooling1d1_2 = layers.AveragePooling1D(pool_size=2, strides=2)(conv8)
concat2 = layers.Concatenate(axis=2)([avgpooling1d1_1, avgpooling1d1_2])
maxpooling1d3 = layers.MaxPooling1D(pool_size=2, strides=2)(concat2)
conv9 = layers.Conv1D(32, 13, padding='same', activation=tf.nn.relu)(maxpooling1d3)
conv10 = layers.Conv1D(32, 13, padding='same', activation=tf.nn.relu)(conv9)
conv11 = layers.Conv1D(32, 3, padding='same', activation=tf.nn.relu)(conv10)
conv12 = layers.Conv1D(32, 3, padding='same', activation=tf.nn.relu)(conv11)
concat3 = layers.Concatenate(axis= 2)([conv10, conv12])
spatialdropout1d2 = layers.SpatialDropout1D(0.16)(concat3)
globalavgpooling1d1 = layers.GlobalAveragePooling1D()(spatialdropout1d2)
globalmaxpooling1d1 = layers.GlobalMaxPooling1D()(spatialdropout1d2)
add = layers.Add()([globalavgpooling1d1, globalmaxpooling1d1])
fc = layers.Dense(1)(add)
model = Model(d, fc)

# path = './model/Full DNN/dimension_reduction1.h5'
# tf.keras.models.save_model(model,path)
# netron.start(path)

data_dir='dataset'
eol_data = np.load('%s/battery_EoL.npy'%(data_dir),allow_pickle='TRUE')
discharge_data=np.load('%s/discharge_data.npy'%(data_dir),allow_pickle='TRUE').tolist()
index=np.load('%s/index_battery.npy'%(data_dir))

train_idx = index[17:]
test_idx = index[:17]
x_train = np.zeros((0, 4, 500))
y_train = np.zeros(0)
for i in train_idx:
    cycles = len(discharge_data[i])
    # print('discharge data shape', discharge_data[i][:100].shape)
    x_train = np.concatenate((x_train, discharge_data[i][:100]), axis=0)
    y_train = np.append(y_train, np.full(100, eol_data[i]), axis=0)
print('x_train len', x_train.shape)
print('y_train shape', y_train.shape)

x_test = np.zeros((0, 4, 500))
y_test = np.zeros(0)
for i in test_idx:
    cycles = len(discharge_data[i])
    x_test = np.concatenate((x_test, discharge_data[i][:100]), axis=0)
    y_test = np.append(y_test, np.full(100, eol_data[i]), axis=0)
print('x_test shape',x_test.shape)
print('y_test shape', y_test.shape)


x_train, x_train_mean, x_train_std, x_fit = norm(np.transpose(x_train, (0,2,1)).reshape(-1,4))
print('discharge norm', x_train_mean, x_train_std)
print(np.vstack([x_train_mean, x_train_std]))
np.save('./Full DNN/discharge_norm.npy', np.vstack([x_train_mean, x_train_std]))
x_train = x_train.reshape(-1, 500, 4)

y_train, y_train_mean, y_train_std, y_fit = norm(y_train.reshape(-1,1))
print('y_train_mean', y_train_mean)

x_test = x_fit.transform(np.transpose(x_test, (0,2,1)).reshape(-1, 4))
x_test = x_test.reshape(-1, 500, 4)
y_test = y_fit.transform(y_test.reshape(-1,1))

opt = tf.keras.optimizers.Adam(learning_rate=0.0005, amsgrad=True)

model.compile(optimizer = opt, loss = 'mse', metrics=[tf.keras.metrics.MeanSquaredError()])

#優化方式
EPOCH = 200
history=model.fit(x_train, y_train, epochs=EPOCH, batch_size=128, shuffle=True,validation_data=(x_test, y_test), verbose=2)
print(history.history['loss'])
print(history.history['val_loss'])

pred1 = model.predict(x_test)
print('pred1', pred1*y_train_std+y_train_mean)
print('rmse', rmse(pred1*y_train_std+y_train_mean, y_test*y_train_std+y_train_mean))
model.save('./Full DNN/Dimreduction1.h5')
plt.plot(np.arange(EPOCH), history.history['loss'], label= 'train')
plt.plot(np.arange(EPOCH), history.history['val_loss'], label= 'val')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.title('ft_sel1 mse loss')
plt.legend()
plt.show()
