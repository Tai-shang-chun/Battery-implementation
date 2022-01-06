import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import netron
from sklearn import preprocessing
import matplotlib.pyplot as plt

def norm(data):
    a= preprocessing.StandardScaler().fit(data)
    d=a.transform(data)
    m=a.mean_
    s=a.scale_
    return d,m,s,a

def rmse(x,y):
    return np.sqrt(np.mean((x-y)**2))


d = layers.Input(shape=(500, 4))
conv1 = layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu)(d)
conv2 = layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu)(conv1)
conv2_1= layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu)(conv2)
conv2_2 = layers.Conv1D(32,5, padding='same', activation= tf.nn.relu)(conv2_1)
maxpooling1d = layers.MaxPooling1D(pool_size=2, strides=2)
concat1 = layers.Concatenate(axis=2)([layers.MaxPooling1D(pool_size=2, strides=2)(conv2_2), layers.MaxPooling1D(pool_size=2, strides=2)(conv2)])
maxpooling1d1 = layers.MaxPooling1D(pool_size=2, strides=2)(concat1)
SpatialMaxPooling1D1 = layers.SpatialDropout1D(0.16)(maxpooling1d1)
conv3 = layers.Conv1D(32, 11, padding='same', activation= tf.nn.relu)(SpatialMaxPooling1D1)
conv4 = layers.Conv1D(32, 11, padding='same', activation=tf.nn.relu)(conv3)
conv4_1 = layers.Conv1D(64, 7, padding='same', activation=tf.nn.relu)(conv4)
conv4_2 = layers.Conv1D(64, 7, padding='same', activation=tf.nn.relu)(conv4_1)
concat2 = layers.Concatenate(axis=2)([layers.MaxPooling1D(pool_size=2, strides=2)(conv4_2), layers.MaxPooling1D(pool_size=2, strides=2)(conv4)])
avgpooling1d = layers.AveragePooling1D(pool_size=2, strides=2)(concat2)
conv5 = layers.Conv1D(256, 7, padding='same', activation=tf.nn.relu)(avgpooling1d)
conv6 = layers.Conv1D(256, 7, padding='same', activation=tf.nn.relu)(conv5)
conv6_1 = layers.Conv1D(256, 5, padding='same', activation=tf.nn.relu)(conv6)
conv6_2 = layers.Conv1D(256, 5, padding='same', activation=tf.nn.relu)(conv6_1)
concat3 = layers.Concatenate(axis=2)([conv6_2, conv6])
SpatialMaxPooling1D2 = layers.SpatialDropout1D(0.16)(concat3)
globalavgpooling1d = layers.GlobalAveragePooling1D()(SpatialMaxPooling1D2)
globalmaxpooling1d = layers.GlobalMaxPooling1D()(SpatialMaxPooling1D2)
print('globalavgpooling1d shape',globalavgpooling1d.shape)
print('globalmaxpooling1d shape',globalmaxpooling1d.shape)
fc = layers.Dense(1)(layers.Add()([globalavgpooling1d, globalmaxpooling1d]))
model = Model(d,fc)

# model.summary()

# path = './h5_model.h5'
# tf.keras.models.save_model(model,path)
# netron.start(path)


# model_dir='pretrained'
data_dir='dataset'
eol_data = np.load('%s/battery_EoL.npy'%(data_dir),allow_pickle='TRUE')
discharge_data=np.load('%s/discharge_data.npy'%(data_dir),allow_pickle='TRUE').tolist()
index=np.load('%s/index_battery.npy'%(data_dir))
print('eol data',eol_data)
print('index ',index)


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
np.save('./get81battery/discharge_norm.npy', np.vstack([x_train_mean, x_train_std]))
print('mean std shape', x_train_mean[np.newaxis,:].shape, x_train_std.shape)
x_train_norm = np.concatenate((x_train_mean[np.newaxis,:], x_train_std[np.newaxis,:]),axis=0)
print('x_train norm shape', x_train_norm.shape)
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
history=model.fit(x_train, y_train, epochs=EPOCH, batch_size=256, shuffle=True,validation_data=(x_test, y_test), verbose=2)
print(history.history['loss'])
print(history.history['val_loss'])

pred1 = model.predict(x_test)
print('pred1', pred1*y_train_std+y_train_mean)
print('rmse', rmse(pred1*y_train_std+y_train_mean, y_test*y_train_std+y_train_mean))
model.save('./get81battery/Dimreduction1.h5')
plt.plot(np.arange(EPOCH), history.history['loss'], label= 'train')
plt.plot(np.arange(EPOCH), history.history['val_loss'], label= 'val')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.title('ft_sel1 mse loss')
plt.legend()
plt.show()
