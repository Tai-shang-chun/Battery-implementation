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

def re_norm(cell_feature):
    log1=[]
    log2=[]
    log3=[]
    for i in tqdm(range(len(cell_feature))):
        len_=len(cell_feature[i])-100
        for k in range(len_):
            for j in range(0,50,1):         
                # record rul data
                log1.append(np.float32(eol_data[i]-k))
                # record charge time
                log2.append(chargetime[i])
                # record shift cycle | [0,0,0...1,1,1...,2,2,2...,,,EOL-100-1,EOL-100-1,...] $$ len= EOL-100   
                log3.append(np.float32(k))
    # all batteries all shift and RUL cycles (81, 50*(EOL-100))
    print('log1 len',len(log1))
    print('log2 len', len(log2))
    log1=np.float32(norm(np.array(log1).reshape(-1,1)))
    
    log2=np.float32(norm(np.array(log2).reshape(-1,1)))
    print('log3 len', len(log3))
    log3=np.float32(norm(np.array(log3).reshape(-1,1)))
    # log1 and log2 .shape = (2, 1)
    return log1,log2,log3

def process2predict(cell_feature,rul_norm,tcharge_norm,s_norm):
    x_in=[]
    y_in=[]
    for i in tqdm(range(len(cell_feature))):
        col1=[]
        col2=[]
        len_=len(cell_feature[i])-100
        # k is shift cycle  
        # print("len_",len_)
        for k in range(len_):
            # j is selected cycles (paper behave 'a')
            # for j in range(0,50,1):            
            # temp.shape( j, 12) | temp will select shift + selected cycle = input cycles  
            col1.append(cell_feature[i][k:(50+k)])
            # print('temp shape', temp.shape)
            # (50, 12) | input cycles padding to 50 cycles, 12 is all features
            # col1.append(np.float32(np.pad(temp, ((0,50-j-1),(0,0)), 'edge')))
            # normalize rul cycles
            col2.append(np.float32(((eol_data[i]-k))-rul_norm[0])/rul_norm[1])
            # normalize chargetime cycles
            col2.append((np.float32(chargetime[i])-tcharge_norm[0])/tcharge_norm[1])
            # normalize shift cycles
            col2.append((np.float32(k)-s_norm[0])/s_norm[1])
        # append one battery padding features
        # print('col.shape',len(col1))
        x_in.append(col1)
        # append one battery RUL and shift cycles 
        y_in.append(col2)
    return x_in,y_in

d = layers.Input(shape=(500,4))
conv1 = layers.Conv1D(64, 11, padding='same', activation=tf.nn.relu)(d)
conv2 = layers.Conv1D(32, 3, padding='same', activation=tf.nn.relu)(conv1)
conv3 = layers.Conv1D(64, 9, padding='same', activation=tf.nn.relu)(conv2)
conv4 = layers.Conv1D(32, 13, padding='same', activation=tf.nn.relu)(conv3)
maxpooling1d1_1 = layers.MaxPooling1D(pool_size=2, strides=2)(conv2)
maxpooling1d1_2 = layers.MaxPooling1D(pool_size=2, strides=2)(conv4)
concat1 = layers.Concatenate(axis=2)([maxpooling1d1_1, maxpooling1d1_2])
conv5 = layers.Conv1D(512, 7, padding='same', activation=tf.nn.relu)(concat1)
conv6 = layers.Conv1D(256, 5, padding='same', activation=tf.nn.relu)(conv5)
conv7 = layers.Conv1D(1024, 7, padding='same', activation=tf.nn.relu)(conv6)
conv8 = layers.Conv1D(512, 3, padding='same', activation=tf.nn.relu)(conv7)
maxpooling1d2_1 = layers.MaxPooling1D(pool_size=2, strides=2)(conv6)
maxpooling1d2_2 = layers.MaxPooling1D(pool_size=2, strides=2)(conv8)
concat2 = layers.Concatenate(axis=2)([maxpooling1d2_1, maxpooling1d2_2])
spatialdropout1d1 = layers.SpatialDropout1D(0.16)(concat2)
conv9_1 = layers.Conv1D(128, 7, padding='same', activation=tf.nn.relu)(spatialdropout1d1)
conv9_2 = layers.Conv1D(128, 3, padding='same', activation=tf.nn.relu)(spatialdropout1d1)
dotted = layers.dot([conv9_1, conv9_2], axes=(2, 2))
activation = layers.Activation(tf.nn.sigmoid)(dotted)
globalavgpooling1d1_1 = layers.GlobalAveragePooling1D()(conv9_1)
globalavgpooling1d1_2 = layers.GlobalAveragePooling1D()(activation)
print("globalavgpooling1d1_1 shape", globalavgpooling1d1_1.shape)
print("globalavgpooling1d1_2 shape", globalavgpooling1d1_2.shape)
concat3 = layers.Concatenate(axis=1)([globalavgpooling1d1_1, globalavgpooling1d1_2])
reshape = layers.Reshape((-1, 1))(concat3)
conv10_1 = layers.Conv1D(256, 7, padding='same', activation=tf.nn.relu)(reshape)
conv10_2 = layers.Conv1D(32, 13, padding='same', activation=tf.nn.relu)(reshape)
conv10_3 = layers.Conv1D(128, 7, padding='same', activation=tf.nn.relu)(reshape)
conv11_1 = layers.Conv1D(128, 7, padding='same', activation=tf.nn.relu)(conv10_1)
conv11_2 = layers.Conv1D(128, 5, padding='same', activation=tf.nn.relu)(conv10_2)
conv11_3 = layers.Conv1D(32, 13, padding='same', activation=tf.nn.relu)(conv10_3)
spatialdropout1d2_1 = layers.SpatialDropout1D(0.16)(conv11_1)
spatialdropout1d2_2 = layers.SpatialDropout1D(0.16)(conv11_2)
spatialdropout1d2_3 = layers.SpatialDropout1D(0.16)(conv11_3)
conv12_1 = layers.Conv1D(64, 13, padding='same', activation=tf.nn.relu)(spatialdropout1d2_1)
conv12_2 = layers.Conv1D(512, 3, padding='same', activation=tf.nn.relu)(spatialdropout1d2_2)
conv12_3 = layers.Conv1D(64, 11, padding='same', activation=tf.nn.relu)(spatialdropout1d2_3)
globalavgpooling1d2_1 = layers.GlobalAveragePooling1D()(conv12_1)
globalmaxpooling1d2_1 = layers.GlobalMaxPooling1D()(conv12_1)
globalavgpooling1d2_2 = layers.GlobalAveragePooling1D()(conv12_2)
globalmaxpooling1d2_2 = layers.GlobalMaxPooling1D()(conv12_2)
globalavgpooling1d2_3 = layers.GlobalAveragePooling1D()(conv12_3)
globalmaxpooling1d2_3 = layers.GlobalMaxPooling1D()(conv12_3)
add1 = layers.Add()([globalavgpooling1d2_1, globalmaxpooling1d2_1])
add2 = layers.Add()([globalavgpooling1d2_2, globalmaxpooling1d2_2])
add3 = layers.Add()([globalavgpooling1d2_3, globalmaxpooling1d2_3])
fc1 = layers.Dense(1)(add1)
fc2 = layers.Dense(1)(add2)
fc3 = layers.Dense(1)(add3)
concat4 = layers.Concatenate(axis=1)([fc1, fc2, fc3])
model = Model(d, concat4)
print('concat 4 shape', concat4.shape)



# path = './model/Full RUL DNN/dimreduction125.h5'
# tf.keras.models.save_model(model,path)
# netron.start(path)


data_dir='dataset'
eol_data = np.load('%s/battery_EoL.npy'%(data_dir),allow_pickle='TRUE')
discharge_data=np.load('%s/discharge_data.npy'%(data_dir),allow_pickle='TRUE').tolist()
index=np.load('%s/index_battery.npy'%(data_dir))
chargetime = np.load('./get81battery/chargetime.npy')

train_idx = index[17:]
test_idx = index[:17]
x_train = np.zeros((0, 4, 500))
shift_train = np.zeros(0)
rul_train = np.zeros(0)
tcharge_train = np.zeros(0)
for i in train_idx:
    cycles = len(discharge_data[i])-100
    # print('discharge data shape', discharge_data[i][:100].shape)
    x_train = np.concatenate((x_train, discharge_data[i][:cycles]), axis=0)
    shift_train = np.append(shift_train, np.arange(0, cycles), axis=0)
    rul_train = np.append(rul_train, np.arange(eol_data[i], eol_data[i]-cycles, -1), axis=0)
    tcharge_train = np.append(tcharge_train, np.full(cycles, chargetime[i]), axis = 0)


shift_train = shift_train.reshape(-1,1)
rul_train = rul_train.reshape(-1,1)
tcharge_train = tcharge_train.reshape(-1,1)
y_train = np.concatenate((rul_train, tcharge_train, shift_train), axis=1)
print('x_train len', x_train.shape)
print('shift_train shape', shift_train.shape)
print('rul train shape', rul_train.shape)
print('tchargetime shape', tcharge_train.shape)
print('y_train shape', y_train.shape)


x_test = np.zeros((0, 4, 500))
shift_test = np.zeros(0)
rul_test = np.zeros(0)
tcharge_test = np.zeros(0)
for i in test_idx:
    cycles = len(discharge_data[i]) - 100
    x_test = np.concatenate((x_test, discharge_data[i][:cycles]), axis=0)
    shift_test = np.append(shift_test, np.arange(0, cycles), axis=0)
    rul_test = np.append(rul_test, np.arange(eol_data[i], eol_data[i]-cycles, -1), axis=0)
    tcharge_test = np.append(tcharge_test, np.full(cycles, chargetime[i]), axis=0)

shift_test = shift_test.reshape(-1,1)
rul_test = rul_test.reshape(-1,1)
tcharge_test = tcharge_test.reshape(-1,1)
y_test = np.concatenate((rul_test, tcharge_test, shift_test), axis=1)
print('x_test shape',x_test.shape)
print('shift_test shape', shift_test.shape)
print('rul_test shape', rul_test.shape)
print('tcharge shape', tcharge_test.shape)


x_train, x_train_mean, x_train_std, x_fit = norm(np.transpose(x_train, (0,2,1)).reshape(-1,4))
print('discharge norm', x_train_mean, x_train_std)
print(np.vstack([x_train_mean, x_train_std]))
np.save('./model/Full RUL DNN/discharge_norm.npy', np.vstack([x_train_mean, x_train_std]))
x_train = x_train.reshape(-1, 500, 4)

y_train, y_train_mean, y_train_std, y_fit = norm(y_train)
print('y_train_mean', y_train_mean)

x_test = x_fit.transform(np.transpose(x_test, (0,2,1)).reshape(-1, 4))
x_test = x_test.reshape(-1, 500, 4)
y_test = y_fit.transform(y_test)

opt = tf.keras.optimizers.Adam(learning_rate=0.0005, amsgrad=True)

model.compile(optimizer = opt, loss = 'mse', metrics=[tf.keras.metrics.MeanSquaredError()])

#優化方式
EPOCH = 30
history=model.fit(x_train, y_train, epochs=EPOCH, batch_size=90, shuffle=True,validation_data=(x_test, y_test), verbose=2)
print(history.history['loss'])
print(history.history['val_loss'])

pred1 = model.predict(x_test)
print('pred1', pred1*y_train_std+y_train_mean)
print('rmse', rmse(pred1*y_train_std+y_train_mean, y_test*y_train_std+y_train_mean))
model.save('./model/Full RUL DNN/Dimreduction125.h5')
plt.plot(np.arange(EPOCH), history.history['loss'], label= 'train')
plt.plot(np.arange(EPOCH), history.history['val_loss'], label= 'val')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.title('ft_sel125 mse loss')
plt.legend()
plt.show()
