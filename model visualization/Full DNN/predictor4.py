# create predictor4 model
d1 = layers.Input(shape=(100, 10))
conv1 = layers.Conv1D(256, 15, padding='same', activation=tf.nn.relu)(d1)
conv2 = layers.Conv1D(256, 13, padding='valid', activation=tf.nn.relu)(conv1)
conv3 = layers.Conv1D(128, 15, padding='same', activation=tf.nn.relu)(conv2)
d2 = layers.Input(shape=(1))
fc1_1 = layers.Dense(88)(d2)
fc1_2 = layers.Dense(128)(d2)
spatialdropout1d1 = layers.SpatialDropout1D(0.16)(conv3)
reshape1_1 = layers.Reshape((88, 1))(fc1_1)
reshape1_2 = layers.Reshape((1, 128))(fc1_2)
multiply1_1 = layers.multiply([spatialdropout1d1, reshape1_1])
multiply1_2 = layers.multiply([spatialdropout1d1, reshape1_2])
conv4_1 = layers.Conv1D(9, 128, padding='same', activation=tf.nn.relu)(multiply1_1)
conv4_2 = layers.Conv1D(9, 512, padding='same', activation=tf.nn.relu)(multiply1_2)
dotted = layers.dot([conv4_1, conv4_2], axes=(2, 2))
activation = layers.Activation(tf.nn.sigmoid)(dotted)
spatialdropout1d2 = layers.SpatialDropout1D(0.16)(activation)
conv5_1 = layers.Conv1D(32, 9, padding='same', activation=tf.nn.relu)(spatialdropout1d2)
conv5_2 = layers.Conv1D(256, 11, padding='same', activation=tf.nn.relu)(spatialdropout1d2)
globalavgpooling1d1_1 = layers.GlobalAveragePooling1D()(conv5_1)
globalmaxpooling1d1_1 = layers.GlobalMaxPooling1D()(conv5_1)
globalavgpooling1d1_2 = layers.GlobalAveragePooling1D()(conv5_2)
globalmaxpooling1d1_2 = layers.GlobalMaxPooling1D()(conv5_2)
conv6_1 = layers.Conv1D(50, 13, padding='same', activation=tf.nn.relu)(conv5_1)
conv6_2 = layers.Conv1D(50, 5, padding='same', activation=tf.nn.relu)(conv5_2)
add1_1 = layers.Add()([globalavgpooling1d1_1, globalmaxpooling1d1_1])
add1_2 = layers.Add()([globalavgpooling1d1_2, globalmaxpooling1d1_2])
concat1 = layers.Concatenate(axis=1)([add1_1, add1_2])
globalavgpooling1d2_1 = layers.GlobalAveragePooling1D()(conv6_1)
globalmaxpooling1d2_1 = layers.GlobalMaxPooling1D()(conv6_1)
globalavgpooling1d2_2 = layers.GlobalAveragePooling1D()(conv6_2)
globalmaxpooling1d2_2 = layers.GlobalMaxPooling1D()(conv6_2)
add2_1 = layers.Add()([globalavgpooling1d2_1, globalmaxpooling1d2_1])
add2_2 = layers.Add()([globalavgpooling1d2_2, globalmaxpooling1d2_2])
concat2 = layers.Concatenate(axis=1)([add2_1, add2_2])
fc2_1 = layers.Dense(256)(concat1)
fc2_2 = layers.Dense(256)(concat1)
fc3_1 = layers.Dense(1)(fc2_1)
fc3_2 = layers.Dense(1)(fc2_2)
model = Model(inputs=[d1, d2], outputs=[fc3_1, fc3_2, concat2])

# save model and visualize
path = './predictor4.h5'
tf.keras.models.save_model(model,path)
netron.start(path)