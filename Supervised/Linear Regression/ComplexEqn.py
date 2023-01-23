import numpy as np
import tensorflow as tf

# Generating random input data
X_train = np.random.rand(1000, 5)

# Generating output data using a simple equation
a = X_train[:, 0]
b = X_train[:, 1]
c = X_train[:, 2]
d = X_train[:, 3]
e = X_train[:, 4]
y_train = 2*a + 4*b + 3*c + 8*d + 5*e


m = tf.keras.Sequential()
m.add(tf.keras.layers.Dense(8, input_shape=[5],activation='relu'))
 # Add a dense layer with 8 input features and 4 output
m.add(tf.keras.layers.Dense(4, activation='relu'))

# Add a dense layer with 4 input features and 1 output
m.add(tf.keras.layers.Dense(1, activation='linear'))

m.compile(optimizer='adam', loss='mean_squared_error')

m.fit(X_train,y_train,epochs=100)

# prediction
p=m.predict([[14.6, 5, 18, 2.89, 5.75]])

# output=[[147.7654]]