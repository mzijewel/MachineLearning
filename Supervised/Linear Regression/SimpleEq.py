import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow import keras

x=np.array([
            [1,3,4],
            [3,4,5],
            [4,2,1],
            [5,3,10]
            ])

# 2x+3y+4z
y=np.array([27,38,18,59])

model = keras.Sequential([
            Dense(units=1,input_shape=[3])
        ])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=600)

# prediction
p=model.predict([[4,5,8]])  

# output=[[54.735012]]