import numpy as np
import matplotlib.pyplot as plt

# Model (especially sine amplitude) depends on random seed
# By twaking the architecture it could be made more robust
np.random.seed(4)

# Generating 23 periods of sine
width = np.pi*2*23
numOfSamples = 2300
lengthTrain = 1200
lengthValidation = 1000
look_back = 1 # Can be set higher, in my experiments it made performance worse though
transientTime = 600 # Time to "burn in" time series

series = np.sin(np.linspace(0,width,numOfSamples))
plt.plot(series)
plt.title("Time series to predict")
plt.show()

def generateTrainData(series, i, look_back):
    return series[i:look_back+i+1]

trainX = np.stack([generateTrainData(series, i, look_back) for i in range(lengthTrain)])
testX = np.stack([generateTrainData(series, lengthTrain + i, look_back) for i in range(lengthValidation)])

trainX = trainX.reshape((lengthTrain,look_back+1,1))
testX = testX.reshape((lengthValidation, look_back + 1, 1))

trainY = trainX[:,1:,:]
trainX = trainX[:,:-1,:]

testY = testX[:,1:,:]
testX = testX[:,:-1,:]

############### Build Model ###############

import keras
from keras.models import Sequential, Model
from keras import layers
from keras import regularizers

inputs = layers.Input(batch_shape=(1,look_back,1), name="main_input")
inputsAux = layers.Input(batch_shape=(1,look_back,1), name="aux_input")

# this layer makes the actual prediction, i.e. decides if and how much it goes up or down
x = layers.recurrent.LSTM(128,return_sequences=True, stateful=True)(inputs)
x = layers.wrappers.TimeDistributed(layers.Dense(1, activation="linear",
                                                 kernel_regularizer=regularizers.l2(0.005),
                                                 activity_regularizer=regularizers.l1(0.005)))(x)

# auxillary input, the current input will be feed directly to the output
# this way the prediction from the step before will be used as a "base", and the Network just have to
# learn if it goes a little up or down
auxX = layers.wrappers.TimeDistributed(layers.Dense(1,
                                                    kernel_initializer=keras.initializers.Constant(value=1),
                                                    bias_initializer='zeros',
                                                    input_shape=(1,1), activation="linear", trainable=False
                                                    ))(inputsAux)

outputs = layers.add([x, auxX], name="main_output")
model = Model(inputs=[inputs, inputsAux], outputs=outputs)
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])
model.summary()
model.fit({"main_input": trainX, "aux_input": trainX[:,look_back-1,:].reshape(lengthTrain,1,1)},{"main_output": trainY}, epochs=4, batch_size=1, shuffle=False)



############### make predictions ###############

burnedInPredictions = np.zeros(transientTime)
testPredictions = np.zeros(len(testX))
# burn series in, here use first transitionTime number of samples from test data
for i in range(transientTime):
    prediction = model.predict([np.array(testX[i, :, 0].reshape(1, look_back, 1)), np.array(testX[i, look_back - 1, 0].reshape(1, 1, 1))])
    testPredictions[i] = prediction[0,0,0]

burnedInPredictions[:] = testPredictions[:transientTime]
# prediction, now dont use any previous data whatsoever anymore, network just has to run on its own output
for i in range(transientTime, len(testX)):
    prediction = model.predict([prediction, prediction])
    testPredictions[i] = prediction[0,0,0]

# for plotting reasons
testPredictions[:np.size(burnedInPredictions)-1] = np.nan



############### plot results ###############
import matplotlib.pyplot as plt
plt.plot(testX[:, 0, 0])
plt.show()
plt.plot(burnedInPredictions, label = "burned in")
plt.plot(testPredictions, label = "free running")
plt.legend()
plt.show()