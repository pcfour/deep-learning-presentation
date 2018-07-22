from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(10)

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

Y = np.array(
    [0, 
    1, 
    1, 
    1])

model.fit(X, Y, epochs=300, batch_size=2)

test1 = np.array([[0,0]])
test2 = np.array([[0,1]])
test3 = np.array([[1,0]])
test4 = np.array([[1,1]])

prediction = model.predict(test1)
print("Result raw: {}".format(prediction))
decoded_prediction = [round(x[0]) for x in prediction]
print("Result decoded: {}".format(decoded_prediction))

prediction = model.predict(test2)
print("Result raw: {}".format(prediction))
decoded_prediction = [round(x[0]) for x in prediction]
print("Result decoded: {}".format(decoded_prediction))

prediction = model.predict(test3)
print("Result raw: {}".format(prediction))
decoded_prediction = [round(x[0]) for x in prediction]
print("Result decoded: {}".format(decoded_prediction))

prediction = model.predict(test4)
print("Result raw: {}".format(prediction))
decoded_prediction = [round(x[0]) for x in prediction]
print("Result decoded: {}".format(decoded_prediction))
