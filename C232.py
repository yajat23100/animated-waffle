from numpy import loadtxt
from keras.model import Sequential
from keras.layers import Dense

dataset = loadtxt('diabetes_dataset.csv', delimiter=',')

x = dataset[:,0:8]
y = dataset[:,8]

model = sequential()
model = add(dense(12, input_dime=8, activation='relu'))
model = add(dense(8, activation='relu'))
model = add(dense(1, activation='sigmod'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=500, batch_size=100)
predictions = model.predict_classes(x)
for i in range(5):
    print(f'{x[i].tolist()} => {predictions} expected {y[i]}')
