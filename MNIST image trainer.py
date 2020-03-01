from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')

X_train /=  255
X_train = X_train.reshape(60000, 28, 28,1)
X_test = X_test.reshape(10000, 28, 28,1)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size= 2, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, kernel_size= 2, activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, kernel_size= 2, activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=7,batch_size=32,validation_split=.2)
model.save('MNIST_Trained_model.h5')
test_results = model.evaluate(X_test,y_test,batch_size=32)
