from keras.models import load_model
from PIL import Image
import numpy as np

lables = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']
model = load_model('MNIST_Trained_model.h5')

input_path = input('enter the image path')
input_image = Image.open(input_path)
input_image = input_image.resize((28,28),resample = Image.LANCZOS)
image_array = np.array(input_image)
image_array = image_array.astype('float32')
image_array = image_array.reshape(1,28,28,1)
answer = model.predict(image_array)
print(lables[np.argmax(answer)],'\nAccuracy:',(100*np.max(answer)))