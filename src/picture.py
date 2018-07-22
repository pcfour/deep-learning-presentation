import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')
target_size = (224, 224)

def predict(image_path):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    return decode_predictions(predictions, top=1)[0] 

prediction1 = predict('images/test_image1.jpg')
print(prediction1)
prediction2 = predict('images/test_image2.jpg')
print(prediction2)
