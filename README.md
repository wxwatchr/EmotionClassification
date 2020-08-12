**Emotion Detection and Classification using Neural Networks**

**Description:**  The purpose of this project will be to detect and then classify facial emotions based on images.  Emotion detection is already used in law enforcement.  An application that makes use of emotion detection and classification could be useful for people with the inability to recognize emotions without assistance (typically people on the autism spectrum).

**Data Science Task:**  Emotion detection and classification will be accomplished by creating a neural network.  

**Data:**  Data is available with the Facial Expression Recognition 2013 (FER2013) dataset, located [here](https://www.kaggle.com/astraszab/facial-expression-dataset-image-folders-fer2013?).  The dataset contains over 35,000 images classified as happiness, sadness, anger, surprise, disgust, fear, and neutral.

The dataset was already split into train, validation, and test images.  

![images nums](https://github.com/wxwatchr/EmotionClassification/blob/master/Graphics/num_images.PNG)

ImageDataGenerator was then used to randomly select images during training of the CNN model.

![sample images](https://github.com/wxwatchr/EmotionClassification/blob/master/Graphics/sample_images.PNG)

The data collection and analysis code can be viewed [here](https://github.com/wxwatchr/EmotionClassification/blob/master/DataCollectionAndEDA.py).

**Data Analysis:**  To develop the emotion classification, a neural network will be created.  Various convolutional layers and architectures will be tested for accuracy.

The weights for the CNN were first tested using the VGG16 model.  Then additional Conv2d, BatchNormalization, MaxPooling, Dropout, and Dense layers were added.  The accuracy was low (46.26%).  The VGG16 model code can be viewed [here](https://github.com/wxwatchr/EmotionClassification/blob/master/VGG16model.py).

The same layers were then used with weights set based on the VGG19 model.  The accuracy was lower than the VGG16 model at 25.31%.  The VGG19 model code can be viewed [here](https://github.com/wxwatchr/EmotionClassification/blob/master/VGG19model.py).

This process was again repeated using base weights from the InceptionV3 model.  The accuracy was 36.56%; performing better than the VGG19 model but worse than the VGG16 model.  The InceptionV3 model code can be viewed [here](https://github.com/wxwatchr/EmotionClassification/blob/master/InceptionV3model.py).

With the low accuracy from the weighted CNNs, a custom CNN was created using multiple layers of Conv2D, BatchNormalization, MaxPooling, Dropout, and Dense layers. The model performed well with an accuracy of 86.22%. The custom CNN model code can be viewed [here](https://github.com/wxwatchr/EmotionClassification/blob/master/customCNN.py).

**Conclusions:**  The custom CNN was able to achieve a high accuracy on the FER2013 data set. The accuracy and loss graphs are shown below (respectively).

![model accuracy](https://github.com/wxwatchr/EmotionClassification/blob/master/Graphics/model_accuracy.PNG)

![model loss](https://github.com/wxwatchr/EmotionClassification/blob/master/Graphics/model_loss.PNG)

The confusion matrix below shows an overall accuracy by emotion of:

Angry: 
Disgust:
Fear:
Happy:
Sad:
Surprise:
Neutral:

![confusion matrix](https://github.com/wxwatchr/EmotionClassification/blob/master/Graphics/confusion_matrix.PNG)
