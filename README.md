**Emotion Detection and Classification using Neural Networks**

**Description:**  The purpose of this project will be to detect and then classify facial emotions based on images.  Emotion detection is already used in law enforcement.  An application that makes use of emotion detection and classification could be useful for people with the inability to recognize emotions without assistance (typically people on the autism spectrum).

**Data Science Task:**  Emotion detection and classification will be accomplished by creating a neural network.  

**Data:**  Data is available with the Facial Expression Recognition 2013 (FER2013) dataset, located [here](https://www.kaggle.com/astraszab/facial-expression-dataset-image-folders-fer2013?).  The dataset contains over 35,000 images classified as happiness, sadness, anger, surprise, disgust, fear, and neutral.

The dataset was already split into train, validation, and test images.  ImageDataGenerator was then used to randomly select images during training of the CNN model.

**Data Analysis:**  To develop the emotion classification, a neural network will be created.  Various convolutional layers and architectures will be tested for accuracy.

The weights for the CNN were first tested using the VGG16 model.  Then additional Conv2d, BatchNormalization, MaxPooling, Dropout, and Dense layers were added.  The accuracy was low (46.26%).

The same layers were then used with weights set based on the VGG19 model.  
