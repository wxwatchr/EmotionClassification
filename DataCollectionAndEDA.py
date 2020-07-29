# Download the dataset as images
import zipfile
with zipfile.ZipFile('/content/drive/My Drive/FER2013Zip.zip', 'r') as zip_ref:
    zip_ref.extractall('/content')


batch_size = 16

train_dataset_dir = 'data/train/'
val_dataset_dir = 'data/val/'
test_dataset_dir = 'data/test/'

# create training set
train_dataGen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# create test set
test_dataGen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# create validation set
validation_dataGen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# load images into training set
train_gen = train_dataGen.flow_from_directory(
        'data/train', 
        target_size=(150, 150))
        #class_mode=None,
        #shuffle=False)  

# load images into test set
test_gen = test_dataGen.flow_from_directory(
        'data/test',
        target_size=(150, 150))
        #class_mode=None,
        #shuffle=False)

# load images into validation set
validation_gen = validation_dataGen.flow_from_directory(
        'data/val',
        target_size=(150, 150))
        #class_mode=None,
        #shuffle=False)

# View a few of the images
sample_train_img, _ = next(train_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_train_img[:5])