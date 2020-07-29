# create CNN
k.clear_session()

# import weights
conv_base = VGG16 (weights = 'imagenet', 
                 include_top = False,
                 input_shape = (150, 150, 3))
conv_base.trainable = False 

model = models.Sequential()
model.add(conv_base)

model.add(Conv2D(32, (3,3), strides=(2,2), activation='relu', kernel_initializer='he_normal', padding='same', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(BatchNormalization())
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile model
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.01, decay=1e-6),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

# Plot graphs
history = model.fit_generator(
    train_data,
    train_labels,
    #steps_per_epoch=100,
    epochs=50,
    validation_data=(validation_data, validation_labels),
    #validation_steps=29,
    verbose = 1,
    use_multiprocessing=True,
    workers=12,
    callbacks=[EarlyStopping(monitor='val_acc', patience = 3, restore_best_weights = True)])

# output plots
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(history_dict['acc']) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)