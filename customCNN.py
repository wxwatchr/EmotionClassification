model = models.Sequential()
model.add(Conv2D(32, (3,1), strides=(2,2), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(Conv2D(32, (3,1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,1), activation='relu', padding='same'))
model.add(Conv2D(64, (3,1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,1), activation='relu', padding='same'))
model.add(Conv2D(128, (3,1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax'))

# Compile model
model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit_generator(train_flow,
                    epochs=50, 
                    verbose=1, 
                    validation_data=validation_flow)

#plot accuracy vs epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate test data.
scores = model.evaluate(X_validation, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])