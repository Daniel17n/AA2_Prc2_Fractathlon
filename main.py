
# Script Python para Clasificación de Fractales con CNN y Transfer Learning

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Cargar datos
train_df = pd.read_csv('train.csv')

# Data augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1,
                             rotation_range=15, zoom_range=0.1,
                             horizontal_flip=True, vertical_flip=True)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='train',
    x_col='relative_path',
    y_col='fractal',
    target_size=(224, 224),
    batch_size=32,
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='train',
    x_col='relative_path',
    y_col='fractal',
    target_size=(224, 224),
    batch_size=32,
    subset='validation'
)

# Modelo con EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_df['fractal'].nunique(), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Guardar el modelo entrenado
model.save('fractal_classifier.h5')

# Visualizar precisión
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.savefig('accuracy_plot.png')