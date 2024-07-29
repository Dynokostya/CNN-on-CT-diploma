import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split


tf.random.set_seed(42)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Configure TensorFlow to use GPU
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')


def build_model(image_size):
    model = Sequential([
        tf.keras.Input(shape=(image_size[0], image_size[1], 1)),  # Specify input shape here
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(16, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def preprocess_image(image_path, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # Assuming images are grayscale
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize to [0, 1] range
    return image


def load_dataset(base_dir, image_size, split_percentage):
    all_image_paths = []
    all_labels = []
    inference_image_paths = []
    inference_labels = []

    for i, patient in enumerate(os.listdir(base_dir)[:16]):
        patient_path = os.path.join(base_dir, patient, 'DICOMCUT')
        patient_images = os.listdir(patient_path)
        if patient_images:
            # Reserve the first image for the inference set
            inference_image_paths.append(os.path.join(patient_path, patient_images[0]))
            inference_labels.append(patient)
            # Add the rest to the training/validation set
            for img_file in patient_images[1:]:
                all_image_paths.append(os.path.join(patient_path, img_file))
                all_labels.append(patient)

    # Convert labels to indices
    label_to_index = {label: idx for idx, label in enumerate(sorted(set(all_labels + inference_labels)))}
    label_indices = [label_to_index[label] for label in all_labels]
    inference_label_indices = [label_to_index[label] for label in inference_labels]

    # Split the dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, label_indices, test_size=1 - split_percentage, random_state=42)

    # Creating training, validation, and inference datasets
    train_ds = paths_to_dataset(train_paths, train_labels, image_size)
    val_ds = paths_to_dataset(val_paths, val_labels, image_size)
    inference_ds = paths_to_dataset(inference_image_paths, inference_label_indices, image_size)

    return train_ds, val_ds, inference_ds


def paths_to_dataset(image_paths, labels, image_size):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(lambda x: preprocess_image(x, image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(labels, num_classes=len(set(labels))))
    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.shuffle(buffer_size=len(image_paths)).batch(1).prefetch(
        tf.data.experimental.AUTOTUNE)  # Batch size set to 1 for inference
    return ds


def train_model(model, train_ds, val_ds, epochs):
    return model.fit(train_ds, epochs=epochs, validation_data=val_ds)


def evaluate_model(model, val_ds):
    return model.evaluate(val_ds)


# Example usage:
base_dir = 'PatientData'
image_size = (512, 512)
split_percentage = 0.8

model = build_model(image_size)
# model = load_model('classifier7.h5')
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_ds, val_ds, inference_ds = load_dataset(base_dir, image_size, split_percentage)
history = train_model(model, train_ds, val_ds, 1)
# model = load_model('classifier.h5')
evaluation = evaluate_model(model, val_ds)
print(evaluation)
model.summary()
# model.save('classifier.h5')

# Use the inference dataset to predict and demonstrate model's prediction on known samples
for images, labels in inference_ds.take(16):
    pred = model.predict(images)
    print("Predicted:", tf.argmax(pred, axis=1).numpy(), "Actual:", tf.argmax(labels, axis=1).numpy())
