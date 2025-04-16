import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt

DATASET_PATH = "C-NMC_Leukemia"
TRAIN_PATH = os.path.join(DATASET_PATH, "training_data")
VALID_PATH = os.path.join(DATASET_PATH, "validation_data")
TEST_PATH = os.path.join(DATASET_PATH, "testing_data")

IMG_SIZE = (224, 224)

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
def load_images_from_folder(folder_path, augment_data=False):
    images = []
    labels = []
    for fold in sorted(os.listdir(folder_path)):
        fold_path = os.path.join(folder_path, fold)
        if not os.path.isdir(fold_path):
            continue
        for cls in ["all", "hem"]:
            cls_path = os.path.join(fold_path, cls)
            if not os.path.isdir(cls_path):
                continue
            label = 0 if cls == "all" else 1
            for img_name in tqdm(os.listdir(cls_path), desc=f"Processing {fold}/{cls}"):
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping corrupted image: {img_path}")
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if augment_data:
                    img = augment(image=img)['image']
                img = img / 255.0
                images.append(img)
                labels.append(label)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def load_images_from_folder1(folder_path, augment_data=False):
    images = []
    labels = []
    classes = sorted(os.listdir(folder_path))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Klasörlerin içinde dosya olup olmadığını kontrol et
    print(f"Classes found in {folder_path}: {classes}")

    for cls in classes:
        cls_path = os.path.join(folder_path, cls)
        if not os.path.isdir(cls_path):  # Eğer klasör değilse geç
            continue

        class_images = os.listdir(cls_path)
        print(f"Found {len(class_images)} images in {cls_path}")

        for img_name in tqdm(class_images, desc=f"Processing {cls}"):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping corrupted image: {img_path}")
                continue  # Bozuk dosyaları atla

            # Boyutlandırma ve RGB'ye çevirme
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Veri artırma (sadece eğitim verisine uygula)
            if augment_data:
                img = augment(image=img)['image']

            # Normalizasyon (0-1 aralığına getirme)
            img = img / 255.0
            images.append(img)
            labels.append(class_to_idx[cls])

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


X_train, y_train = load_images_from_folder(TRAIN_PATH, augment_data=True)
X_valid, y_valid = load_images_from_folder1(VALID_PATH, augment_data=False)
X_test, y_test = load_images_from_folder1(TEST_PATH, augment_data=False)


print(f"Train: {X_train.shape}, Validation: {X_valid.shape}, Test: {X_test.shape}")


model = Sequential([
    Conv2D(32, (3, 3), activation='sigmoid', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='sigmoid'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='sigmoid'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='sigmoid'),
    Dropout(0.5),
    Dense(len(os.listdir(TRAIN_PATH)), activation='sigmoid')  # Sınıf sayısı
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=10,
    batch_size=32
)


plt.figure(figsize=(12, 4))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Test verisinden rastgele bir görüntü seçer
idx = random.randint(0, len(X_test))
test_img = X_test[idx]
true_label = y_test[idx]

# Modelin tahmini
prediction = model.predict(np.expand_dims(test_img, axis=0))
predicted_label = np.argmax(prediction)

# Görüntüyü göster ve tahmini yazdır
plt.imshow(test_img)
plt.axis("off")
plt.title(f"Gerçek: {true_label}, Tahmin: {predicted_label}")
plt.show()