from preprocess import load_data
from model import build_model
from keras.utils import to_categorical
from sklearn.utils import shuffle

labels = {name: i for i, name in enumerate(sorted(os.listdir('data/train')))}

X_train, y_train = load_data('data/train', labels)
X_valid, y_valid = load_data('data/valid', labels)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

y_train = to_categorical(y_train, num_classes=len(labels))
y_valid = to_categorical(y_valid, num_classes=len(labels))

model = build_model(len(labels))

model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=10,
    batch_size=32
)

model.save("model.keras")
