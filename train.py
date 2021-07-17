import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tensorflow.keras import optimizers
from dataset import TauDatasetTrain, TauDatasetValidation
import tensorflow as tf
import matplotlib.pyplot as plt
from model import get_baseline

def main():
    model = get_baseline()
    model.summary()
    model.compile(optimizer='Nadam',
              loss='categorical_crossentropy',
              metrics=['acc'])
    
    trainset = TauDatasetTrain()
    valset = TauDatasetValidation()
    
    history = model.fit(trainset, validation_data=valset, shuffle=True, epochs=100,
                   callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='./best_weights.hdf5',
                                                                 save_best_only=True,
                                                                 monitor='val_loss',
                                                                 mode='min'),
                             tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)])
    

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png')


if __name__ == "__main__":
    main()
    
    
