import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tensorflow.keras import optimizers
from dataset import TauDatasetTrain, TauDatasetValidation
import tensorflow as tf
import matplotlib.pyplot as plt
from model import get_baseline, get_vit
import tensorflow_addons as tfa
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')

def main(args):
    if args.model=='CNN':
        model = get_baseline()
        model.summary()
    elif args.model=='ViT':
        model = get_vit()
        model.summary()
    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.000001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    trainset = TauDatasetTrain()
    valset = TauDatasetValidation()
    
    if finetune:
        model.load_weights(finetune)
    
    history = model.fit(trainset, validation_data=valset, shuffle=True, epochs=100,
                   callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=f'./best_weights_{args.model}.hdf5',
                                                                 save_best_only=True,
                                                                 monitor='val_loss',
                                                                 mode='min'),
                             tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)])
    

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'loss_{args.model}.png')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training TauID')
    parser.add_argument('--model', dest='model', default='ViT')
    parser.add_argument('--finetune', dest='finetune', default='best_weights_ViT_fix.hfd5')
    args = parser.parse_args()
    main(args)
   
    
