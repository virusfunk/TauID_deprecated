from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, MaxPool2D, Dropout, Conv2D, Activation, Input, Flatten, Add
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras


def get_baseline():
    input_images = Input(shape=(256,256,4))

    x = Conv2D(32, 3, 1)(input_images)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(64, 3, 1)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(128, 3, 1)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)

    x = Dense(64)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)

    x = Dense(32)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    
    output_onehots = Dense(6, activation='softmax')(x)
    model = Model(inputs=input_images, outputs=output_onehots)
    return model

num_classes = 6
input_shape = (256, 256, 4)
image_size = 256
patch_size = 16
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):
        return {"patch_size": self.patch_size}

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        projection = self.projection(patch)
        embedding = self.position_embedding(positions)
        encoded = projection + embedding
        return encoded
    
    def get_config(self):
        return {"num_patches": self.num_patches, "projection_dim": self.projection_dim}

def get_vit():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(encoded_patches, encoded_patches)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.Flatten()(attention_output)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes, activation='softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model



