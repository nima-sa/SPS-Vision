import keras
from keras import layers
import tensorflow as tf


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


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier_raw(
        inputs,
        patch_size,
        num_patches,
        projection_dim,
        transformer_layers,
        num_heads,
        transformer_units,
        mlp_head_units,
        num_classes,
        residual=None,
):
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for idx in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f'vit_ln_1_{idx}')(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name=f'vit_mha_{idx}')(x1, x1)

        x2 = layers.Add(name=f'vit_add_1_{idx}')([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6, name=f'vit_ln_2_{idx}')(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add(name=f'vid_add_2_{idx}')([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6, name=f'vit_ln_3_{idx}')(encoded_patches)
    representation = layers.Flatten(name=f'vit_flatten_{idx}')(representation)
    representation = layers.Dropout(0.5)(representation)
    if residual is not None:
        mixed = layers.Concatenate(name=f'vit_concat_{idx}')([representation, *residual])
    else:
        mixed = representation

    features = mlp(mixed, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes, name=f'vit_dense_2_{idx}')(features)
    return logits


def create_vit_classifier(
        input_shape,
        patch_size,
        num_patches,
        projection_dim,
        transformer_layers,
        num_heads,
        transformer_units,
        mlp_head_units,
        num_classes,
        residual=None,
):
    inputs = layers.Input(shape=input_shape, name='vit_input')
    logits = create_vit_classifier_raw(
        inputs,
        patch_size,
        num_patches,
        projection_dim,
        transformer_layers,
        num_heads,
        transformer_units,
        mlp_head_units,
        num_classes,
        residual,
    )
    model = keras.Model(inputs=inputs, outputs=logits, name='vit')
    return model
