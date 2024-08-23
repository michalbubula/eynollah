import tensorflow as tf
layers = tf.keras.layers

PROJECTION_DIM = 64
PATCH_SIZE = 1
NUM_PATCHES = 21*21 #14*14#28*28#14*14#28*28

class Patches(layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = PATCH_SIZE

    def call(self, inputs, *args, **kwargs):
        batch_size = tf.shape(inputs)[0]
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config
    
    
class PatchEncoder(layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_patches = NUM_PATCHES
        self.projection = layers.Dense(units=PROJECTION_DIM)
        self.position_embedding = layers.Embedding(input_dim=NUM_PATCHES, output_dim=PROJECTION_DIM)

    def call(self, inputs, *args, **kwargs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # XXX: pyright thinks self.projection(inputs) is None
        encoded = self.projection(inputs) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config


