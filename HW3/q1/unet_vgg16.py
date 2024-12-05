import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

def unet_vgg16(input_shape=(256, 256, 3)):
    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    vgg16.trainable = False
    
    s1 = vgg16.get_layer("block1_conv2").output  # (256, 256, 64)
    s2 = vgg16.get_layer("block2_conv2").output  # (128, 128, 128)
    s3 = vgg16.get_layer("block3_conv3").output  # (64, 64, 256)
    s4 = vgg16.get_layer("block4_conv3").output  # (32, 32, 512)
    bottleneck = vgg16.get_layer("block5_conv3").output  # (16, 16, 512)

    u1 = UpSampling2D((2, 2))(bottleneck)  # (32, 32, 512)
    u1 = concatenate([s4, u1]) # (32, 32, 1024)
    u1 = Conv2D(512, (3, 3), activation="relu", padding="same")(u1) # (32, 32, 512)
    u1 = Conv2D(512, (3, 3), activation="relu", padding="same")(u1) # (32, 32, 512)
    u1 = Conv2D(256, (3, 3), activation="relu", padding="same")(u1) # (32, 32, 256)

    u2 = UpSampling2D((2, 2))(u1)  # (64, 64, 256)
    u2 = concatenate([s3, u2]) # (64, 64, 512)
    u2 = Conv2D(256, (3, 3), activation="relu", padding="same")(u2) # (64, 64, 256)
    u2 = Conv2D(256, (3, 3), activation="relu", padding="same")(u2) # (64, 64, 256)
    u2 = Conv2D(128, (3, 3), activation="relu", padding="same")(u2) # (64, 64, 128)

    u3 = UpSampling2D((2, 2))(u2)  # (128, 128, 128)
    u3 = concatenate([s2, u3]) # (128, 128, 256)
    u3 = Conv2D(128, (3, 3), activation="relu", padding="same")(u3) # (128, 128, 128)
    u3 = Conv2D(64, (3, 3), activation="relu", padding="same")(u3) # (128, 128, 64)

    u4 = UpSampling2D((2, 2))(u3)  # (256, 256, 64)
    u4 = concatenate([s1, u4]) # (256, 256, 128)
    u4 = Conv2D(64, (3, 3), activation="relu", padding="same")(u4) # (256, 256, 64)
    u4 = Conv2D(64, (3, 3), activation="relu", padding="same")(u4) # (256, 256, 64)
    
    output = Conv2D(1, (1, 1), activation="sigmoid")(u4) # (256, 256, 1)

    model = Model(inputs=vgg16.input, outputs=output)
    return model