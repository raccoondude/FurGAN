from imutils import paths
import cv2
import random
import numpy as np
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display


#configureable variables
DATASET = "Art"
GETDATAFROMDIR = True
OUTPUTX = 28
OUTPUTY = 28
OUTPUTZ = 3
LOAD = True
EPOCHS = 5
noise_dim = 100
num_examples_to_generate = 16

def getImageArray(Dataset):
    #Define main variable
    data = []
    imagePaths = sorted(list(paths.list_images(Dataset)))
    random.seed(42)
    random.shuffle(imagePaths)
    print("Loading")
    click = 0
    for imagePath in imagePaths:
        click = click + 1
        print("{}/{}".format(click, len(imagePaths)))
        image = cv2.imread(imagePath)
        #Resize image 50x50x3
        image = cv2.resize(image, (OUTPUTX,OUTPUTY))
        print(image)
        print(image.shape)
        data.append(image)
    data = np.array(data)
    return data

if GETDATAFROMDIR == True:
    data = getImageArray(DATASET)
else:
    pass #Add your own way of getting data

data = data.reshape(data.shape[0], OUTPUTX, OUTPUTY, OUTPUTZ).astype("float32")
data = (data - 127.5) / 127.5
BUFFER_SIZE = 60000
BATCH_SIZE = 12

train_dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


#configureable!
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(50*50*100, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((50, 50, 100)))
    assert model.output_shape == (None, 50, 50, 100) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(80, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 50, 80)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(60, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 100, 100, 60)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(40, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 200, 200, 40)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, OUTPUTX, OUTPUTY, OUTPUTZ)

    return model

generator = make_generator_model()



def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[OUTPUTX, OUTPUTY, OUTPUTZ]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


discriminator = make_discriminator_model()


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = "./Save"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


if LOAD == True:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

decision = discriminator(generated_image)
print (decision)

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  print("Training....")
  uwu = 0
  for a in dataset:
      uwu = uwu + 1
  for epoch in range(epochs):
    print("{}/{}".format(epoch, EPOCHS))
    start = time.time()
    owo = 0
    for image_batch in dataset:
      owo = owo + 1
      print("{}/{}".format(owo, uwu))
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)



train(train_dataset, EPOCHS)
