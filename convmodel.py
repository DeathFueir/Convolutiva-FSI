# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 2
batch_size = 4

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot([i], 3)  # [one_hot(float(i), num_classes)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y

example_batch_train, label_batch_train = dataSource(["train/0/*.jpg", "train/1/*.jpg", "train/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["valid/0/*.jpg", "valid/1/*.jpg", "valid/2/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["test/0/*.jpg", "test/1/*.jpg", "test/2/*.jpg"], batch_size=batch_size)

label_batch_train = tf.cast(label_batch_train, tf.float32)
label_batch_valid = tf.cast(label_batch_valid, tf.float32)
label_batch_test = tf.cast(label_batch_test, tf.float32)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:

    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    epoch = 0
    error = 99999
    errores = []
    #for _ in range(430):
    while True:
        sess.run(optimizer)
        if epoch % 20 == 0:
            print("Iter:", epoch, "---------------------------------------------")
            #print(sess.run(label_batch_valid))
            #print(sess.run(example_batch_valid_predicted))
            print("Error:", sess.run(cost_valid))
        errorAnt = error
        error = sess.run(cost_valid)
        errores.append(error)
        epoch += 1
        if (abs(error-errorAnt) / errorAnt) < 0.0001 or epoch > 200:
            break
    contador = 0
    inter = 0
    for correcto, tested in zip(label_batch_test.eval(), example_batch_test_predicted.eval()):
        inter = inter + 1
        # print(correcto, "-->", tested)
        if (np.argmax(correcto) == np.argmax(tested)):
            contador += 1
    print("----------------------------------------------------------------------------------")
    print("Porcentaje acierto = " + str(( contador/ inter)*100) + "%")


    plt.plot(errores)
    plt.xlabel("Epocas")
    plt.ylabel("Error de validación")
    plt.savefig("figura1.png")
    plt.show()
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
            
    coord.request_stop()
    coord.join(threads)
