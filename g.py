def discriminator(images, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()

    weight1 = tf.get_variable('weight1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.2))
    bias1 = tf.get_variable( bias1', [32], initializer=tf.constant_initializer(0)) 
    d1 = tf.nn.conv2d(input=images, filter=weight1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 +   bias1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional and pool layers
    # This finds 64 different 5 x 5 pixel features
    weight2 = tf.get_variable('weight2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    bias2 = tf.get_variable('bias2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=weight2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + bias2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # First fully connected layer
    d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
    d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)

    # Second fully connected layer
    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
    d4 = tf.matmul(d3, d_w4) + d_b4

    # d4 contains unscaled values
    return d4
