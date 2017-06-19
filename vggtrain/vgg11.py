import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg11:
    """
    A trainable version VGG11.
    """

    def __init__(self):
        self.var_dict = {}

    def build(self, rgb):
        """
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        rgb_scaled = rgb * 255.0

        red, green, blue = tf.split(axis=3,
                                    num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.pool1 = self.max_pool(self.conv1_1, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.pool2 = self.max_pool(self.conv2_1, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_2, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_2, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_2, 'pool5')
        
        # 25088 = (224 // (2 ** 5)) * 512
        _a = self.pool5.get_shape().as_list()[2] * 512
        self.fc6 = self.fc_layer(self.pool5, _a, 4096, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, 4096, 40, "fc8")
        
        # return tf.nn.softmax(self.fc8, name="prob")
        return self.fc8

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels,
                                                  out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        f_init = tf.truncated_normal([filter_size, filter_size,
                                      in_channels, out_channels], 0.0, 0.001)
        filters = tf.Variable(f_init, name + "_filters")

        b_init = tf.truncated_normal([out_channels], .0, .001)
        biases = tf.Variable(b_init, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        w_init = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = tf.Variable(w_init, name + "_weights")

        b_init = tf.truncated_normal([out_size], .0, .001)
        biases = tf.Variable(b_init, name + "_biases")

        return weights, biases

