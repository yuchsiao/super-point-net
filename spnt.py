import tensorflow as tf

class PointNet(object):

    def __init__(self, batch, keep_rate, is_training):
        """
        Args:
            batch: B*P*C (BATCH_SIZE * NUM_POINTS * NUM_INPUT_COORDS)
        """
        # batch_size = batch.get_shape()[0].value
        num_points = batch.get_shape()[1].value
        num_input_coords = batch.get_shape()[2].value

        with tf.variable_scope('input_tnet') as sc:
            T_input = self.transform_net(batch, is_training)  # C*C
        # print(T_input.get_shape())
        batch = tf.matmul(batch, T_input)  # B*P*C

        batch = self.fully_connected(batch, 64, 'fc1', is_training)  # B*P*64
        batch = self.fully_connected(batch, 64, 'fc2', is_training)  # B*P*64

        with tf.variable_scope('middle_tnet') as sc:
            T_middle = self.transform_net(batch, is_training)  # C*C
        batch = tf.matmul(batch, T_middle)  # B*P*C

        batch = self.fully_connected(batch, 64, 'fc3', is_training)  # B*P*64
        batch = self.fully_connected(batch, 128, 'fc4', is_training)  # B*P*128
        batch = self.fully_connected(batch, 1024, 'fc5', is_training)  # B*P*1024

        # max pooling
        with tf.variable_scope('maxpool') as sc:
            batch = tf.expand_dims(batch, -1)  # B*P*1024*1
            batch = tf.nn.max_pool(batch,
                                   ksize=[1, num_points, 1, 1],
                                   strides=[1, 1, 1, 1],
                                   padding='VALID',
                                   name=sc.name)  # B*1*1024*1
            batch = tf.squeeze(batch)  # B*1024
        # print(batch.get_shape())
        batch = self.fully_connected(batch, 512, 'fc6', is_training)  # B*512
        batch = tf.layers.dropout(batch, rate=1 - keep_rate, training=is_training, name='dp1')
        batch = self.fully_connected(batch, 256, 'fc7', is_training)  # B*256
        batch = tf.layers.dropout(batch, rate=1 - keep_rate, training=is_training, name='dp2')
        batch = self.fully_connected(batch, 40, 'fc8', is_training, bn=False, relu=False)  # B*40

        self.output = batch
        self.T_middle = T_middle

    def fully_connected(self, x, num_outputs, scope_name, is_training, bn=True, relu=True):
        with tf.variable_scope(scope_name):
            x = tf.contrib.layers.fully_connected(x, num_outputs,
                                                  activation_fn=None,
                                                  scope='dense')
            if bn:
                x = tf.contrib.layers.batch_norm(x,
                                                 center=True, scale=True,
                                                 is_training=is_training,
                                                 scope='bn')
            if relu:
                x = tf.nn.relu(x, 'relu')

            return x

    def transform_net(self, batch, is_training):
        batch_size = batch.get_shape()[0].value
        num_points = batch.get_shape()[1].value
        num_input_coords = batch.get_shape()[2].value
        # print("num_input_coords:", num_input_coords)

        batch = self.fully_connected(batch, 64, 'fc1', is_training)  # B*P*64
        batch = self.fully_connected(batch, 128, 'fc2', is_training)  # B*P*128
        batch = self.fully_connected(batch, 1024, 'fc3', is_training)  # B*P*1024

        # max pooling
        with tf.variable_scope('maxpool') as sc:
            batch = tf.expand_dims(batch, -1)  # B*P*1024*1
            batch = tf.nn.max_pool(batch,
                                   ksize=[1, num_points, 1, 1],
                                   strides=[1, 1, 1, 1],
                                   padding='VALID',
                                   name=sc.name)  # B*1*1024*1
            batch = tf.squeeze(batch)  # B*1024
        # print(batch.get_shape())

        batch = self.fully_connected(batch, 512, 'fc4', is_training)  # B*512
        batch = self.fully_connected(batch, 256, 'fc5', is_training)  # B*256
        batch = self.fully_connected(batch, num_input_coords * num_input_coords,
                                     'fc6', is_training, bn=False, relu=False)  # B*(C*C)
        # print(batch.get_shape())
        t_net = tf.reshape(batch, (-1, num_input_coords, num_input_coords))
        return t_net

