import argparse
import datetime
import os
import sys

import h5py
import numpy as np
import tensorflow as tf

from spnt import PointNet


def parse_args(arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default="data/modelnet40_ply_hdf5_2048/train_files.txt")
    parser.add_argument('--test', default="data/modelnet40_ply_hdf5_2048/test_files.txt")
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--name', default=None)
    parser.add_argument('--model-path-file', default='', help='specify model file for evaluation')
    parser.add_argument('--model', default='spnt')
    parser.add_argument('--num-points', type=int, default=256)
    parser.add_argument('--num-input-coords', default=3)
    parser.add_argument('--num-epochs', default=250)
    parser.add_argument('--num-batches-to-report', default=64)
    parser.add_argument('--batch-size', default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--decay-step', type=int, default=200000)
    parser.add_argument('--decay-rate', type=float, default=0.7)
    parser.add_argument('--keep-rate', type=float, default=0.7)
    parser.add_argument('--l2-weight', type=float, default=0.001)
    if arguments is not None:
        args = parser.parse_args(arguments)
    else:
        args = parser.parse_args()
    return args


# Parse args and set up global variables
args = parse_args()

NUM_EPOCHS = args.num_epochs
NUM_BATCHES_TO_REPORT = args.num_batches_to_report

NUM_POINTS = args.num_points
BATCH_SIZE = args.batch_size
NUM_INPUT_COORDS = args.num_input_coords

LEARNING_RATE = args.learning_rate
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate

KEEP_RATE = args.keep_rate
L2_WEIGHT = args.l2_weight

if args.name is None:
    MODEL_NAME = 'models'
else:
    MODEL_NAME = args.name

MODEL_PATH_FILE = args.model_path_file
LOG_DIR = 'log'


def load_data(filenames):
    data = []
    label = []
    for filename in filenames:
        f = h5py.File(filename, 'r')
        data.extend(f['data'][:])
        label.extend(f['label'][:])
        f.close()
    return np.array(data), np.array(label).squeeze()


def get_learning_rate(step_idx):
    learning_rate = tf.train.exponential_decay(
                        LEARNING_RATE,  # Base learning rate.
                        step_idx * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def train_epoch(sess, data, labels, ops, train_writer):
    def rotate_points(batch):
        rotated_data = np.copy(batch)
        for k in range(batch.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            single_image = batch[k, ...]
            num_coord_sets = single_image.shape[1] // 3
            for i in range(num_coord_sets):  # do not rotate weight
                coords = single_image[:, 3 * i:3 * (i + 1)]
                rotated_data[k, :, 3 * i:3 * (i + 1)] = coords.reshape((-1, 3)).dot(rotation_matrix)
        return rotated_data

    def jitter_points(batch, sigma=0.01, clip=0.05):
        B, N, C = batch.shape
        assert (clip > 0)
        if C == 3:
            jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        else:
            jittered_data = np.zeros((B, N, C))
            jittered_data[:, :, :-1] += np.clip(sigma * np.random.randn(B, N, C - 1), -1 * clip, clip)

        jittered_data += batch
        return jittered_data

    is_training = True
    num_batches = data.shape[0] // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        # shuffle data
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        data = data[idx, ...]
        labels = labels[idx]

        # batch indexing
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        # Augment batched point clouds by rotation and jittering
        rotated_data = rotate_points(
            data[start_idx:end_idx, 0:NUM_POINTS, :]
        )
        jittered_data = jitter_points(rotated_data)
        batch_data = jittered_data

        feed_dict = {ops['points_ph']: batch_data,
                     ops['labels_ph']: labels[start_idx:end_idx],
                     ops['is_training_ph']: is_training}

        step, _, loss_val, pred_val = sess.run(
            [
                ops['step'],
                ops['train_op'],
                ops['loss'],
                ops['pred']
            ], feed_dict=feed_dict
        )
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == labels[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

        if (batch_idx + 1) % NUM_BATCHES_TO_REPORT == 0:
            print('---- batch {:>3d} ---- {}'.format(batch_idx + 1, datetime.datetime.now()))
            print('mean loss: %f' % (loss_sum / float(num_batches)))
            print('accuracy: %f' % (total_correct / float(total_seen)))
            sys.stdout.flush()
            # reset counters
            total_correct = 0
            total_seen = 0
            loss_sum = 0

    print('---- end of epoch - {}'.format(datetime.datetime.now()))
    print('mean loss: %f' % (loss_sum / float(total_seen)))
    print('accuracy: %f' % (total_correct / float(total_seen)))
    print('----------------------')
    sys.stdout.flush()


def eval_epoch(sess, data, labels, ops, test_writer):
    is_training = False
    num_batches = data.shape[0] // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        # batch indexing
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        # Augment batched point clouds by rotation and jittering
        batch_data = data[start_idx:end_idx, 0:NUM_POINTS, :]

        feed_dict = {ops['points_ph']: batch_data,
                     ops['labels_ph']: labels[start_idx:end_idx],
                     ops['is_training_ph']: is_training}

        loss_val, pred_val = sess.run(
            [
                ops['loss'],
                ops['pred']
            ], feed_dict=feed_dict
        )
        #         train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == labels[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

    print('eval mean loss: %f' % (loss_sum / float(total_seen)))
    print('eval accuracy: %f' % (total_correct / float(total_seen)))
    sys.stdout.flush()

    return total_correct / float(total_seen)


def train():
    tf.reset_default_graph()

    points_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, NUM_INPUT_COORDS))
    labels_ph = tf.placeholder(tf.int64, shape=(BATCH_SIZE))
    is_training_ph = tf.placeholder(tf.bool, shape=())

    pn = PointNet(points_ph, keep_rate=KEEP_RATE, is_training=is_training_ph)
    pred = pn.output
    T_middle = pn.T_middle

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels_ph)
    )
    num_features = T_middle.get_shape()[1].value
    mat_diff = tf.matmul(T_middle, tf.transpose(T_middle, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(num_features), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)

    loss += L2_WEIGHT * mat_diff_loss

    batch_idx = tf.Variable(0)
    learning_rate = get_learning_rate(batch_idx)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=batch_idx)

    saver = tf.train.Saver(max_to_keep=1000)
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                         tf.get_default_graph())
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

    ops = {
        'points_ph': points_ph,
        'labels_ph': labels_ph,
        'is_training_ph': is_training_ph,
        'pred': pred,
        'loss': loss,
        'train_op': train_op,
        'merged': merged,
        'step': batch_idx
    }

    best_acc = 0.

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(NUM_EPOCHS):
            print('**** EPOCH {:03d} **** {}'.format(epoch, datetime.datetime.now()))
            sys.stdout.flush()

            train_epoch(sess, train_data, train_labels, ops, train_writer)
            acc = eval_epoch(sess, test_data, test_labels, ops, test_writer)

            if acc > best_acc:
                save_path = saver.save(sess, os.path.join(LOG_DIR, MODEL_NAME + "_best"))
                print("Best so far. Model saved in file: {}, {}".format(save_path, datetime.datetime.now()))
                sys.stdout.flush()
                best_acc = acc

            # Save the variables to disk.
            if (epoch + 1) % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, MODEL_NAME), global_step=epoch)
                print("Model saved in file: {}, {}".format(save_path, datetime.datetime.now()))
                sys.stdout.flush()


def evaluate():
    tf.reset_default_graph()

    points_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, NUM_INPUT_COORDS))
    labels_ph = tf.placeholder(tf.int64, shape=(BATCH_SIZE))
    is_training_ph = tf.placeholder(tf.bool, shape=())

    pn = PointNet(points_ph, keep_rate=1., is_training=is_training_ph)
    pred = pn.output
    T_middle = pn.T_middle

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels_ph)
    )
    num_features = T_middle.get_shape()[1].value
    mat_diff = tf.matmul(T_middle, tf.transpose(T_middle, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(num_features), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)

    loss += L2_WEIGHT * mat_diff_loss

    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

    ops = {
        'points_ph': points_ph,
        'labels_ph': labels_ph,
        'is_training_ph': is_training_ph,
        'pred': pred,
        'loss': loss,
    }

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, MODEL_PATH_FILE)
        eval_epoch(sess, test_data, test_labels, ops, test_writer)


if __name__ == '__main__':

    if not args.eval:  # training
        train_filenames = [x.rstrip() for x in open(args.train)]
        train_data, train_labels = load_data(train_filenames)

    test_filenames = [x.rstrip() for x in open(args.test)]
    test_data, test_labels = load_data(test_filenames)

    if not args.eval:
        train()
    else:
        evaluate()







