import logging
import time
import numpy as np
import tensorflow as tf
import cv2
from convnet import convnet
from data import data_loader
from adamax import AdamaxOptimizer

flags = tf.app.flags
FLAGS = flags.FLAGS
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

flags.DEFINE_string('result_dir', '/ifp/users/haichao/projects/camid/results/exp2', 'for saving results')
flags.DEFINE_boolean('use_pretrained', False, 'If true, load pretrained model.')
flags.DEFINE_string('model_path', 'None', 'Where pretrained model is saved.')
flags.DEFINE_string('train_list', '/ifp/users/haichao/projects/camid/data/dresden/filelist_train', 'File list of training data.')
flags.DEFINE_string('val_list', '/ifp/users/haichao/projects/camid/data/dresden/filelist_valid', 'File list of validation data.')
flags.DEFINE_integer('num_val_data', '371', 'Input size')
flags.DEFINE_integer('batch_size', '128', 'Bach size of Training')
flags.DEFINE_integer('num_epochs', '100', 'Max epoch number')
flags.DEFINE_float('learning_rate', '0.015', 'Learning rate')
flags.DEFINE_boolean('mem_growth', True, 'If true, use gpu memory on demand.')
flags.DEFINE_integer('disp_steps', '10', 'Display interval')
flags.DEFINE_integer('val_steps', '500', 'Display interval')
flags.DEFINE_integer('ckpt_steps', '500', 'Checkpoint saving interval')

def train():
    with tf.Graph().as_default():
        # train data provider
        input_, label = data_loader(FLAGS.train_list, FLAGS.num_epochs)
        input_shuffle, label_shuffle = tf.train.shuffle_batch([input_, label], batch_size=FLAGS.batch_size, num_threads=8, capacity=512, min_after_dequeue=256)
        label_shuffle = tf.expand_dims(label_shuffle, -1)

        # val data provider
        input_val, label_val = data_loader(FLAGS.val_list, None)
        input_batch, label_batch = tf.train.batch([input_val, label_val], batch_size=FLAGS.batch_size)
        label_batch = tf.expand_dims(label_batch, -1)

        input_pl = tf.placeholder_with_default(input_shuffle, shape=[FLAGS.batch_size, 64, 64, 3])
        label_pl = tf.placeholder_with_default(label_shuffle, shape=[FLAGS.batch_size, 1])
        learning_rate = tf.placeholder(tf.float32, shape=())

        logits = convnet(input_pl, 26)

        predictions = tf.argmax(logits, axis=-1)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label_pl, logits=logits)
        accu = tf.cast(tf.equal(tf.cast(label_pl, tf.int64), predictions), tf.float32)
        accu = tf.reduce_mean(accu)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        #train_op = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9).minimize(loss, global_step=global_step)
        #train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)
        #train_op = AdamaxOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)

        '''
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = FLAGS.mem_growth
        sess = tf.Session(config=config)
        '''
        sess = tf.Session()

        saver = tf.train.Saver(save_relative_paths=True, max_to_keep=100)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            start_time = time.time()
            learning_rate_i = FLAGS.learning_rate * (0.5 ** (step // 50000))
            while not coord.should_stop():
                if step % FLAGS.val_steps == 0:
                    loss_sum = 0
                    accu_sum = 0
                    iters = FLAGS.num_val_data // FLAGS.batch_size
                    for i in range(iters):
                        input_v, label_v = sess.run([input_batch, label_batch])
                        loss_v, accu_v = sess.run([loss, accu], feed_dict={input_pl: input_v, label_pl: label_v})
                        loss_sum += loss_v
                        accu_sum += accu_v
                    aver_loss_val = loss_sum / iters
                    aver_accu_val = accu_sum / iters
                    logging.info('step %d: val loss %.4f, accu %.4f' % (step, aver_loss_val, aver_accu_val))

                if step % FLAGS.disp_steps == (FLAGS.disp_steps-1):
                    _, loss_v, accu_v = sess.run([train_op, loss, accu], feed_dict={learning_rate: learning_rate_i})
                    duration = time.time() - start_time
                    logging.info(('step %d: loss %.4f, accu %.4f (%.1f s).') % (step, loss_v, accu_v, duration))
                else:
                    _, = sess.run([train_op], feed_dict={learning_rate: learning_rate_i})

                if step % FLAGS.ckpt_steps == (FLAGS.ckpt_steps - 1):
                    save_path = FLAGS.result_dir + '/ckpt/check_point'
                    saver.save(sess, save_path, global_step=step)
                    logging.info('Model saved to ' + save_path)
                step += 1
        except tf.errors.OutOfRangeError:
            logging.info('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            coord.request_stop()
            coord.join(threads)

        save_path = FLAGS.result_dir + '/ckpt/check_point'
        saver.save(sess, save_path, global_step=step)
        logging.info('Model saved to ' + save_path)
        sess.close()


if __name__ == '__main__':
    train()
