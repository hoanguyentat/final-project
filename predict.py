import cv2
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './input/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './input/train',
                           """Directory where to read model checkpoints.""")


def get_single_img():
    file_path = './input/data/single/test_image.tif'
    pixels = cv2.imread(file_path, 0)
    return pixels


def eval_single_img():
    # below code adapted from @RyanSepassi, however not functional
    # among other errors, saver throws an error that there are no
    # variables to save
    with tf.Graph().as_default():

        # Get image.
        image = get_single_img()

        # Build a Graph.
        # TODO

        # Create dummy variables.
        x = tf.placeholder(tf.float32)
        w = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))
        b = tf.Variable(tf.ones([1, 1], dtype=tf.float32))
        y_hat = tf.add(b, tf.matmul(x, w))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Checkpoint found')
            else:
                print('No checkpoint found')

            # Run the model to get predictions
            predictions = sess.run(y_hat, feed_dict={x: image})
            print(predictions)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    eval_single_img()


if __name__ == '__main__':
    tf.app.run()
