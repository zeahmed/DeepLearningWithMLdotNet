
import tensorflow as tf

f_size = 15 # Number of features passed from ML.Net
num_output = 2 # Number of outputs
tf.set_random_seed(1)
X = tf.placeholder('float', [None, f_size], name="X")
Y = tf.placeholder('float', [None, num_output], name="Y")
lr = tf.placeholder(tf.float32, name = "learning_rate")


# Set model weights
W = tf.Variable(tf.random_normal([f_size,num_output], stddev=0.1), name = 'W')
b = tf.Variable(tf.zeros([num_output]), name = 'b')

l1 = 0
l2 = 0
RegScores = tf.add(tf.matmul(X, W), b, name='RegScores')
loss = tf.reduce_mean(tf.square(Y-tf.squeeze(RegScores))) / 2  + l2 * tf.nn.l2_loss(W) + l1 * tf.reduce_sum(tf.abs(W))
loss = tf.identity(loss, name="Loss")
optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9, name='MomentumOptimizer').minimize(loss)

init = tf.global_variables_initializer()
# Launch the graph.
with tf.Session() as sess:
    sess.run(init)
    tf.saved_model.simple_save(sess, r'NYCTaxi/model', inputs={'X': X, 'Y': Y}, outputs={'RegScores': RegScores} )