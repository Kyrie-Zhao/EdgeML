import tensorflow as tf
import numpy as np
import time
tf.set_random_seed(123)
np.random.seed(123)

class Ddpg(object):
    def __init__(self, dis_dim, scal_dim, s_dim, var_scal, a_bound):
        """
        :param dis_dim: Dimension of discrete action
        :param scal_dim: Dimension of scalar action
        :param s_dim: Dmention of state
        :param var_scal: Scalar of standard division of variance
        :param a_bound: Bound of actions (is a list)
        """
        # Hyper parameter
        self.LR_A = 0.0001               # Learning rate for actor
        self.LR_C = 0.0001               # Learning rate for critic
        self.GAMMA = 0.9                # Reward discount
        self.TAU = 0.01                 # Soft replacement
        self.MEMORY_CAPACITY = 10000     # Memory capacity
        self.BATCH_SIZE = 64            # Learn batch size
        self.DECREASE_RATE = 0.995     # Var Decrease rate
        self.MIN_VAR = 0.001              # Min explore var

        # Memory and pointer
        # memory = [s, a, r, s_] x Memory length
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + dis_dim + scal_dim + 1),
                               dtype=np.float32)
        # Point memory piece
        self.pointer = 0

        # Init tf session
        self.sess = tf.Session()


        # Dimension bound and var
        self.dis_dim, self.scal_dim, self.s_dim = dis_dim, scal_dim, s_dim
        # Init a_bound
        self.a_bound = np.zeros([self.dis_dim + self.scal_dim])
        self.init_a_bound(a_bound)
        # Init vars
        self.var_scal = var_scal
        self.var = var_scal
        self.init_vars()

        # Placeholder for state, next state, and reward
        self.S = tf.placeholder(tf.float32, [None, s_dim], "s")
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], "s_")
        self.R = tf.placeholder(tf.float32, [None, 1], "r")

        # Actor construct
        with tf.variable_scope("Actor"):
            self.a = self._build_a(self.S, self.var, scope="eval", trainable=True)
            self.a_ = self._build_a(self.S_, self.var, scope="target", trainable=False)

        # Critic construct
        with tf.variable_scope("Critic"):
            self.q = self._build_c(self.S, self.a, scope="eval", trainable=True)
            self.q_ = self._build_c(self.S_, self.a_, scope="target", trainable=False)

        # Networks parameters
        # For soft replacement
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/eval")
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/target")
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/eval")
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/target")

        # Target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.TAU) * ta + self.TAU * ea),
                              tf.assign(tc, (1 - self.TAU) * tc + self.TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        # Calculate critic loss and one step train
        q_target = self.R + self.GAMMA * self.q_

        # In the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.train.RMSPropOptimizer(self.LR_C).minimize(td_error, var_list=self.ce_params)

        # Calculate actor loss and one step train
        a_loss = - tf.reduce_mean(self.q)
        self.atrain = tf.train.RMSPropOptimizer(self.LR_A).minimize(a_loss, var_list=self.ae_params)

        try:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, "./RL_checkpoint/ddpg")
            print("read")
        except Exception:
            self.run_init()
            self.save_model_checkpoint()
            print("save")

    def save_model_checkpoint(self):
        print("Saved RL")
        self.saver.save(self.sess, "./RL_checkpoint/ddpg")

    def run_init(self):
        # Run init
        self.sess.run(tf.global_variables_initializer())

    def init_vars(self):
        """Initial variable"""
        self.var = np.multiply(self.a_bound, self.var_scal)

    def init_a_bound(self, a_bound):
        """Initial a_bound, map dis_dim's bound to dis_dim dimension"""
        # Take out dis_dim bound
        bound_0 = a_bound[0]
        # Take out continuous_dim bound
        bound_1_n = a_bound[1:]
        # Init a_bound -> [[n_dim bound], continuous_dim bound]
        for i in range(self.dis_dim):
            self.a_bound[i] = bound_0
        for i in range(len(bound_1_n)):
            self.a_bound[self.dis_dim+i] = bound_1_n[i]

    def choose_action(self, s):
        """Choose action"""
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def store_transition(self, s, a, r, s_):
        """Store the transition into memory"""
        # Get transition
        transition = np.hstack((s, a, [r], s_))
        # Refresh pointer
        index = self.pointer % self.MEMORY_CAPACITY
        # Store and add pointer
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        """Take transition from memory and update the parm"""
        # Soft replace
        self.sess.run(self.soft_replace)

        # Choice batch size numbers indices for learn
        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        # Get batch transition
        bt = self.memory[indices, :]
        # Get batch s, a, r, s_
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.dis_dim + self.scal_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # Train actor and critic
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        # Decrease var but not lower than MIN_VAR
        self.var = np.maximum(self.var * self.DECREASE_RATE, np.multiply(self.a_bound, self.MIN_VAR))

    def _build_a(self, s, var, scope, trainable):
        """Build actor network"""
        with tf.variable_scope(scope):
            # hidden layer: 30 node
            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name="l1", trainable=trainable)
            a = tf.layers.dense(net, self.dis_dim+self.scal_dim,
                                activation=tf.nn.sigmoid, name="a", trainable=trainable)
            a = tf.multiply(a, self.a_bound)
            # add a mask for explore environment
            mask = tf.truncated_normal(tf.shape(a), 0.0, var)
            a = tf.clip_by_value(tf.add(a, mask),
                                 tf.cast(tf.zeros_like(self.a_bound), tf.float32),
                                 tf.cast(tf.constant(self.a_bound), tf.float32))
            return a

    def _build_c(self, s, a, scope, trainable):
        """Build critic network"""
        with tf.variable_scope(scope):
            # hidden layer: 300
            n_l1 = 300
            w1_s = tf.get_variable("w1_s", [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.scal_dim + self.dis_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable("b1", [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)


if __name__ == '__main__':
# when use:
    # init hyper parameters
    dis = 4
    scal = 2
    bound = [100, 10, 10]
    s_dim = 2
    var_scal=0.1

    # state simulate
    s__ = np.array([1, 2, 3])

    # init Ddpg class
    dpg = Ddpg(dis, scal, s_dim, var_scal, bound)

    # time1
    time_begin = time.time()

    # choose action
    action = dpg.choose_action(s__)
    print(action)

    # reward simulate
    r = 1

    # transition is [s__, action, r, s__]
    # store transition
    dpg.store_transition(s__, action, r, s__)

    # learn if you want
    dpg.learn()

    # total time
    time_step = time.time() - time_begin
    print(time_step)



"""


constrain: [pow_u, late_u]
pow_u: power constrain
late_u: latency constrain

state: [acc, acc_, pow, late]
acc: accuracy
acc_: last accuracy
power: power
late: latency

reward: tanh(M_ - M) - P
M_ = -log(acc_)
M  = -log(acc)
P  = max(0, 1 - log(pow - pow_u)) + max(0, 1 - log(late - late_u)
R  = tanh(M_ - M) - P

bound: [T_bound, P_bound, R_bound]
T_bound = 100    # T range
P_bound = 9     # P range
R_bound = 6     # R range

action_space: [T_n, P_n, R_n]
T_n = sigmoid(output[0])*T_bound    # threshold
P_n = sigmoid(output[1])*P_bound    # partation_point
R_n = sigmoid(output[2])*R_bound    # Round


"""
