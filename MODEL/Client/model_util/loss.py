import tensorflow as tf

def cross_entropy(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, 
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy_, name='cross_entropy')
    return cross_entropy_mean

def top_k_error(predictions, labels, k=1):
    batch_size = predictions.get_shape().as_list()[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    return num_correct / float(batch_size)