
from __future__ import print_function

import tensorflow as tf
try:
  tf.contrib.eager.enable_eager_execution()
  print("TF imported with eager execution!")
except ValueError:
  print("TF already imported with eager execution!")

primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print("primes:", primes)
ones = tf.ones([6], dtype=tf.int32)
print("ones:", ones)
just_beyond_primes = tf.add(primes, ones)
print("just_beyond_primes:", just_beyond_primes)
twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
primes_doubled = primes * twos
print("primes_doubled:", primes_doubled)

some_matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
print(some_matrix.numpy())

#Ex Broadcasting:
one = tf.constant(1, dtype=tf.int32)
print("one:", one)
just_beyond_primes = tf.add(primes, one)
print("just_beyond_primes:", just_beyond_primes)
two = tf.constant(2, dtype=tf.int32)
primes_doubled = primes * two
print("primes_doubled:", primes_doubled)

