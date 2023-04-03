import collections
import functools, re
import tensorflow as tf
import numpy as np
from keras import layers as tfl
from keras import backend as K

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 
    'survival_prob', 'relu_fn', 'use_se',
    'se_coefficient', 'clip_projection_output'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'fused_conv',
    'condconv', 'activation_fn'
])
# example string: 'r1_k3_s11_e6_i192_o320_se0.25'
# r - num repeats - actually not used but can be adapted to generate several blocks
# k - conv kernel size
# i - # of input filters for _expand_conv or _fused_conv
# o - # of output filters for _project_conv
# e - expand ratio: filters = input_filters * expand_ratio 
# noskip - whether to add skip connection (default: yes)
# se - squeeze-and-excitation ratio
# s - conv strides: these are actually two numbers without space, hence s11 is default !!
# f - whether to use fused convolutions instead of depthwise separable convs
# cc - whether to use condconv (NYI)
# a - activation function: a0 - relu, a1 - swish, else - None

# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.
    Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused
  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random.normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = tf.div(inputs, survival_prob) * binary_tensor
  return output

class MBConvBlock(tfl.Layer):
  def __init__(self, block_args, global_params):
    super(MBConvBlock, self).__init__()
    self._block_args = block_args
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    # NOTE this could use TPU-specific batch norm
    self._batch_norm = tfl.BatchNormalization
    self._se_coefficient = global_params.se_coefficient

    self._relu_fn = (self._block_args.activation_fn
                     or global_params.relu_fn or tf.nn.swish)
    self._has_se = (
        global_params.use_se and self._block_args.se_ratio is not None and
        0 < self._block_args.se_ratio <= 1)

    self._clip_projection_output = global_params.clip_projection_output
    self.endpoints = None

    # Builds the block accordings to arguments.
    self._build()

  def _build(self):
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    kernel_size = self._block_args.kernel_size

    # Fused expansion phase. Called if using fused convolutions.
    self._fused_conv = tfl.Conv2D(
        filters=filters, kernel_size=kernel_size,
        strides=self._block_args.strides,
        kernel_initializer=conv_kernel_initializer,
        padding='same', use_bias=False)

    # Expansion phase. Called if not using fused convolutions and expansion
    # phase is necessary.
    self._expand_conv = tfl.Conv2D(
        filters=filters, kernel_size=[1, 1], strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same', use_bias=False)
        
    self._bn0 = self._batch_norm(
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

    # Depth-wise convolution phase. Called if not using fused convolutions.
    self._depthwise_conv = tfl.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same', use_bias=False)

    self._bn1 = self._batch_norm(
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

    if self._has_se:
      num_reduced_filters = int(self._block_args.input_filters * (
          self._block_args.se_ratio * (self._se_coefficient
                                       if self._se_coefficient else 1)))
      num_reduced_filters = max(1, num_reduced_filters)

      # Squeeze and Excitation layer.
      self._se_reduce = tfl.Conv2D(
          num_reduced_filters, kernel_size=1, strides=1,
          kernel_initializer=conv_kernel_initializer,
          padding='same', use_bias=True)

      self._se_expand = tfl.Conv2D(
          filters, kernel_size=1, strides=1,
          kernel_initializer=conv_kernel_initializer,
          padding='same', activation='sigmoid', use_bias=True)

    # Output phase.
    filters = self._block_args.output_filters
    self._project_conv = tfl.Conv2D(
        filters=filters, kernel_size=1, strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same', use_bias=False)

    self._bn2 = self._batch_norm(
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

  def call(self, inputs, training=True, survival_prob=None):
    x = inputs

    fused_conv_fn = self._fused_conv
    expand_conv_fn = self._expand_conv
    depthwise_conv_fn = self._depthwise_conv
    project_conv_fn = self._project_conv

    if self._block_args.fused_conv:
      # If use fused mbconv, skip expansion and use regular conv.
      x = fused_conv_fn(x)
    else:
      # Otherwise, first apply expansion and then apply depthwise conv.
      if self._block_args.expand_ratio != 1:
        x = self._relu_fn(self._bn0(expand_conv_fn(x), training=training))
      x = depthwise_conv_fn(x)

    x = self._relu_fn(self._bn1(x, training=training))

    if self._has_se:
      input = x
      x = tfl.GlobalAveragePooling2D(keepdims=True)(input)
      x = input * self._se_expand(self._relu_fn(self._se_reduce(x)))

    self.endpoints = {'expansion_output': x}
    x = self._bn2(project_conv_fn(x), training=training)
    # Add identity so that quantization-aware training can insert quantization
    # ops correctly.
    # x = tf.identity(x)
    if self._clip_projection_output:
      x = tf.clip_by_value(x, -6, 6)
    if self._block_args.id_skip:
      #tf.print(f"Using ID skip: {self._block_args.strides}; shapes: {inputs.shape} and {x.shape}")
      if all(s == 1 for s in self._block_args.strides) \
          and inputs.shape[-1] == x.shape[-1]:
        # Apply only if skip connection presents.
        if survival_prob:
          x = drop_connect(x, training, survival_prob)
        x = tf.add(x, inputs)
    return x

  def get_summary(self, input_shape):
    x = tfl.Input(shape=input_shape)
    return tf.keras.Model(inputs=x, outputs=self.call(x), name='actor')

class BlockDecoder(object):

  def _decode_block_string(self, block_string):
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    return BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]),
                 int(options['s'][1])],
        fused_conv=int(options['f']) if 'f' in options else 0,
        condconv=('cc' in block_string),
        activation_fn=(tf.nn.relu if int(options['a']) == 0
                       else tf.nn.swish) if 'a' in options else None)

  def __call__(self, string_list):
    return [self._decode_block_string(s) for s in string_list]

class Model:
  def __init__(self, global_args, block_string):
    block_args = BlockDecoder()(block_string)

    self.global_args = global_args
    self.blocks = []
    for blk in block_args:
      for _ in range(blk.num_repeat):
        self.blocks.append(MBConvBlock(blk, global_args))

  def __call__(self, inputs, training=None):
    X = inputs
    survival_prob = self.global_args.survival_prob
    for block in self.blocks:
      X = block.call(X, training=training, survival_prob=survival_prob)
    
    return tf.keras.Model(inputs=inputs, outputs=X)
