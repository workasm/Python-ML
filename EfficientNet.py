import collections
import functools, re
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as tfl
import tensorflow.keras.backend as K
from IPython.display import display, Image

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'se_coefficient', 'local_pooling', 'condconv_num_experts',
    'clip_projection_output', 'blocks_args', 'fix_head_stem', 'use_bfloat16'
])
# Note: the default value of None is not necessarily valid. It is valid to leave
# width_coefficient, depth_coefficient at None, which is treated as 1.0 (and
# which also allows depth_divisor and min_depth to be left at None).
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'fused_conv',
    'condconv', 'activation_fn'
])
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
  """A class of MBConv: Mobile Inverted Residual Bottleneck.
  """

  def __init__(self, block_args, global_params):
    super(MBConvBlock, self).__init__()
    self._block_args = block_args
    self._local_pooling = global_params.local_pooling
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    # NOTE this could use TPU-specific batch norm
    self._batch_norm = tfl.BatchNormalization
    self._condconv_num_experts = global_params.condconv_num_experts
    self._data_format = global_params.data_format
    self._se_coefficient = global_params.se_coefficient

    self._relu_fn = (self._block_args.activation_fn
                     or global_params.relu_fn or tf.nn.swish)
    self._has_se = (
        global_params.use_se and self._block_args.se_ratio is not None and
        0 < self._block_args.se_ratio <= 1)

    self._clip_projection_output = global_params.clip_projection_output

    self.endpoints = None

    #self.conv_cls = tfl.Conv2D
    #self.depthwise_conv_cls = tfl.DepthwiseConv2D
    # TODO: disable condconv layers for now..
    #if self._block_args.condconv:
    #  self.conv_cls = functools.partial(
    #      condconv_layers.CondConv2D, num_experts=self._condconv_num_experts)
    #  self.depthwise_conv_cls = functools.partial(
    #      condconv_layers.DepthwiseCondConv2D,
    #      num_experts=self._condconv_num_experts)

    # Builds the block accordings to arguments.
    self._build()

  def block_args(self):
    return self._block_args

  def _build(self):
    
    if self._block_args.condconv:
      # Add the example-dependent routing function
      self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
      self._routing_fn = tf.layers.Dense(
          self._condconv_num_experts, activation=tf.nn.sigmoid)

    filters = self._block_args.input_filters * self._block_args.expand_ratio
    kernel_size = self._block_args.kernel_size

    # Fused expansion phase. Called if using fused convolutions.
    self._fused_conv = tfl.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=self._block_args.strides,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)

    # Expansion phase. Called if not using fused convolutions and expansion
    # phase is necessary.
    self._expand_conv = tfl.Conv2D(
        filters=filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)
        
    self._bn0 = self._batch_norm(
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

    # Depth-wise convolution phase. Called if not using fused convolutions.
    self._depthwise_conv = tfl.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)

    self._bn1 = self._batch_norm(
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

    if self._has_se:
      num_reduced_filters = int(self._block_args.input_filters * (
          self._block_args.se_ratio * (self._se_coefficient
                                       if self._se_coefficient else 1)))

      num_reduced_filters = max(1, num_reduced_filters)
      tf.print(num_reduced_filters)

      # Squeeze and Excitation layer.
      self._se_reduce = tfl.Conv2D(
          num_reduced_filters,
          kernel_size=1,
          strides=1,
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True)

      self._se_expand = tfl.Conv2D(
          filters,
          kernel_size=1,
          strides=1,
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          activation='sigmoid',
          use_bias=True)

    # Output phase.
    filters = self._block_args.output_filters
    self._project_conv = tfl.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)

    self._bn2 = self._batch_norm(
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

  def _call_se(self, input):
    """Call Squeeze and Excitation layer.
    """
    x = tfl.GlobalAveragePooling2D(keepdims=True)(input)
    x = self._se_reduce(x)
    x = self._relu_fn(x)
    x = self._se_expand(x)
    return x * input

  def call(self, inputs, training=True, survival_prob=None):
    x = inputs

    fused_conv_fn = self._fused_conv
    expand_conv_fn = self._expand_conv
    depthwise_conv_fn = self._depthwise_conv
    project_conv_fn = self._project_conv

    # if self._block_args.condconv:
    #   pooled_inputs = self._avg_pooling(inputs)
    #   routing_weights = self._routing_fn(pooled_inputs)
    #   # Capture routing weights as additional input to CondConv layers
    #   fused_conv_fn = functools.partial(
    #       self._fused_conv, routing_weights=routing_weights)
    #   expand_conv_fn = functools.partial(
    #       self._expand_conv, routing_weights=routing_weights)
    #   depthwise_conv_fn = functools.partial(
    #       self._depthwise_conv, routing_weights=routing_weights)
    #   project_conv_fn = functools.partial(
    #       self._project_conv, routing_weights=routing_weights)

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
      x = self._call_se(x)

    self.endpoints = {'expansion_output': x}

    x = self._bn2(project_conv_fn(x), training=training)
    # Add identity so that quantization-aware training can insert quantization
    # ops correctly.
    x = tf.identity(x)
    if self._clip_projection_output:
      x = tf.clip_by_value(x, -6, 6)
    if self._block_args.id_skip:
      tf.print(f"Using ID skip: {self._block_args.strides}; shapes: {inputs.shape} and {x.shape}")
      if all(
          s == 1 for s in self._block_args.strides
      ) and inputs.shape[-1] == x.shape[-1]:
        # Apply only if skip connection presents.
        if survival_prob:
          x = drop_connect(x, training, survival_prob)
        x = tf.add(x, inputs)
    return x

  def summary(self, input_shape):
    x = tfl.Input(shape=input_shape)
    M = tf.keras.Model(inputs=x, outputs=self.call(x), name='actor')
    print(M.summary())
    tf.keras.utils.plot_model(M, to_file='model.png', show_shapes=True, 
          show_layer_activations=True, show_layer_names=True)
    display(Image('model.png'))


class BlockDecoder(object):
  """Block Decoder for readability."""

  def _decode_block_string(self, block_string):
    """Gets a block through a string notation of arguments."""
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

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.
    Args:
      string_list: a list of strings, each string is a notation of block.
    Returns:
      A list of namedtuples to represent blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args
