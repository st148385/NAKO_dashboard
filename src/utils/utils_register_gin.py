import gin
import tensorflow as tf


@gin.configurable
def sparse_categorical_loss(
	from_logits: bool = False,
	ignore_class: int = None,
	reduction: str = "sum_over_batch_size",
	name: str = "sparse_categorical_crossentropy",
) -> tf.keras.losses.SparseCategoricalCrossentropy:
	"""
	Creates and returns a SparseCategoricalCrossentropy loss function for Keras/TensorFlow.

	This function serves as a Gin-configurable wrapper for the
	`tf.keras.losses.SparseCategoricalCrossentropy` loss function. It allows you
	to conveniently set the parameters of the loss function through your Gin configuration files.

	:param from_logits: Whether the model outputs logits (raw, unnormalized scores) or probabilities. Default is False.
	:type from_logits: bool
	:param ignore_class: Optional integer value to ignore when calculating the loss. Default is None.
	:type ignore_class: int, optional
	:param reduction: Type of reduction to apply to the loss.
	                  See `tf.keras.losses.Reduction` for valid options. Default is "sum_over_batch_size".
	:type reduction: str
	:param name: Name for the operation. Default is "sparse_categorical_crossentropy".
	:type name: str
	:return: An instance of ``tf.keras.losses.SparseCategoricalCrossentropy``.
	:rtype: tf.keras.losses.SparseCategoricalCrossentropy

	**Example (Gin Configuration):**

	.. code-block::
		sparse_categorical_loss.from_logits = True
	    Runner.loss = @sparse_categorical_loss()
	"""

	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
		from_logits=from_logits,
		ignore_class=ignore_class,
		reduction=reduction,
		name=name,
	)
	return loss_object


@gin.configurable
def adam_optimizer(
	learning_rate: float = 0.001,
	beta_1: float = 0.9,
	beta_2: float = 0.999,
	epsilon: float = 1e-07,
	amsgrad: bool = False,
	name: str = "Adam",
	**kwargs,
):
	"""
	Creates and returns an Adam optimizer with configurable parameters.

	:param learning_rate: The learning rate.
	:type learning_rate: float
	:param beta_1: The exponential decay rate for the first moment estimates.
	:type beta_1: float
	:param beta_2: The exponential decay rate for the second moment estimates.
	:type beta_2: float
	:param epsilon: A small constant for numerical stability.
	:type epsilon: float
	:param amsgrad: Whether to apply the AMSGrad variant of Adam.
	:type amsgrad: bool
	:param name: The name of the optimizer.
	:type name: str
	:param \**kwargs: Additional keyword arguments to pass to the optimizer constructor.
	:return: An Adam optimizer instance.
	:rtype: tf.keras.optimizers.Adam

	"""
	optimizer = tf.keras.optimizers.Adam(
		learning_rate=learning_rate,
		beta_1=beta_1,
		beta_2=beta_2,
		epsilon=epsilon,
		amsgrad=amsgrad,
		name=name,
		**kwargs,  # Additional keyword arguments
	)
	return optimizer
