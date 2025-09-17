import itertools

import keras
from keras import ops
from keras import tree

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Embedder")
class Embedder(Task):
    """Base class for embedding tasks.

    `Embedder` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used
    for generating embeddings from input data (e.g., text, images).

    `Embedder` tasks provide a high-level `embed()` function
    which can be used to get embedding vectors for raw inputs (e.g., strings in,
    vectors out).

    All `Embedder` tasks include a `from_preset()` constructor which can be used
    to load a pre-trained config and weights.

    Example:
    ```python
    # Load a Gemma embedding model (preset name is fictional)
    embedder = keras_hub.models.Embedder.from_preset(
        "gemma_2b_en_embedder",
    )
    embeddings = embedder.embed("Keras is a")
    print(embeddings.shape)
    ```
    """

    def __init__(self, *args, **kwargs):
        """Initializes the `Embedder` task.

        Args:
            *args: Arguments passed to the base `Task` class.
            **kwargs: Keyword arguments passed to the base `Task` class.
        """
        super().__init__(*args, **kwargs)

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        **kwargs,
    ):
        """Configures the `Embedder` task for training.

        The `Embedder` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `weighted_metrics`. By default, `loss` and `weighted_metrics` are
        set to `None` as embedders are often used for inference or with
        custom contrastive losses (like `keras.losses.CosineSimilarity`).

        To override these defaults, pass any value to these arguments.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses `Adam(2e-5)`.
                See `keras.Model.compile` and `keras.optimizers` for more.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, which resolves to `None`. This is
                suitable for inference, but a loss must be provided
                for training. See `keras.Model.compile` and `keras.losses`
                for more.
            weighted_metrics: `"auto"`, or a list of metrics. Defaults to
                `"auto"`, which resolves to `None`. See `keras.Model.compile`
                and `keras.metrics` for more.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(2e-5)
        if loss == "auto":
            loss = None
        if weighted_metrics == "auto":
            weighted_metrics = None
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            **kwargs,
        )
        self.embed_function = None

    def embed_step(self, inputs):
        raise NotImplementedError

    def make_embed_function(self):
        """Create or return the compiled embedding function."""
        if self.embed_function is not None:
            return self.embed_function

        self.embed_function = self.embed_step

        if keras.config.backend() == "openvino":
            from keras_hub.src.utils.openvino_utils import ov_infer

            if ov_infer is None:
                raise ImportError(
                    "OpenVINO backend is selected, but OpenVINO is not "
                    "installed or `openvino_utils.py` is not found."
                )

            def fn_wrapper(inputs, stop_token_ids):
                return self.embed_step(inputs)

            def wrapped_embed_function(inputs):
                inputs = tree.map_structure(ops.array, inputs)
                return ov_infer(
                    self, inputs, stop_token_ids=None, fn=fn_wrapper
                )

            self.embed_function = wrapped_embed_function

        elif keras.config.backend() == "torch":
            import torch

            def wrapped_embed_function(inputs):
                with torch.no_grad():
                    return self.embed_step(inputs)

            self.embed_function = wrapped_embed_function

        elif keras.config.backend() == "tensorflow" and not self.run_eagerly:
            jit_compile = getattr(self, "jit_compile", True)
            self.embed_function = tf.function(
                self.embed_step, jit_compile=jit_compile
            )

        elif keras.config.backend() == "jax" and not self.run_eagerly:
            import jax

            @jax.jit
            def compiled_embed_function(inputs, state):
                (
                    trainable_variables,
                    non_trainable_variables,
                ) = state
                mapping = itertools.chain(
                    zip(self.trainable_variables, trainable_variables),
                    zip(self.non_trainable_variables, non_trainable_variables),
                )

                with keras.StatelessScope(state_mapping=mapping):
                    outputs = self.embed_step(inputs)

                return outputs

            def wrapped_embed_function(inputs):
                state = (
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                inputs = tree.map_structure(ops.convert_to_tensor, inputs)
                outputs = compiled_embed_function(
                    inputs,
                    state,
                )
                return outputs

            self.embed_function = wrapped_embed_function

        return self.embed_function

    def _normalize_embed_inputs(
        self,
        inputs,
    ):
        """Normalize user input to the embed function.

        This function converts all inputs to tensors, adds a batch dimension if
        necessary, and returns a iterable "dataset like" object (either an
        actual `tf.data.Dataset` or a list with a single batch element).
        """
        if tf and isinstance(inputs, tf.data.Dataset):
            return inputs.as_numpy_iterator(), False

        if self.preprocessor is None:
            return [inputs], False

        def normalize(x):
            if isinstance(x, str):
                return [x], True
            if tf and isinstance(x, tf.Tensor) and x.shape.rank == 0:
                return x[tf.newaxis], True
            return x, False

        if isinstance(inputs, dict):
            for key in inputs:
                inputs[key], input_is_scalar = normalize(inputs[key])
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

    def _normalize_embed_outputs(
        self,
        outputs,
        input_is_scalar,
    ):
        """Normalize user output from the embed function.

        This function converts all output to numpy. If a batch dimension was
        added to the input, it is removed from the output (so embed can be
        string in, vector out).
        """

        def normalize(x):
            if isinstance(x[0], list):
                outputs = []
                for batch in x:
                    for e in batch:
                        outputs.append(e)
                return outputs[0] if input_is_scalar else outputs
            outputs = ops.concatenate(x, axis=0)
            outputs = ops.squeeze(outputs, 0) if input_is_scalar else outputs
            return ops.convert_to_numpy(outputs)

        if isinstance(outputs[0], dict):
            normalized = {}
            for key in outputs[0]:
                normalized[key] = normalize([x[key] for x in outputs])
            return normalized

        return normalize([x for x in outputs])

    def embed(
        self,
        inputs,
    ):
        """Generate embeddings for given `inputs`.

        This method generates embeddings based on given `inputs`.

        If `inputs` are a `tf.data.Dataset`, outputs will be generated
        "batch-by-batch" and concatenated. Otherwise, all inputs will be handled
        as a single batch.

        If a `preprocessor` is attached to the model, `inputs` will be
        preprocessed inside the `embed()` function and should match the
        structure expected by the `preprocessor` layer (usually raw strings).
        If a `preprocessor` is not attached, inputs should match the structure
        expected by the `backbone`.

        Args:
            inputs: python data, tensor data, or a `tf.data.Dataset`. If a
                `preprocessor` is attached to the model, `inputs` should match
                the structure expected by the `preprocessor` layer. If a
                `preprocessor` is not attached, `inputs` should match the
                structure expected the `backbone` model.

        Returns:
            A numpy array or dictionary of numpy arrays containing the
            embeddings.
        """
        embed_function = self.make_embed_function()

        def preprocess(x):
            if self.preprocessor is None:
                return x
            if hasattr(self.preprocessor, "generate_preprocess"):
                return self.preprocessor.generate_preprocess(x)
            return self.preprocessor(x)

        def postprocess(x):
            return x

        inputs, input_is_scalar = self._normalize_embed_inputs(inputs)

        if self.preprocessor is not None:
            inputs = [preprocess(x) for x in inputs]

        outputs = [embed_function(x) for x in inputs]

        if self.preprocessor is not None:
            outputs = [postprocess(x) for x in outputs]

        return self._normalize_embed_outputs(outputs, input_is_scalar)

    def export_to_transformers(self, path):
        """Export the full Embedder model to HuggingFace Transformers format.

        This exports the trainable model, tokenizer, and configurations in a
        format compatible with HuggingFace Transformers. For unsupported model
        architectures, a ValueError is raised.

        If the preprocessor is attached (default), both the trainable model and
        tokenizer are exported. To export only the trainable model, set
        `self.preprocessor = None` before calling this method, then export the
        preprocessor separately via `preprocessor.export_to_transformers(path)`.

        Args:
            path: str. Path to save the exported model.
        """
        from keras_hub.src.utils.transformers.export.hf_exporter import (
            export_to_safetensors,
        )

        export_to_safetensors(self, path)
