import keras
from keras import layers
from keras import ops
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma.rms_normalization import RMSNormalization
from keras_hub.src.models.gemma3.gemma3_decoder_block import Gemma3DecoderBlock
from keras_hub.src.models.gemma3.gemma3_mean_pooling import MeanPooling

@keras_hub_export("keras_hub.models.Gemma3Backbone")
class Gemma3Backbone(Backbone):
    def __init__(
        self,
        vocabulary_size,
        image_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=1024,
        local_rope_scaling_factor=1.0,
        global_rope_scaling_factor=1.0,
        vision_encoder=None,
        layer_norm_epsilon=1e-6,
        use_bidirectional_attention=False,
        dropout=0,
        is_embedding_model=False,
        pooling_intermediate_dim=None,
        embedding_dim=None,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=True,
            dtype=dtype,
            name="token_embedding",
        )

        self.transformer_layers = []
        for i in range(num_layers):
            layer = Gemma3DecoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                use_bidirectional_attention=use_bidirectional_attention,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )
            self.transformer_layers.append(layer)

        self.layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_normalization",
        )

        # == Model inputs ==
        token_id_input = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask_input = keras.Input(shape=(None,), dtype="int32", name="padding_mask")
        inputs = {"token_ids": token_id_input, "padding_mask": padding_mask_input}

        # == Forward Pass ==
        x = self.token_embedding(token_id_input)
        x = x * ops.cast(ops.sqrt(hidden_dim), x.dtype)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)
        
        sequence_output = self.layer_norm(x)

        if is_embedding_model:
            # 1. Mean Pooling
            pooled_output = MeanPooling(dtype=dtype, name="mean_pooling")(
                [sequence_output, padding_mask_input]
            )

            # 2. Dense Projection Head
            pooled_output = layers.Dense(
                pooling_intermediate_dim, dtype=dtype, name="pooling_dense_1", use_bias=False
            )(pooled_output)
            pooled_output = layers.Dense(
                embedding_dim, dtype=dtype, name="embedding_projection", use_bias=False
            )(pooled_output)

            # 3. L2 Normalization (Essential for Parity)
            pooled_output = ops.cast(pooled_output, "float32")
            l2_norm = ops.sqrt(ops.sum(ops.square(pooled_output), axis=-1, keepdims=True) + 1e-12)
            pooled_output = pooled_output / l2_norm
            
            # Use safe dtype resolution for Keras 3
            target_dtype = dtype.compute_dtype if hasattr(dtype, "compute_dtype") else (dtype if dtype else "float32")
            if isinstance(target_dtype, dict): target_dtype = "float32"
            pooled_output = ops.cast(pooled_output, target_dtype)

            outputs = {"sequence_output": sequence_output, "pooled_output": pooled_output}
        else:
            outputs = sequence_output

        super().__init__(inputs=inputs, outputs=outputs, dtype=dtype, **kwargs)
        self.image_size = image_size
        self.is_embedding_model = is_embedding_model

    def get_config(self):
        config = super().get_config()
        config.update({"image_size": self.image_size, "is_embedding_model": self.is_embedding_model})
        return config