import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma.rms_normalization import RMSNormalization
from keras_hub.src.models.gemma3.gemma3_decoder_block import Gemma3DecoderBlock
from keras_hub.src.models.gemma3.gemma3_interleave_embeddings import (
    Gemma3InterleaveEmbeddings,
)
from keras_hub.src.models.gemma3.gemma3_mean_pooling import MeanPooling


@keras_hub_export("keras_hub.models.Gemma3Backbone")
class Gemma3Backbone(Backbone):
    """Gemma3 core network with hyperparameters."""

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
            embeddings_initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="untruncated_normal",
            ),
            dtype=dtype,
            logit_soft_cap=final_logit_soft_cap,
            name="token_embedding",
        )

        self.vision_encoder = vision_encoder
        text_only_model = True if vision_encoder is None else False
        if not text_only_model:
            self.interleave_embeddings = Gemma3InterleaveEmbeddings(
                num_vision_tokens_per_image=self.vision_encoder.num_vision_tokens_per_image,
                dtype=dtype,
                name="interleave_embeddings",
            )

        self.transformer_layers = []
        for i in range(num_layers):
            sliding_window = use_sliding_window_attention and (i % 6 < 5)
            rope_wavelength = 10_000.0 if sliding_window else 1_000_000.0
            rope_scaling_factor = (
                local_rope_scaling_factor
                if sliding_window
                else global_rope_scaling_factor
            )
            layer = Gemma3DecoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                query_head_dim_normalize=query_head_dim_normalize,
                use_query_key_norm=use_query_key_norm,
                use_post_ffw_norm=use_post_ffw_norm,
                use_post_attention_norm=use_post_attention_norm,
                gate_dim_reduction=1,
                logit_soft_cap=attention_logit_soft_cap,
                use_sliding_window_attention=sliding_window,
                sliding_window_size=sliding_window_size,
                rope_wavelength=rope_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                use_bidirectional_attention=use_bidirectional_attention,
                dropout=dropout,
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
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }

        if not text_only_model:
            image_input = keras.Input(
                shape=(None, image_size, image_size, 3), name="images"
            )
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )
            vision_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_mask"
            )
            inputs.update({
                "images": image_input,
                "vision_indices": vision_indices_input,
                "vision_mask": vision_mask_input,
            })

        # == Forward Pass ==
        x = self.token_embedding(token_id_input)
        x = x * ops.cast(ops.sqrt(hidden_dim), x.dtype)

        if not text_only_model:
            img_embeddings = self.vision_encoder(image_input)
            x = self.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=x,
                vision_indices=vision_indices_input,
            )

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                padding_mask=padding_mask_input,
                vision_mask=None if text_only_model else vision_mask_input,
            )
        sequence_output = self.layer_norm(x)

        if is_embedding_model:
            if embedding_dim is None or pooling_intermediate_dim is None:
                raise ValueError(
                    "`embedding_dim` and `pooling_intermediate_dim` must be "
                    "specified when `is_embedding_model` is `True`."
                )

            pooled_output = MeanPooling(dtype=dtype, name="mean_pooling")(
                [sequence_output, padding_mask_input]
            )

            pooled_output = layers.Dense(
                pooling_intermediate_dim,
                dtype=dtype,
                name="pooling_dense_1",
                use_bias=False,
            )(pooled_output)

            pooled_output = layers.Dense(
                embedding_dim,
                dtype=dtype,
                name="embedding_projection",
                use_bias=False,
            )(pooled_output)

            # --- L2 Normalization with Robust DType Handling ---
            pooled_output = ops.cast(pooled_output, "float32")
            l2_norm = ops.sqrt(
                ops.sum(ops.square(pooled_output), axis=-1, keepdims=True) + 1e-12
            )
            pooled_output = pooled_output / l2_norm
            
            # Manually resolve target_dtype to avoid AttributeError 
            # or dictionary hashing issues during __init__
            target_dtype = dtype
            if hasattr(target_dtype, "compute_dtype"):
                target_dtype = target_dtype.compute_dtype
            elif isinstance(target_dtype, dict):
                target_dtype = target_dtype.get("config", {}).get("name", "float32")
            
            if target_dtype is None:
                target_dtype = "float32"

            pooled_output = ops.cast(pooled_output, target_dtype)

            outputs = {
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            }
        else:
            outputs = sequence_output

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

        # === Config Storage ===
        self.vocabulary_size = vocabulary_size
        self.image_size = image_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.query_head_dim_normalize = query_head_dim_normalize
        self.use_query_key_norm = use_query_key_norm
        self.use_post_ffw_norm = use_post_ffw_norm
        self.use_post_attention_norm = use_post_attention_norm
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.final_logit_soft_cap = final_logit_soft_cap
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.local_rope_scaling_factor = local_rope_scaling_factor
        self.global_rope_scaling_factor = global_rope_scaling_factor
        self.use_bidirectional_attention = use_bidirectional_attention
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.is_embedding_model = is_embedding_model
        self.pooling_intermediate_dim = pooling_intermediate_dim
        self.embedding_dim = embedding_dim
        self.text_only_model = text_only_model

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "image_size": self.image_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "query_head_dim_normalize": self.query_head_dim_normalize,
                "use_query_key_norm": self.use_query_key_norm,
                "use_post_ffw_norm": self.use_post_ffw_norm,
                "use_post_attention_norm": self.use_post_attention_norm,
                "attention_logit_soft_cap": self.attention_logit_soft_cap,
                "final_logit_soft_cap": self.final_logit_soft_cap,
                "use_sliding_window_attention": self.use_sliding_window_attention,
                "sliding_window_size": self.sliding_window_size,
                "local_rope_scaling_factor": self.local_rope_scaling_factor,
                "global_rope_scaling_factor": self.global_rope_scaling_factor,
                "vision_encoder": layers.serialize(self.vision_encoder) if self.vision_encoder else None,
                "use_bidirectional_attention": self.use_bidirectional_attention,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "is_embedding_model": self.is_embedding_model,
                "pooling_intermediate_dim": self.pooling_intermediate_dim,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("vision_encoder") is not None:
            config["vision_encoder"] = layers.deserialize(config["vision_encoder"])
        return cls(**config)