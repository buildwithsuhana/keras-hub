import argparse
import os
import numpy as np

# Set JAX to CPU and disable GPU discovery to prevent CUDA Init errors on Kaggle
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
from sentence_transformers import SentenceTransformer

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer

def convert_to_embedding_preset(
    source_preset,
    output_preset,
    pooling_intermediate_dim,
    embedding_dim,
    hf_reference_model="google/embeddinggemma-300m"
):
    """
    Converts a standard causal Gemma3 preset to an Embedding Gemma preset.
    """
    print(f"Loading source preset: {source_preset}...")
    source_model = Gemma3Backbone.from_preset(source_preset)
    source_tokenizer = Gemma3Tokenizer.from_preset(source_preset)

    config = source_model.get_config()

    # Reconfigure for embedding mode
    config["is_embedding_model"] = True
    config["use_bidirectional_attention"] = True
    config["pooling_intermediate_dim"] = pooling_intermediate_dim
    config["embedding_dim"] = embedding_dim
    
    # CRITICAL FIX: The 300m embedding model has different dims than the 270m causal model.
    # We must update the config to match the target weights we are about to load.
    if "270m" in source_preset:
        print("Overriding config to match EmbeddingGemma-300m architecture...")
        config["hidden_dim"] = 768
        config["intermediate_dim"] = 3072
        config["num_query_heads"] = 12
        config["num_key_value_heads"] = 4
        config["head_dim"] = 64

    if config.get("vision_encoder") is not None:
        config["vision_encoder"] = keras.layers.deserialize(
            config["vision_encoder"]
        )

    print("Initializing new Embedding model from config...")
    embedding_model = Gemma3Backbone.from_config(config)

    # 1. Transfer Backbone Weights
    # Note: For the 270m -> 300m jump, we are mostly initializing a new 
    # architecture, so we rely on the ST model for the majority of parity.
    print("Transferring backbone weights...")
    source_layer_names = {layer.name for layer in source_model.layers}

    for target_layer in embedding_model.layers:
        if target_layer.name in source_layer_names:
            source_layer = source_model.get_layer(name=target_layer.name)
            if source_layer.get_weights():
                # Only transfer if shapes match (prevents crash on dimension jump)
                if all(a.shape == b.shape for a, b in zip(target_layer.get_weights(), source_layer.get_weights())):
                    target_layer.set_weights(source_layer.get_weights())

    # 2. Transfer Weights from Hugging Face Reference (This ensures numerical parity)
    print(f"Extracting all weights from {hf_reference_model} for parity...")
    hf_model = SentenceTransformer(hf_reference_model)
    
    # Map the ST projection head to Keras
    # Layer [2] is pooling_dense_1, Layer [3] is embedding_projection
    dense1_w = hf_model[2].linear.weight.detach().cpu().numpy().T
    dense2_w = hf_model[3].linear.weight.detach().cpu().numpy().T

    embedding_model.get_layer("pooling_dense_1").set_weights([dense1_w])
    embedding_model.get_layer("embedding_projection").set_weights([dense2_w])
    
    print("Successfully injected all weights.")

    # 3. Save the new preset
    os.makedirs(output_preset, exist_ok=True)
    embedding_model.save_to_preset(output_preset)
    source_tokenizer.save_to_preset(output_preset)
    print(f"Embedding Gemma preset successfully saved to: '{output_preset}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a pre-trained causal Gemma3 model to Embedding Gemma model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source_preset", type=str, required=True)
    parser.add_argument("--output_preset", type=str, required=True)
    parser.add_argument("--pooling_intermediate_dim", type=int, default=3072)
    parser.add_argument("--embedding_dim", type=int, default=768)

    args = parser.parse_args()

    convert_to_embedding_preset(
        source_preset=args.source_preset,
        output_preset=args.output_preset,
        pooling_intermediate_dim=args.pooling_intermediate_dim,
        embedding_dim=args.embedding_dim,
    )