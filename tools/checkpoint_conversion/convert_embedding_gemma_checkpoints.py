import argparse
import os
import numpy as np

# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    if config.get("vision_encoder") is not None:
        config["vision_encoder"] = keras.layers.deserialize(
            config["vision_encoder"]
        )

    print("Initializing new Embedding model from config...")
    embedding_model = Gemma3Backbone.from_config(config)

    # 1. Transfer Backbone Weights (Transformer Layers & Embeddings)
    print("Transferring backbone weights...")
    transferred_layers = 0
    source_layer_names = {layer.name for layer in source_model.layers}

    for target_layer in embedding_model.layers:
        if target_layer.name in source_layer_names:
            source_layer = source_model.get_layer(name=target_layer.name)
            if source_layer.get_weights():
                target_layer.set_weights(source_layer.get_weights())
                transferred_layers += 1

    # 2. Transfer Pooling Head Weights from Hugging Face Reference
    # This is necessary because the causal model has no pooling head weights.
    print(f"Extracting head weights from {hf_reference_model} for parity...")
    hf_model = SentenceTransformer(hf_reference_model)
    
    # HF Linear layer weights are (out_dim, in_dim). Keras expects (in_dim, out_dim).
    # Layer [2] and [3] in the ST Sequential correspond to the two dense projections.
    dense1_w = hf_model[2].linear.weight.detach().cpu().numpy().T
    dense2_w = hf_model[3].linear.weight.detach().cpu().numpy().T

    embedding_model.get_layer("pooling_dense_1").set_weights([dense1_w])
    embedding_model.get_layer("embedding_projection").set_weights([dense2_w])
    print("Successfully injected pooling head weights.")

    # 3. Save the new preset
    os.makedirs(output_preset, exist_ok=True)
    embedding_model.save_to_preset(output_preset)
    source_tokenizer.save_to_preset(output_preset)
    print(f"Embedding Gemma preset successfully saved to: '{output_preset}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a pre-trained causal Gemma3 model to "
        "Embedding Gemma model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source_preset",
        type=str,
        required=True,
        help="Path or name of the source causal Gemma3 preset.",
    )
    parser.add_argument(
        "--output_preset",
        type=str,
        required=True,
        help="Path to save the new Embedding Gemma preset.",
    )
    parser.add_argument(
        "--pooling_intermediate_dim",
        type=int,
        default=3072, # Changed default to match 300m model specs
        help="Intermediate dimension for the pooling head's first dense layer.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="The final output dimension of the embedding projection.",
    )

    args = parser.parse_args()

    convert_to_embedding_preset(
        source_preset=args.source_preset,
        output_preset=args.output_preset,
        pooling_intermediate_dim=args.pooling_intermediate_dim,
        embedding_dim=args.embedding_dim,
    )