import ml_collections

def get_s16_config():
  """Returns the ViT-S/16 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'ViT-S_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.split = 'non-overlap'
  config.hidden_size = 384
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 1536
  config.transformer.num_heads = 6
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = 'token'
  config.representation_size = None
  return config
