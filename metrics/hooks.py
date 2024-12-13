
# the same forward process as original clip attention module, but we fetch the Q, K, V
def clip_encoder_layer_forward_hook(module, input, output):
    hidden_states, _, _ = input

    bsz, tgt_len, embed_dim = hidden_states.size()

    query_states = module.self_attn.q_proj(hidden_states)
    key_states = module.self_attn.k_proj(hidden_states)
    value_states = module.self_attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, -1, module.self_attn.num_heads, module.self_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, -1, module.self_attn.num_heads, module.self_attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, -1, module.self_attn.num_heads, module.self_attn.head_dim).transpose(1, 2)

    module.stores = []
    module.stores = [query_states, key_states, value_states, (bsz, tgt_len, embed_dim)]

# the same forward process as original clip attention module, but we fetch the Q, K, V
def clip_encoder_layer_feature_forward_hook(module, input, output):
    module.stores = [output[0]]

def dinov2_self_attention_forward_hook(module, input, output):
    hidden_states, _, _ = input
    mixed_query_layer = module.query(hidden_states)

    key_layer = module.transpose_for_scores(module.key(hidden_states))
    value_layer = module.transpose_for_scores(module.value(hidden_states))
    query_layer = module.transpose_for_scores(mixed_query_layer)

    module.stores = []
    module.stores = [query_layer, key_layer, value_layer]
    
def dinov2_self_attention_forward_feature_hook(module, input, output):
    module.stores = [output[0]]

def diffusion_self_attention_forward_hook(module, input, output):
    module.stores = [output]