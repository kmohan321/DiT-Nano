#change accordingly 
config = {
    'image_size': 64,
    'patch_size': 4,
    'input_channel': 3,
    'out_channel': 3,
    'num_classes': 1000,
    'hidden_dim': 384,
    'freq_dim': 256,
    'heads': 6,
    'head_dim': 64,
    'mlp_multiplier':4,
    'num_blocks': 12,
    'epochs': 50,
    'cfm_weight': 0.05,
    'lr': 1e-4,
    "batch_size": 384,
    'use_amp' : True,
    'resume' : False   #False -> start training from scratch
}