_base_ = ['./segformer_mit-b2_8xb2-160k_ade20k-512x512.py']

# Override total training iterations to 30000 (matching paper's adversarial training setup)
train_cfg = dict(max_iters=30000, val_interval=5000)

# Adversarial training configuration
model = dict(
    adv_train_cfg=dict(
        name='cospgd',
        norm='linf',
        epsilon=4.0,
        alpha=0.01,
        iterations=3,
        metric='cosine',
    ),
    normalize_mean_std=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    enable_normalization=False,
)
