_base_ = ['./upernet_internimage_s_512_160k_ade20k.py']

# Override total training iterations to 30,000 (matching paper's adversarial training setup)
runner = dict(type='IterBasedRunner', max_iters=30000)
evaluation = dict(intervall=5000, metric='mIoU', save_best='mIoU')

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
