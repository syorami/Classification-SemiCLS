train = dict(
    eval_step=1024,
    total_steps=1024*512,
    trainer=dict(
        type="FixMatch_Relax_Reweight",
        threshold=0.95,
        T=1.,
        topk=2,
        temperature=0.07,
        lambda_u=1.,
        lambda_contrast=1,
        loss_x=dict(type="cross_entropy", reduction="mean"),
        loss_u=dict(type="cross_entropy", reduction="none"),
    ))

num_classes = 100
seed = 1

model = dict(
    type="wideresnet",
    depth=28,
    widen_factor=8,
    dropout=0,
    low_dim=64,
    num_classes=num_classes,
    proj=True
)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

data = dict(
    # CIFAR10SSL, CIFAR100SSL
    type="CIFAR100SSL",
    num_workers=4,
    num_labeled=400,
    num_classes=num_classes,
    batch_size=16,
    expand_labels=False,
    mu=7,

    root="./data/CIFAR",
    lpipelines=[[
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop",
             size=32,
             padding=int(32 * 0.125),
             padding_mode='reflect'),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=cifar100_mean, std=cifar100_std)
    ]],
    upipelinse=[[
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop",
             size=32,
             padding=int(32 * 0.125),
             padding_mode='reflect'),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=cifar100_mean, std=cifar100_std)],[
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop",
             size=32,
             padding=int(32 * 0.125),
             padding_mode='reflect'),
        dict(type="RandAugmentMC", n=2, m=10),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=cifar100_mean, std=cifar100_std)],[
        dict(type="RandomResizedCrop",size=32),
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomApply",
            transforms=[
                dict(type="ColorJitter",
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1),
            ],
            p=0.8),
        dict(type="RandomGrayscale",p=0.2),
        dict(type="ToTensor")]],
    vpipeline=[
        dict(type="ToTensor"),
        dict(type="Normalize", mean=cifar100_mean, std=cifar100_std)
    ])

scheduler = dict(
    type='cosine_schedule_with_warmup',
    num_warmup_steps=0,
    num_training_steps=train['total_steps']
)

ema = dict(use=True, pseudo_with_ema=False, decay=0.999)
amp = dict(use=True, opt_level="O2")

log = dict(interval=50)
ckpt = dict(interval=1)
evaluation = dict(eval_both=True)

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=True)
