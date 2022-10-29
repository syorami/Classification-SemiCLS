train = dict(
    eval_step=1024,
    total_steps=1024 * 512,
    trainer=dict(
        type='HyperMatch',
        threshold=0.95,
        T=1.,
        topk=2,
        temperature=0.07,
        lambda_u=1.,
        lambda_contrast=1,
        gmm_thr=0.6,
        loss_x=dict(type="cross_entropy", reduction="mean"),
        loss_u=dict(type="cross_entropy", reduction="none"),
))
num_classes = 10
model = dict(
    type='resnet18',
    low_dim=64,
    num_class=10,
    proj=True,
    width=1,
    in_channel=3)
stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)
data = dict(
    type='STL10SSL',
    folds=1,
    num_workers=4,
    num_classes=10,
    batch_size=64,
    mu=7,
    root='./data/stl10',
    lpipelines=[[{
        'type': 'RandomHorizontalFlip',
        'p': 0.5
    }, {
        'type': 'RandomCrop',
        'size': 96,
        'padding': 12,
        'padding_mode': 'reflect'
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2471, 0.2435, 0.2616)
    }]],
    upipelinse=[[{
        'type': 'RandomHorizontalFlip',
        'p': 0.5
    }, {
        'type': 'RandomCrop',
        'size': 96,
        'padding': 12,
        'padding_mode': 'reflect'
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2471, 0.2435, 0.2616)
    }],
                [{
                    'type': 'RandomHorizontalFlip',
                    'p': 0.5
                }, {
                    'type': 'RandomCrop',
                    'size': 96,
                    'padding': 12,
                    'padding_mode': 'reflect'
                }, {
                    'type': 'RandAugmentMC',
                    'n': 2,
                    'm': 10
                }, {
                    'type': 'ToTensor'
                }, {
                    'type': 'Normalize',
                    'mean': (0.4914, 0.4822, 0.4465),
                    'std': (0.2471, 0.2435, 0.2616)
                }],
                [{
                    'type': 'RandomResizedCrop',
                    'size': 96,
                    'scale': (0.2, 1.0)
                }, {
                    'type': 'RandomHorizontalFlip',
                    'p': 0.5
                }, {
                    'type':
                    'RandomApply',
                    'transforms': [{
                        'type': 'ColorJitter',
                        'brightness': 0.4,
                        'contrast': 0.4,
                        'saturation': 0.4,
                        'hue': 0.1
                    }],
                    'p':
                    0.8
                }, {
                    'type': 'RandomGrayscale',
                    'p': 0.2
                }, {
                    'type': 'ToTensor'
                }, {
                    'type': 'Normalize',
                    'mean': (0.4914, 0.4822, 0.4465),
                    'std': (0.2471, 0.2435, 0.2616)
                }]],
    vpipeline=[
        dict(type='ToTensor'),
        dict(
            type='Normalize',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2471, 0.2435, 0.2616))
    ],
    eval_step=1024)
scheduler = dict(
    type='cosine_schedule_with_warmup',
    num_warmup_steps=0,
    num_training_steps=524288)
ema = dict(use=True, pseudo_with_ema=False, decay=0.999)
amp = dict(use=False, opt_level='O1')
log = dict(interval=50)
ckpt = dict(interval=1)
optimizer = dict(
    type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True)
resume = 'True'
