_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='KnowledgeDistillationSingleStageDetector',
    pretrained='torchvision://resnet18',
    output_feature=True,
    teacher_config='configs/gfocal/gfocal_r101_fpn_ms2x.py',
    teacher_ckpt='/home/syy/GFocalV2/work_dirs/gfocal_r101_fpn_ms2x.pth',  # noqa
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='PCDHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_ld_vlr=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        loss_kd=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=2.5, T=10),
        loss_q=dict(type='MSELoss', loss_weight=10.0),
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
)
optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[8,11])
runner = dict(type='EpochBasedRunner',max_epochs=12)
