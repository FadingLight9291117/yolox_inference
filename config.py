from easydict import EasyDict as edict


config = edict(
    img_size=(640, 640),
    device='gpu',
    num_classes=80,
    confthre=0.5,
    nmsthre=0.65,
)
