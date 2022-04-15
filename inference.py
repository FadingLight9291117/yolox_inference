from cmath import inf
from distutils.ccompiler import gen_preprocess_options
import os

import cv2
from imageio import save
from pip import main
import tqdm
import torch

from yolox.models import create_yolox_model
from preprocess import preprocess
from postprocess import postprocess
from visual import visual

from config import config as cfg

# todo


def get_model(name='yolox-l', device='gpu'):
    # load model
    model = create_yolox_model(name=name)
    model.eval()
    if device == 'gpu':
        model = model.cuda()
    return model


def inference(model, img_path, save_dir):
    # read image
    img_raw = cv2.imread(img_path)
    # img_info
    img_h, img_w, _ = img_raw.shape
    ratio = min(cfg.img_size[0] / img_h, cfg.img_size[1] / img_w)
    img_info = dict(
        id=0,
        file_name=None,
        height=img_h,
        weight=img_w,
        raw_img=img_raw,
        ratio=ratio,
    )

    # preprocess
    img = preprocess(img_raw,  cfg.img_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    if cfg.device == 'gpu':
        img = img.cuda()

    # inference
    with torch.no_grad():
        outputs = model(img)

    # postprocess
    outputs = postprocess(
        outputs, cfg.num_classes, cfg.confthre,
        cfg.nmsthre, class_agnostic=True
    )

    # visual
    result_image = visual(outputs[0], img_info, cfg.confthre)
    cv2.imwrite(os.path.join(
        save_dir, os.path.basename(img_path)), result_image)


if __name__ == '__main__':
    img_path = './images'
    save_dir = './results'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('loading model...')
    model = get_model(device=cfg.device)

    print('inferencing...')
    if os.path.isfile(img_path):
        inference(model, img_path, save_dir)

    elif os.path.isdir(img_path):
        img_files = os.listdir(img_path)
        img_files = [os.path.join(img_path, path) for path in img_files]
        img_files = tqdm.tqdm(img_files, desc='inference')
        for img_file in img_files:
            inference(model, img_file, save_dir)
    print(f'results saved in {save_dir}')
    print('end.')
