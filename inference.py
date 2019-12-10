from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
# from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np
from matplotlib import pyplot as plt
from data import BaseTransform, TQQDetection, TQQAnnotationTransform, TQQ_CLASSES as labelmap
from ssd import build_ssd
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
# parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
parser.add_argument('--weights', default='weights/TQQ.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
parser.add_argument('--num', default=0, type=int)
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

TQQ_ROOT = ''

def demo(net, transform, num):
    testset = TQQDetection(TQQ_ROOT, [('TQQ', 'trainval')], None, TQQAnnotationTransform())
    img_id = num
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.imread(image_path)
    height, width = image.shape[:2]
    x = torch.from_numpy(transform(image)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)  # forward pass
    plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)
    currentAxis = plt.gca()
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labelmap[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            # pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            # cv2.rectangle(image,
            #               (int(pt[0]), int(pt[1])),
            #               (int(pt[2]), int(pt[3])),
            #               COLORS[i % 3], 2)
            # image = CvPutJaText.puttext(image, labelmap[i - 1], (int(pt[0]), int(pt[1])), font_path, 5, (255, 0, 0))
            # cv2.putText(image, labelmap[i - 1], (int(pt[0]), int(pt[1])),
            #             FONT, 2, (255, 0, 0), 2, cv2.LINE_AA)
            j += 1
    # cv2.imwrite('./results/' + str(num) + '.png', image)
    plt.show()
    plt.close()

if __name__ == '__main__':
    num_classes = len(labelmap) + 1

    num = args.num

    net = build_ssd('test', 300, num_classes)    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    demo(net.eval(), transform, num)
