import cv2 as cv
import random

# 文件需要加载的文件
cfg = "./model/yolov4-bm-test.cfg"
weights = "./model/yolov4-bm_final.weights"
imgName = "./picture/img4.jpg"
className = "./model/voc-bm.names"

# 网络设置
net = cv.dnn_DetectionModel(cfg, weights)
net.setInputSize(608, 608)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)
with open(className, 'rt') as f:
    names = f.read().rstrip('\n').split('\n')

# 加载图像及类别
img = cv.imread(imgName)
cv.imshow('read img', img)

# 模型检测
classes, confidences, boxes = net.detect(img, confThreshold=0.1, nmsThreshold=0.4)

# 将检测结果显示到图像上
for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
    label = '%s: %.2f' % (names[classId], confidence)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    left, top, width, height = box
    top = max(top, labelSize[1])
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
    cv.rectangle(img, box, color=(b, g, r), thickness=2)
    cv.rectangle(img, (left-1, top - labelSize[1]), (left + labelSize[0], top), (b, g, r), cv.FILLED)
    cv.putText(img, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255-b, 255-g, 255-r))

cv.imshow('detect out', img)
cv.waitKey(0)
