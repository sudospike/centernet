import sys
sys.path.remove('/home/lingfan/anaconda2/envs/centernet/lib/python3.6/site-packages')
test_url = "/home/lingfan/work/mywork/data/VOCdevkit/VOC2007/JPEGImages"
import os
import cv2
import random
import numpy as np
import torch
import torch.utils.data as data

class VOCDataset(data.Dataset):

    def __init__(self,root,list_file,mode,transform):
        self.root = root
        self.images = []
        self.boxes = []
        self.labels = []
        self.mean = (123,117,104)
        self.train = mode
        self.transform = transform
        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            box = []
            label = []
            elements = line.strip().split(' ')
            image_name = elements[0]
            self.images.append(image_name)
            num_box = int(elements[1])

            for i in range(num_box):
                x1 = float(elements[5*i+2])
                y1 = float(elements[5*i+3])
                x2 = float(elements[5*i+4])
                y2 = float(elements[5*i+5])
                c = float(elements[5*i+6])
                box.append([x1,y1,x2,y2])
                label.append(c)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

        self.numbers = len(self.boxes)

    def __getitem__(self, index):

        fname = self.images[index]
        print(os.path.join(self.root,fname))
        image = cv2.imread(os.path.join(self.root,fname))
        boxes = self.boxes[index].clone()
        labels = self.labels[index].clone()

        if self.train:
            image,boxes = self.Random_flip(image,boxes)
            #image,boxes = self.RandomScale(image,boxes)
            #image = self.RandomBlur(image)
            #image = self.RandomBrightness(image)
            #image = self.RandomHue(image)
            #image = self.RandomSaturation(image)

            #image,boxes,labels = self.RandomShift(image,boxes,labels)

        return image,boxes,labels

    def __len__(self):

        return self.numbers
    #cv默认是BGR，pytorch是RGB
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v,0,255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(v,0,255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h,0,255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def RandomShift(self, bgr, boxes, labels):

        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)

            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def RandomScale(self, bgr, boxes):
        # 固定住高度，以0.6-1.4伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.6, 1.4)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def RandomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def SubMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def Random_flip(self, im, boxes):
        '''
        随机翻转
        '''
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def Random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im

myVOC = VOCDataset(test_url,'test.txt',1,None)
pic,boxes,labels = myVOC[9]

print(boxes)
print(labels)
cv2.imshow('ss',pic)
cv2.waitKey(0)
