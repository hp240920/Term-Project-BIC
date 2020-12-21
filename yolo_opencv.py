#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################
import cv as cv
import cv2
import time
import argparse
import numpy as np
from lif_model import count_spikes
from lif_model import lif
import matplotlib.image as mpimg
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', default='test5.jpg',
                help='path to input image')
ap.add_argument('-c', '--config', default='yolov3.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default='yolov3.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='yolov3.txt',
                help='path to text file containing class names')
args = ap.parse_args()
print(args)
print(type(args))
weights = np.load('weights_0.7.npy')

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS
    color = (0,0,0)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    crop_img = img[y:y_plus_h, x:x_plus_w]
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return crop_img


def detect_color1(img1):
    R = img1[:, :, 0]
    G = img1[:, :, 1]

    # R-> 0 , G-> 1, Y-> 2
    pixel_arr = np.append(np.reshape(R, [1, 100]), np.reshape(G, [1, 100]))
    # pixel_arr = np.append(pixel_arr, np.reshape(B, [1, 100]))
    output = weights.dot(np.transpose(pixel_arr))
    current_output = np.argmax(output)
    if current_output == 0:
        print("Red")
    elif current_output == 1:
        print("Green")
    else:
        print("Yellow")


def detect_color(img1):
    R1 = (img1[0:17, 17:34, 0])
    R2 = (img1[17:34, 17:34, 0])
    R3 = (img1[34:50, 17:34, 0])
    R4 = (img1[17:34, 0:17, 0])
    R5 = (img1[17:34, 34: 50, 0])

    G1 = (img1[0:17, 17:34, 1])
    G2 = (img1[17:34, 17:34, 1])
    G3 = (img1[34:50, 17:34, 1])
    G4 = (img1[17:34, 0:17, 1])
    G5 = (img1[17:34, 34: 50, 1])

    B1 = (img1[0:17, 17:34, 2])
    B2 = (img1[17:34, 17:34, 2])
    B3 = (img1[34:50, 17:34, 2])
    B4 = (img1[17:34, 0:17, 0])
    B5 = (img1[17:34, 34: 50, 0])

    # Vertical + Horizontal
    var = {count_spikes(lif(sum(sum(R1)))): "R1", count_spikes(lif(sum(sum(R3)))): "R3",
           count_spikes(lif(sum(sum(G1)))): "G1", count_spikes(lif(sum(sum(G3)))): "G3",
           count_spikes(lif(sum(sum(R4)))): "R4", count_spikes(lif(sum(sum(R5)))): "R5",
           count_spikes(lif(sum(sum(G4)))): "G4", count_spikes(lif(sum(sum(G5)))): "G5"
           }
    maxtb = max(var)

    color = -1
    if maxtb < count_spikes(lif(sum(sum(R2)))) + count_spikes(lif(sum(sum(G2)))) / 2:
        color = 1
        print("Yellow")
    else:
        temp_c = var.get(maxtb)
        if temp_c in ["R1", "R3", "R4", "R5"]:
            color = 0
            print("Red")
        else:
            color = 2
            print("Green")


# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)

#while cv2.waitKey(1):
#    hasFrame, frame = cap.read()
#    if not hasFrame:
#        print("Done processing !!!")
#        break
    # cv2.waitKey(50)
timer = time.time()
image = cv2.imread(args.image)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392
cv2.imshow('Capturing video',image)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(get_output_layers(net))
# print(time.time() - timer)

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
print(indices)
# print(type(class_ids[0]))
# out_images = np.array(class_ids)[indices.astype(int)]
for i in indices:
    try:
        i = i[0]
        if class_ids[i] != 9:
            continue
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        crop_img = draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w),
                                   round(y + h))

        # cv2.imshow("object detection", crop_img)
        # cv2.waitKey()
        cv2.imwrite("processing_img.png", crop_img)
        imaget = Image.open("processing_img.png")
        print(imaget.size)
        img_resized = imaget.resize((10, 10))
        img_resized.save("processing_img.png")
        img1 = mpimg.imread("processing_img.png")
        # detect_color(img1)
        detect_color1(img1)
    except:
        print("Error")

# cv2.waitKey()


#cv2.imshow("object detection", image)
#cv2.waitKey()

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
