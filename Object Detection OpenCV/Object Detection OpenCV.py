import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # read input from webcam
cap.set(3, 600)
cap.set(4, 600)
cap.set(10, 100)
thres = 0.50
class_file = 'coco.names'
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'

class_names = []
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# print(classnames)
eye = cv2.dnn_DetectionModel(weights_path, configPath)
eye.setInputSize(328, 328)
eye.setInputScale(1.0 / 127.5)
eye.setInputMean((127.5, 127.5, 127.5))
eye.setInputSwapRB(True)

while True:
    success, frame = cap.read()
    classIds, confs, bbox = eye.detect(frame, thres)
    bbox = list(bbox)
    confs = np.array(confs).reshape(1, -1).tolist()[0]
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold=0.25)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(frame, (x, y), (x + w, h + y), color=(0, 0, 255), thickness=2)
        cv2.putText(frame, class_names[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        if len(classIds) != 0:
            for confidence in confs:
                cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 190, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 0, 0), 1)

    # if len(classIds) != 0:
    #     for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    #         cv2.rectangle(frame, box, color=(0, 0, 255), thickness=2)
    #         cv2.putText(frame, class_names[classId - 1].upper(), (box[0] + 10, box[1] + 30),
    #                     cv2.FONT_HERSHEY_COMPLEX_SMALL,
    #                     1, (0, 255, 0), 2)
    #         cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 150, box[1] + 30),
    #                     cv2.FONT_HERSHEY_COMPLEX_SMALL,
    #                     1, (0, 0, 0), 2)

    cv2.imshow("Output", frame)
    cv2.waitKey(1)
