import argparse
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_scene', type=str, required=True)
    parser.add_argument('--text_prompt', type=str, required=True)
    return parser.parse_args()

def load_image(image_path):
    return cv2.imread(image_path)

def detect_object(image, text_prompt):
    # Load pre-trained object detection model
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    # Perform object detection
    outputs = net.forward(image)
    # Extract bounding boxes and class IDs
    boxes = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Filter by class ID and confidence
                x, y, w, h = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                boxes.append([x, y, w, h])
                class_ids.append(class_id)
    # Create binary mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    return mask

def apply_red_mask(image, mask):
    red_mask = np.zeros(image.shape, dtype=np.uint8)
    red_mask[:, :, 2] = 255  # Set red channel to 255
    return cv2.bitwise_and(image, red_mask, mask=mask)

def main():
    args = parse_args()
    image = load_image(args.input_scene)
    mask = detect_object(image, args.text_prompt)
    output_image = apply_red_mask(image, mask)
    cv2.imwrite('output.png', output_image)

if __name__ == '__main__':
    main()
