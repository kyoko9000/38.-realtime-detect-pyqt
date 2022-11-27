import cv2
import torch

# define a video capture object
vid = cv2.VideoCapture("movies/2.mp4")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
# model.classes = [2]  # 0: human, 2: car, 14: bird
classes = model.names  # name of objects
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def score_frame(frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    param frame: input frame in numpy/list/tuple format.
    return: Labels and Coordinates of objects detected by model in the frame.
    """
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, cord


def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    return classes[int(x)]


def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    param results: contains labels and coordinates predicted by model on the given frame.
    param frame: Frame which has been scored.
    return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    print("labels", labels)
    print("cord", cord[:, :-1])
    clas = 14
    if len(labels) != 0:
        print("list is not empty")
        for label in labels:
            if label == clas:
                print("send objects")
            else:
                print("wrong objects")
    else:
        print("list is empty")
        print("no objects")

    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # print("predict", round(cord[i][4], 2))
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]) + " " + str(round(row[4], 2)), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    results = score_frame(frame)
    # print(results)
    frame = plot_boxes(results, frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
