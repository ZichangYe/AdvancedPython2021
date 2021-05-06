import cv2
import numpy as np

def array_to_mp4(filename, array, framerate):
    '''
    :param filename: path to writen file
    :param array: 4d numpy array
    :param framerate: fps
    '''
    frame, height, width, channel = array.shape
    frameSize = (width, height)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), framerate, frameSize)

    for i in range(frame):
        out.write(array[i])

    out.release()


def mp4_to_array(filename):
    '''
    :param filename: path of file to read
    :return: 4d numpy array each pixel is bgr
    '''
    cap = cv2.VideoCapture(filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f'hello: {frameCount}, {frameWidth}, {frameHeight}, {cap.isOpened()}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    
    while (fc < frameCount and ret and cap.isOpened()):
        try:
            ret, buf[fc] = cap.read()
            fc += 1
        except:
            pass

    cap.release()

    return buf, fps