# import the opencv library
import cv2 as cv
import numpy as np

def edgeDetection(image):
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grayscale, (3, 3), 1)
    cannyEdge = cv.Canny(blur, 20, 50)
    cannyEdge = cv.dilate(cannyEdge, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)
    return cannyEdge

def regionOfInterest(image):
    height = image.shape[0]
    polygons = np.array([ [(10, height), (630, height), (630, height - 100), (380, 100), (260, 100), (10, height - 100)] ])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygons, 255)
    cropped = cv.bitwise_and(image, mask)
    return cropped

def lineDetection(image):
    lines = cv.HoughLinesP(image, 3, 2*(np.pi/180), 200, np.array([]), minLineLength=20, maxLineGap = 50)
    lineImage = np.zeros_like(image)
    lanes = None
    if lines is not None:
        lanes = avgLines(image, lines)
        for lane in lanes:
            x1, y1, x2, y2 = lane
            cv.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return lineImage, lanes

def avgLines(image, lines):
    leftCandidates = []
    rightCandidates = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            leftCandidates.append((slope, intercept))
        else:
            rightCandidates.append((slope, intercept))

    leftLine = None
    rightLine = None
    if len(leftCandidates) > 0:
        leftAverage = np.average(leftCandidates, axis=0)
        left_y1 = image.shape[0]
        left_y2 = int(left_y1 * (2/3))
        left_x1 = int((left_y1 - leftAverage[1]) / leftAverage[0])
        left_x2 = int((left_y2 - leftAverage[1]) / leftAverage[0])
        leftLine = np.array([left_x1, left_y1, left_x2, left_y2])

    if len(rightCandidates) > 0:
        rightAverage = np.average(rightCandidates, axis=0)
        right_y1 = image.shape[0]
        right_y2 = int(right_y1 * (2/3))
        right_x1 = int((right_y1 - rightAverage[1]) / rightAverage[0])
        right_x2 = int((right_y2 - rightAverage[1]) / rightAverage[0])
        rightLine = np.array([right_x1, right_y1, right_x2, right_y2])

    if leftLine is not None and rightLine is not None:
        return np.array([leftLine, rightLine])
    elif leftLine is not None:
        # did not detect 2 lane lines
        return np.array([leftLine])
    elif rightLine is not None:
        # did not detect 2 lane lines
        return np.array([rightLine])
    else:
        return []

def removeHorizontal(image):
    # initalize masks so we can remove the horizontal lines from the image
    horizontal = np.copy(image)
    horizontal = cv.blur(horizontal,(2,2))

    # make a mask for the horizontal lines
    cols = horizontal.shape[1]
    horizontal_size = cols // 10
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    horizontal = cv.blur(horizontal,(6,6))
    horizontal = (255-horizontal)
    horizontal = cv.threshold(horizontal,250,255,cv.THRESH_BINARY)[1]

    return cv.bitwise_and(image, horizontal), horizontal

# aligns the vehicle to the road
def align(lanes):
    if lanes is not None:
        if len(lanes) == 2:
            slopeLeft = slope(lanes[0])
            slopeRight = slope(lanes[1])
            '''if (slopeLeft + slopeRight) > 0.5:
                print('Turn Right')
            elif (slopeLeft + slopeRight) < -0.5:
                print('Turn Left')
            else:
                print('Go Straight')'''

            print(slopeLeft + slopeRight)

# stop the vehicle at a crosswalk
def stop(crosswalk):
    height = crosswalk.shape[0]
    width = crosswalk.shape[1]
    for i in range(height - 130, height):
        for j in range(int(width//2) - 3, int(width//2) + 3):
            if crosswalk[i][j] == 0:
                return True
    return False

def slope(line):
    x1, y1, x2, y2 = line
    if (x2 - x1) == 0:
        return 100
    return (y2 - y1) / (x2 - x1)

# define a video capture object
vid = cv.VideoCapture(0)

while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cannyEdge = edgeDetection(frame)
    mask = regionOfInterest(cannyEdge)
    mask, crosswalk = removeHorizontal(mask)
    try:
        lineImage, lanes = lineDetection(mask)
        if (stop(crosswalk)):
            print('STOP')
        else:
            align(lanes)
        cv.imshow('line', lineImage)
        cv.imshow('edge', cannyEdge)
        cv.imshow('crosswalk', crosswalk)
    except:
        continue

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
