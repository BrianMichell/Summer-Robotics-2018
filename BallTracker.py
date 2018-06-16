import cv2 as cv
import numpy as np
import ROI
import time

def normPoints(x, y, w, h, xHeight, yHeight, buffer = 5):
    xMin = max(0, x - buffer)
    xMax = min(xHeight, x + w + buffer)
    yMin = max(0, y - buffer)
    yMax = min(yHeight, y + h + buffer)
    
    return xMin, xMax, yMin, yMax

cap = cv.VideoCapture(0)
ball = cv.imread('ball4.png')  # This image is 50 x 50 pixels
ball = cv.cvtColor(ball, cv.COLOR_BGR2GRAY)

lowLight = True

PIX_PER_INCH = 160 * 12

FONT = cv.FONT_HERSHEY_PLAIN
AA = cv.LINE_AA

# Low (artificial) light bounding
lbLL = np.array([22, 25, 200])  # Lower bound
ubLL = np.array([55, 160, 255])  # Upper bound

# High (natural) light bounding
lbHL = np.array([25, 125, 100])
ubHL = np.array([35, 255, 200])

# No HSV bounding
# lbHL = np.array([0,0,0])
# ubHL = np.array([255,255,255])

kernel = np.ones((2, 2), np.uint8)

tracker = []

lastTime = time.time()

while True:
    _, frame = cap.read()
    gs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame = cv.bilateralFilter(frame, 20, 25, 75)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    if(lowLight):
        mask = cv.inRange(hsv, lbLL, ubLL)
        res = cv.bitwise_and(frame, frame, mask=mask)
        res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        # res = cv.inRange(res, 150, 255)
    else:
        mask = cv.inRange(hsv, lbHL, ubHL)
        res = cv.bitwise_and(frame, frame, mask=mask)
        res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    
    opening = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    # erode = cv.erode(res, kernel, iterations = 2)
    
    # bilat = cv.bilateralFilter(res, 5, 10, 20)
    
    # median = cv.medianBlur(closing, 9)
    median = cv.medianBlur(closing, 13)
    # median = cv.medianBlur(erode, 13)
    # median = cv.medianBlur(closing, 21)
    
    edges = cv.Canny(median, 250, 255)
    
    _, contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    region = []
    for cnt in contours:
        
        moment = cv.moments(cnt)
        
        x, y, w, h = cv.boundingRect(cnt)
        
        xMin, xMax, yMin, yMax = normPoints(x, y , w, h, 480, 640)
        
        item = gs[yMin:yMax, xMin:xMax]
        region.append(ROI.Region(x, y, h, w, item))
        
        size = len(region)
        prev = 0
        cur = size
        if(size > 1):
            while prev < cur:
                cur -= 1
                if(region[prev].contains(region[cur])):
                    region.pop(cur)
                    # region[cur].removeColor()
                else:
                    try:  # Find center of mass
                        cx = int(moment['m10'] / moment['m00'])
                        cy = int(moment['m01'] / moment['m00'])
                        region[cur].addCOM(cx, cy)
                    except (ZeroDivisionError):
                        pass
    
    flag = False
    
    itemsOfInterest = np.zeros(gs.shape, np.uint8)
    individuals = []
    # itemsOfInterest = np.zeros((480,640))
    # itemsOfInterest[0:480, 0:640] = gs[0:480, 0:640]
    for elem in region:

        corner1, corner2 = elem.getPoints()
        cv.rectangle(frame, corner1, corner2, elem.color, 2)
        cv.putText(frame, 'Width: {}, Height: {}'.format(elem.w, elem.h), corner1, FONT, 1, elem.color, 1, AA)
        
        if(not flag):
            individuals.append(elem.getResized())
            xMin, xMax, yMin, yMax = normPoints(elem.x, elem.y, elem.w, elem.h, 480,640)
            itemsOfInterest[yMin:yMax, xMin:xMax] = elem.image
            # cv.rectangle(itemsOfInterest, corner1, corner2, elem.color, 2)
            
            # flag = True
            
        distance = PIX_PER_INCH / ((elem.w + elem.h) / 2)
        cv.putText(frame, 'Distance: {:.3} inches'.format(distance), corner2, FONT, 2, (0, 255, 0), 2, AA)
        cv.circle(frame, (elem.cx, elem.cy), 3, (255, 0, 0), -1)
        cv.putText(frame, '({},{})'.format(elem.cx, elem.cy), (elem.cx, elem.cy), FONT, 1, elem.color, 1, AA)
        tracker.append(elem.cx)
        tracker.append(elem.cy)
    
    for item in individuals:
        matches = cv.matchTemplate(item, ball, cv.TM_CCOEFF_NORMED)
        balls = np.where(matches >= 0.7)
        
        for pt in zip(*balls[::-1]):
            cv.circle(frame, (pt[0], pt[1]), 20, (255, 255, 255), -1)
    '''    
    length = len(tracker)
    for i in range(0, length, 2):
        cv.circle(frame, (tracker[i], tracker[i + 1]), 1, (0, 255, 0), -1)
    '''   
    
    # edges[0:100,0:100] = (255,255,255)
    
    cv.imshow('Items of interest', itemsOfInterest)
        
    cv.imshow('frame', frame)
    cv.imshow('res', res)
    cv.imshow('mask', mask)
    # cv.imshow('erode', erode)
    cv.imshow('closing', closing)
    cv.imshow('median', median)
    cv.imshow('edges', edges)
    cv.imshow('HSV masked', cv.bitwise_and(hsv, hsv, mask=mask))
    
    # print('Frametime is {:.4} milliseconds'.format((time.time() - lastTime) * 1000))
    lastTime = time.time()
    
    k = cv.waitKey(5) & 0xFF
    if(k == 27):
        break
    
    if(k == 108):
        if(lowLight):
            tracker = []
            lowLight = False
        else:
            tracker = []
            lowLight = True
    
cv.destroyAllWindows()
cap.release()
    
