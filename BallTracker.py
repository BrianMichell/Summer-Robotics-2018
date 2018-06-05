import cv2 as cv
import numpy as np
import ROI

cap = cv.VideoCapture(0)

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

# Old HSV bounding
# lbLL = np.array([10,75,90]) #Lower bound
# ubLL = np.array([45,255,255]) #Upper bound
# lbHL = np.array([15,105,90])
# ubHL = np.array([35,255,200])

# No HSV bounding
# lbHL = np.array([0,0,0])
# ubHL = np.array([255,255,255])

kernel = np.ones((2, 2), np.uint8)

tracker = []

while True:
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    if(lowLight):
        mask = cv.inRange(hsv, lbLL, ubLL)
        res = cv.bitwise_and(frame, frame, mask=mask)
        # res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        # res = cv.inRange(res, 150, 255)
    else:
        mask = cv.inRange(hsv, lbHL, ubHL)
        res = cv.bitwise_and(frame, frame, mask=mask)
        res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    
    opening = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    # erode = cv.erode(res, kernel, iterations = 2)
    
    # median = cv.medianBlur(closing, 9)
    median = cv.medianBlur(closing, 13)
    # median = cv.medianBlur(closing, 21)
    
    edges = cv.Canny(median, 250, 255)
    
    _, contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    region = []
    for cnt in contours:
        
        moment = cv.moments(cnt)
        
        x, y, w, h = cv.boundingRect(cnt)
        region.append(ROI.Region(x, y, h, w))
        
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
    
    for elem in region:
        corner1, corner2 = elem.getPoints()
        cv.rectangle(frame, corner1, corner2, elem.color, 2)
        cv.putText(frame, 'Width: {}, Height: {}'.format(elem.w, elem.h), corner1, FONT, 1, elem.color, 1, AA)
        distance = PIX_PER_INCH / ((elem.w + elem.h) / 2)
        cv.putText(frame, 'Distance: {:.3} inches'.format(distance), corner2, FONT, 2, (0, 255, 0), 2, AA)
        cv.circle(frame, (elem.cx, elem.cy), 3, (255, 0, 0), -1)
        cv.putText(frame, '({},{})'.format(elem.cx, elem.cy), (elem.cx, elem.cy), FONT, 1, elem.color, 1, AA)
        tracker.append(elem.cx)
        tracker.append(elem.cy)
        
    '''
    length = len(tracker)
    for i in range(0, length, 2):
        cv.circle(frame, (tracker[i], tracker[i + 1]), 1, (0, 255, 0), -1)
       
    '''    
    cv.imshow('frame', frame)
    cv.imshow('res', res)
    cv.imshow('mask', mask)
    cv.imshow('closing', closing)
    cv.imshow('median', median)
    cv.imshow('edges', edges)
    cv.imshow('HSV masked', cv.bitwise_and(hsv, hsv, mask=mask))
    
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
