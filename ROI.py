import cv2 as cv
import numpy as np

class Region:
    
    x = 0
    y = 0
    w = 0
    h = 0
    cx = 0
    cy = 0
    
    image = np.zeros((1,1), np.uint8)
    
    color=(0,0,255)
    
    def __init__(self,x,y,h,w, image):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = 0
        self.cy = 0
        
        self.image = image

    def update(self, x, y, h, w):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
    def contains(self, region):
        if(region.x > self.x and region.y > self.y):
            if(region.x + region.w < self.x + self.w and region.y + region.h < self.y + self.h):
                return True
        return False
        
        '''
        if((region.x > self.x and region.w < self.w) and region.x + region.w < self.x + self.w):
            if((region.y > self.y and region.h < self.h) and region.y + region.h <self.y + self.h):
                return True
        return False
        '''
    
    def addCOM(self, x, y):
        self.cx = x
        self.cy = y
        
    def removeColor(self):
        self.color = (0,255,0)
    
    def getPoints(self):
        return (self.x,self.y) , (self.x+self.w, self.y+self.h)
    
    """
    This function will sometimes catch an Assertion failed error.
    If this happens the function will return an 'image' of all black pixels
    """
    def getResized(self):
        try:
            return cv.resize(self.image, (50,50))
        except:
            return np.zeros((50,50), np.uint8)
