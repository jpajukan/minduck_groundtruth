import time
import cv2
import numpy as np
import random


def segmentation(arg,image_gray):
        if arg == 1: #threshold segmentaatio
                (thresh, image_bw) = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                return image_bw
        elif arg == 2: #watershed kayttaen aina otsua              
                ret,thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                fg = cv2.erode(thresh,None,iterations = 2)
                bgt = cv2.dilate(thresh,None,iterations = 3)
                ret,bg = cv2.threshold(bgt,1,128,1)
                marker = cv2.add(fg,bg)
                marker32 = np.int32(marker)
                cv2.watershed(image,marker32)
                m = cv2.convertScaleAbs(marker32)
                ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                return thresh
        elif arg == 3: #adaptive threshold
                image_bw = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 5)
                return image_bw
        elif arg == 4: #grabCut
                print "tyhja"
        elif arg == 5: #watershed kayttaen edellista kuvaa       
                ret,thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                fg = cv2.erode(thresh,None,iterations = 2)
                bgt = cv2.dilate(thresh,None,iterations = 3)
                ret,bg = cv2.threshold(bgt,1,128,1)
                marker = cv2.add(fg,bg)
                marker32 = np.int32(marker)
                cv2.watershed(image,marker32)
                m = cv2.convertScaleAbs(marker32)
                ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                return thresh
        elif arg == 0:
                return image_gray
        return image_gray

# kuvan filterointi
def blur(arg,image_):
        if arg == 1:#Averaging
                blurred = cv2.blur(image_,(5,5))
        elif arg == 2:#Gaussian Blurring
                blurred = cv2.GaussianBlur(image_,(5,5),0)
        elif arg == 3:#Median Blurring
                blurred = cv2.medianBlur(image_,5)
        elif arg == 4:#Bilateral Filtering
                blurred = cv2.bilateralFilter(image_,9,75,75)
        elif arg == 5:#pyramid mean shift filtering   huom vain varikuvalle
                blurred = cv2.pyrMeanShiftFiltering(image_, 5, 7)
        elif arg == 0:
                blurred = image_
        return blurred

# laske cosini kolmen pisteen valilla
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

# etsi suurin contour jossa on 4 kulmaa ja ne on tarpeeksi lahella 90 astetta
def contourfindrectangle(image,image_bw):
        global centroidx, centroidy, areafoundfirsttime, areafound
        biggest = []
        image, contours, hierarchy = cv2.findContours(image_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        maximumarea = 0
        for cnt in contours:                
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)  # alkuperainen: 0.02*cnt_len
                area = cv2.contourArea(cnt)
                
                if len(cnt) == 4 and area > 1000 and cv2.isContourConvex(cnt):
                        orig = cnt
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                        
                        if max_cos < 0.5:  # kulmien asteiden heitto 90sta asteesta
                                if maximumarea < area: # tallenna suurin contour
                                        if maximumarea != 0:
                                                biggest.pop()
                                        biggest.append(orig)
                                        maximumarea = area
                               
        if len(biggest) > 0: #etsi centroid (jos alue on loytynyt)
                areafoundfirsttime = True
                areafound = True
                m = cv2.moments(biggest[0])
                centroidx = int(m['m10']/m['m00'])
                centroidy = int(m['m01']/m['m00'])                
                return biggest[0],(centroidx,centroidy) # pelialue loytyi
        areafound = False
        return 0,(centroidx,centroidy) # pelialuetta ei loytynyt

#etsi contour jossa on edellisen framen centroid
def contourThatHasCentroid(image_bw):
        global centroidx, centroidy, areafound
        contours, hierarchy = cv2.findContours(image_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
            cnt = cv2.convexHull(cnt)
            #tarkista onko edellinen centroid uudessa alueessa
            insidearea = cv2.pointPolygonTest(cnt, (centroidx, centroidy), False) 
            if insidearea == 1:
                #laske centroid
                m = cv2.moments(cnt)
                centroidx = int(m['m10']/m['m00'])
                centroidy = int(m['m01']/m['m00'])
                areafound = True
                return cnt , (centroidx,centroidy) # pelialue loytyi
        areafound = False
        return 0,(centroidx,centroidy) # pelialuetta ei loytynyt

# ota kuvia kamerasta
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):


def mockupalgorithm(input_array):
        image = input_array ## oletetaan olevan valmiiksi oikeassa muodossa

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # muuta kuva hsv vareiksi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # segmentointi, jos parametri on:
        # 0 == ei mitaan
        # 1 == threshold (otsu)
        # 2 == #watershed kayttaen aina otsua
        # 3 == adaptive threshold
        # 4 == grabCut
        # 5 == watershed kayttaen edellista kuvaa
        image_bw = segmentation(1, image_gray);

        # canny reunantunnistus
        # canny = cv2.Canny(image_gray,100,200)

        # dilation
        # image_bw = cv2.dilate(image_bw, None)

        cnt, origin = contourfindrectangle(image, image_bw)  # eka frame tai kokoajan

        return cnt
