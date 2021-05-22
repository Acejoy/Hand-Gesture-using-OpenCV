import cv2
import numpy as np

def getRange(image):
    avgH = np.average(image[:,:,0])
    avgL = np.average(image[:,:,1])
    avgS = np.average(image[:,:,2])

    stdH = np.std(image[:,:,0])
    stdL = np.std(image[:,:,1])
    stdS = np.std(image[:,:,2])

    return [avgH-2*stdH, avgL-2*stdL, avgS-2*stdS], [avgH+stdH, avgL+stdL, avgS+stdS]


def calcDistance(pt1, pt2):
    x_diff = (pt1[0]-pt2[0])
    y_diff = (pt1[1]-pt2[1])
    return int(np.sqrt(x_diff**2+y_diff**2))


def getConvexHull(imgSrc, mask):
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(c, returnPoints=False)
        hull[::-1].sort(axis=0)

        if cv2.contourArea(c)/(imgSrc.shape[0]*imgSrc.shape[1]) >= 0.14:
            # print('The area is:', cv2.contourArea(
            #     c)/(imgSrc.shape[0]*imgSrc.shape[1]))

            defects = cv2.convexityDefects(c, hull)
            reducedHullPoints = gethullClusters(hull, c)
            reducedDefects = getDefectClusters(c, defects)

            hullAreaFraction = getHullArea(
                reducedHullPoints)/(imgSrc.shape[0]*imgSrc.shape[1])

            # print('the convex hull area is', hullAreaFraction)
            drawHullPoints(imgSrc, reducedHullPoints)
            drawDefects(imgSrc, c, reducedDefects)

            if hullAreaFraction<=0.19 and hullAreaFraction>=0.17:
                return 1
            elif hullAreaFraction<=0.17:
                return 0
            else:
                return len(reducedDefects)+1

def getHullArea(hullPtList):
    pt_list = []
    for pt in hullPtList:
        pt_list.append(np.asarray([pt]))

    pt_list = np.asarray(pt_list)
    area = cv2.contourArea(pt_list)
    return area


def getDefectClusters(contour, defects):
    clusterPts = []
    for defect in defects:
        dis = defect[0][3]
        ptA = contour[defect[0][0]][0]
        ptB = contour[defect[0][1]][0]
        defectPt = contour[defect[0][2]][0]
        sideA = np.linalg.norm(ptB-defectPt)
        sideB = np.linalg.norm(ptA-defectPt)
        defectSide = np.linalg.norm(ptB-ptA)

        # finding the angle
        alpha = np.arccos((np.square(sideA) + np.square(sideB) -
                           np.square(defectSide))/(2*sideA*sideB))
        # converting it into degrees
        alpha = alpha*(180/np.pi)
        # print(':',alpha)
        if dis >= 3600 and alpha < 90.0:

            clusterPts.append(defect)

    # print()

    return clusterPts

    pass


def getRequiredDefects(defects):
    defectClusters = getDefectClusters(defects)
    pass


def gethullClusters(hull, contour):
    listPoints = []
    for pointIdx in hull:
        listPoints.append(list(contour[pointIdx[0]][0]))

    return getClusters(listPoints)


def drawHullPoints(imgSrc, pointsList):

    for pt in pointsList:
        # print('::',pt)
        imgSrc = cv2.circle(imgSrc, (pt[0], pt[1]), 13, (255, 0, 255), 1)


def getClusters(pointSet, thres=30):
    clusterPts = []
    for pt in pointSet:
        if clusterPts == []:
            clusterPts.append((pt[0], pt[1]))
        else:
            inserted = False
            for clusterCenter in clusterPts:
                if calcDistance(clusterCenter, ((pt[0], pt[1]))) <= thres:
                    clusterCenter = (
                        (clusterCenter[0]+pt[0])//2, (clusterCenter[1]+pt[1])//2)
                    inserted = True
                    break
            if not inserted:
                clusterPts.append((pt[0], pt[1]))

    return clusterPts


def drawDefects(imgSrc, contour, ptsList):

    for i in range(len(ptsList)):
        s, e, f, d = ptsList[i][0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        cv2.line(imgSrc, start, end, [0, 255, 0], 2)
        cv2.circle(imgSrc, far, 5, [0, 0, 255], 2)
        cv2.circle(imgSrc, start, 5, [255, 255, 255], 2)
        cv2.circle(imgSrc, end, 5, [255, 255, 255], 2)


def drawConvexHull(imgSrc, hull):
    cv2.drawContours(imgSrc, [hull], -1, (255, 0, 0), 2)


font = cv2.FONT_HERSHEY_SIMPLEX
thickness =2
fontScale = 1
color = (0,0,0)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap:
    print('Camera cant be opened')
    exit()

writeImage = None
index=0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    cv2.rectangle(frame, (330 , 90), (620, 400), (255,255,255), 2)
    roi = frame[92:388, 332:618]
    

    # if registeredBackground:
    roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # if lowerHSV == [] and upperHSV == []:   
        #     print('Ha')         
        #     lowerHSV, upperHSV = getRange(roiHSV)
        
    mask = cv2.inRange(roiHSV, (36,46,141), (88, 166, 211))
    mask = 255-mask

    try:
        detection = getConvexHull(roi, mask)
        if detection !=None:
            frame = cv2.putText(frame, str(detection), (350,117), font, 
                    fontScale, color, thickness, cv2.LINE_AA)        
    except:
        frame = cv2.putText(frame, 'No Detection ', (350,117), font, 
                    fontScale, color, thickness, cv2.LINE_AA)        
        
            # print('===================================>Detected:', detection)

    cv2.imshow('mask', mask)

    cv2.imshow('my roi',roi)
    cv2.imshow('frame', frame)

    if writeImage:
        cv2.imwrite('./res/'+str(index)+'.png', frame[92:388, 332:618])

    pressedKey = cv2.waitKey(1) & 0xFF
    index+=1
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord('w'):
        writeImage = True


cap.release()
cv2.destroyAllWindows()