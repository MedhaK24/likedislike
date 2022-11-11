import cv2
import mediapipe as mp
mp_hands=mp.solutions.hands
hands = mp_hands.Hands() 
mp_draw = mp.solutions.drawing_utils
fingertip=[8,12,16,20]
thumbtip=4
camera=cv2.VideoCapture(0)
def ld(img,hand_landmarks):
    if hand_landmarks:
        for hand in hand_landmarks:
            lmlist=[]
            for id,lm in enumerate(hand.landmark):
                lmlist.append(lm)
            fingerfoldstatus=[]
            for tip in fingertip: 
                if lmlist[tip].x < lmlist[tip-3].x:
                    fingerfoldstatus.append(True)
                else: 
                    fingerfoldstatus.append(False)
            if all (fingerfoldstatus):
                if lmlist[thumbtip].y<lmlist[thumbtip-1].y<lmlist[thumbtip-2].y:
                    cv2.putText(img ,"LIKE", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                if lmlist[thumbtip].y>lmlist[thumbtip-1].y>lmlist[thumbtip-2].y:
                    cv2.putText(img ,"DISLIKE", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2), mp_draw.DrawingSpec((0,255,0),4,2))


while True:
    ret, img= camera.read()
    img=cv2.flip(img,1)
    h,w,c = img.shape
    results=hands.process(img)
    hand_landmarks=results.multi_hand_landmarks
    ld(img,hand_landmarks)
    cv2.imshow('image',img)
    if cv2.waitKey(1)==32:
        break