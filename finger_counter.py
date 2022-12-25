import cv2
import mediapipe as mp
import time


wcam=980
hcam=800

fingercoord=[(8,6),(12,10),(16,14),(20,18)]
thumbcoord=(4,3)
cap=cv2.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
mpdraw=mp.solutions.drawing_utils
ptime=0
cap.set(3,wcam)
cap.set(4,hcam)


while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    imrgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(imrgb)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(frame,f'FPS:-{int(fps)}',(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    if results.multi_hand_landmarks:
        upcount=0
        hlist=[]
        for a in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(frame,a,mphands.HAND_CONNECTIONS)
            for id,lm in enumerate(a.landmark):
                h,w,c=frame.shape
                cx=int(lm.x*w)
                cy=int(lm.y*h)
                hlist.append((cx,cy))
                print((cx,cy))
        if len(hlist)!=0:
            for coord in fingercoord:
                if hlist[coord[0]][1]<hlist[coord[1]][1]:
                    upcount+=1
            if hlist[thumbcoord[0]][0]<hlist[thumbcoord[1]][0]:
                upcount+=1
        cv2.putText(frame,f'{int(upcount)}',(50,300),cv2.FONT_HERSHEY_COMPLEX,6,(0,255,0),5)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1)==ord('e'):
        break
cap.release()
cv2.destroyAllWindows()
