import cv2
from cvzone.HandTrackingModule import HandDetector
import math

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.7)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    
    # Detect hands
    hands, img = detector.findHands(img, draw=True)
    
    if len(hands) == 2:
        # Get center positions of both hands
        lmList1 = hands[0]["lmList"]
        lmList2 = hands[1]["lmList"]
        cx1, cy1 = hands[0]['center']
        cx2, cy2 = hands[1]['center']

        # Draw circles at centers
        cv2.circle(img, (cx1, cy1), 10, (0,255,0), cv2.FILLED)
        cv2.circle(img, (cx2, cy2), 10, (0,255,0), cv2.FILLED)
        
        # Compute distance between centers
        dist = math.hypot(cx2 - cx1, cy2 - cy1)
        cv2.putText(img, f'Dist: {int(dist)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
