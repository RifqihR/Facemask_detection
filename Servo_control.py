import time
from adafruit_servokit import ServoKit

mykit=ServoKit(channels=16)

def mapping(v):
    lmin = 0
    lmax = 180
    rmin = 0
    rmax = 270

    lscale = lmax - lmin
    rscale = rmax - rmin

    vl = float(v-lmin)/float(lscale)
    vr =  rmin + (vl * rscale)
    return int(vr)



while True:
    count = 5
    with open ('detection_id.txt','r') as file:
        id = file.read()
        if id == '0':
            while count:
                mykit.servo[0].angle= mapping(0)
                time.sleep(1)
                count -= 1

        mykit.servo[0].angle= mapping(90)

        file.close()
    
