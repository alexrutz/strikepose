import os, sys, math, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

from openpose3d_editor import *
from PIL import Image
def shot(sk, cam, W, H):
    """Framed exactly as the editor and the CLI frame it."""
    return anatomy_depth_image([sk], cam, frame_rect(900, 700, W / H), W, H)
if __name__ == "__main__":
    sk=Skeleton(); c=Camera(900,700)
    t=time.time(); img=shot(sk,c,512,768); print("ellipsoids=%d render %.2fs" % (sum(len(p) for p in body_parts(sk)), time.time()-t))
    img.save(os.path.join(OUT, "m_front.png"))
    c2=Camera(900,700); c2.yaw=math.radians(90); shot(sk,c2,512,768).save(os.path.join(OUT, "m_side.png"))
    f=Skeleton(preset_params("Female, average")); shot(f,Camera(900,700),512,768).save(os.path.join(OUT, "f_front.png"))
    c3=Camera(900,700); c3.yaw=math.radians(90); shot(f,c3,512,768).save(os.path.join(OUT, "f_side.png"))
