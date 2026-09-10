import os, sys, math, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

from openpose3d_editor import *
from PIL import Image
def shot(sk, cam, W, H):
    aspect=W/H; fh=700*0.92; fw=fh*aspect
    x0,y0,x1,y1=(900-fw)/2,(700-fh)/2,(900+fw)/2,(700+fh)/2
    s=W/(x1-x0); k=cam.zoom*s
    right,up,fwd=cam.basis()
    parts=[]
    for part in body_parts(sk):
        o=[]
        for kind,c,ax,r in part:
            pc=cam.project(c)
            o.append((kind,((pc[0]-x0)*s,(pc[1]-y0)*s,pc[2]*k),
                      tuple((vdot(u,right),-vdot(u,up),vdot(u,fwd)) for u in ax),
                      tuple(v*k for v in r)))
        parts.append(o)
    return render_depth(parts,W,H,blend=2.0*k)
if __name__ == "__main__":
    sk=Skeleton(); c=Camera(900,700)
    t=time.time(); img=shot(sk,c,512,768); print("ellipsoids=%d render %.2fs" % (sum(len(p) for p in body_parts(sk)), time.time()-t))
    img.save(os.path.join(OUT, "m_front.png"))
    c2=Camera(900,700); c2.yaw=math.radians(90); shot(sk,c2,512,768).save(os.path.join(OUT, "m_side.png"))
    f=Skeleton(preset_params("Female, average")); shot(f,Camera(900,700),512,768).save(os.path.join(OUT, "f_front.png"))
    c3=Camera(900,700); c3.yaw=math.radians(90); shot(f,c3,512,768).save(os.path.join(OUT, "f_side.png"))
