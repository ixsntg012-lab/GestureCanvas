"""
Gesture Drawing — AR Whiteboard
================================
Real-time hand-gesture drawing system using MediaPipe + OpenCV.

GESTURES:
  ☝  Index finger only        → DRAW
  ✌  Index + Middle           → ERASE
  🤏  Pinch (thumb+index)      → GRAB / MOVE stroke
  🤏🤏 Both hands pinch         → ZOOM + ROTATE selected stroke
  ✌✌  Index+Middle both hands  → THICKNESS control (spread = thicker)

FEATURES:
  • Neon glow rendering
  • Freehand → Shape auto-detection (circle / rectangle / line)
  • Move, zoom, rotate any stroke
  • Partial erase (splits strokes)
  • Undo / Redo (60 steps)
  • PNG export, Video recording

KEYBOARD:
  S → Save PNG    R → Start/Stop recording    Z → Undo    Y → Redo    Q → Quit

pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

WIN_W, WIN_H   = 1280, 720
TOOLBAR_H      = 80
DRAW_MIN_DIST  = 3
PINCH_THR      = 40
ERASER_R       = 25
CONFIRM_FRAMES = 6        # frames to hold gesture before mode switches

PALETTE = {
    "Red":    (40,  40,  220),
    "Orange": (10,  130, 255),
    "Yellow": (10,  215, 255),
    "Green":  (30,  185,  55),
    "Blue":   (220,  90,  25),
    "Purple": (190,  40, 170),
    "White":  (255, 255, 255),
    "Black":  ( 15,  15,  15),
}
PAL_NAMES = list(PALETTE.keys())

# ─────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────
@dataclass
class Stroke:
    stype:     str   = "free"
    points:    list  = field(default_factory=list)  # list of (x,y)
    color:     tuple = (15, 15, 15)
    thickness: int   = 4
    offset:    list  = field(default_factory=lambda: [0.0, 0.0])
    scale:     float = 1.0
    rotation:  float = 0.0

# ─────────────────────────────────────────
#  GEOMETRY
# ─────────────────────────────────────────
def dist2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def rot_pt(px, py, cx, cy, a):
    c, s = math.cos(a), math.sin(a)
    dx, dy = px-cx, py-cy
    return cx+dx*c-dy*s, cy+dx*s+dy*c

def apply_tf(pts, offset, scale, rotation):
    if not pts: return []
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2
    out=[]
    for p in pts:
        sx=cx+(p[0]-cx)*scale; sy=cy+(p[1]-cy)*scale
        rx,ry=rot_pt(sx,sy,cx,cy,rotation)
        out.append((rx+offset[0], ry+offset[1]))
    return out

def bbox(s: Stroke):
    pts=apply_tf(s.points,s.offset,s.scale,s.rotation)
    if not pts: return None
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return min(xs),min(ys),max(xs),max(ys)

# ─────────────────────────────────────────
#  SHAPE AUTO-DETECTION
# ─────────────────────────────────────────
def detect_shape(points):
    """
    Classify freehand stroke into: circle | rect | triangle | line | free
    Lenient thresholds — works with shaky hand input.
    Returns (shape_type, clean_points_for_rendering)
    """
    if len(points) < 8:
        return "free", points

    pts = np.array(points, dtype=np.float32)
    xs, ys = pts[:,0], pts[:,1]
    cx, cy = float(xs.mean()), float(ys.mean())
    w  = float(xs.max() - xs.min())
    h  = float(ys.max() - ys.min())
    span = max(w, h, 1.0)

    # ── Need minimum size to snap ────────────────────────────────────────
    if span < 30:
        return "free", points

    # ── LINE ─────────────────────────────────────────────────────────────
    # All points close to the line joining first → last point
    x1,y1 = float(points[0][0]), float(points[0][1])
    x2,y2 = float(points[-1][0]), float(points[-1][1])
    seg_len = math.hypot(x2-x1, y2-y1)
    if seg_len > 30:
        # perpendicular distance of each point from the line
        num = abs((y2-y1)*xs - (x2-x1)*ys + x2*y1 - y2*x1)
        perp_mean = float((num / seg_len).mean())
        if perp_mean < span * 0.12:          # very straight
            return "line", [points[0], points[-1]]

    # ── CIRCLE / ELLIPSE ─────────────────────────────────────────────────
    radii  = np.sqrt((xs-cx)**2 + (ys-cy)**2)
    r_mean = float(radii.mean())
    r_cv   = float(radii.std()) / max(r_mean, 1)   # coefficient of variation
    aspect = min(w, h) / max(w, h, 1)
    closed = dist2(points[0], points[-1]) / span   # 0=closed, 1=open

    if r_cv < 0.30 and aspect > 0.55 and closed < 0.45 and r_mean > 20:
        # Snap to perfect circle
        r = int(r_mean)
        icx, icy = int(cx), int(cy)
        return "circle", [(icx - r, icy), (icx + r, icy)]

    # ── Use Douglas-Peucker to count corners ─────────────────────────────
    pts_cv  = pts.reshape(-1, 1, 2).astype(np.float32)
    arc_len = cv2.arcLength(pts_cv, False)
    epsilon = 0.03 * arc_len                        # lenient simplification
    approx  = cv2.approxPolyDP(pts_cv, epsilon, False)
    n_corners = len(approx)

    # ── TRIANGLE ─────────────────────────────────────────────────────────
    if n_corners == 3 and closed < 0.40:
        corners = [(int(p[0][0]), int(p[0][1])) for p in approx]
        return "triangle", corners

    # ── RECTANGLE / QUADRILATERAL ────────────────────────────────────────
    if 4 <= n_corners <= 6 and closed < 0.40 and aspect > 0.30:
        x0  = int(xs.min()); y0  = int(ys.min())
        x1_ = int(xs.max()); y1_ = int(ys.max())
        return "rect", [(x0, y0), (x1_, y1_)]

    return "free", points


# ─────────────────────────────────────────
#  GESTURE
# ─────────────────────────────────────────
class GestureDetector:
    def __init__(self):
        mp_h = mp.solutions.hands
        self.hands = mp_h.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.75, min_tracking_confidence=0.65)
        self.du = mp.solutions.drawing_utils
        self.conn = mp_h.HAND_CONNECTIONS

    def process(self, rgb):
        return self.hands.process(rgb)

    def lms_px(self, h, w, ht):
        return [(int(lm.x*w), int(lm.y*ht)) for lm in h.landmark]

    def fingers_up(self, lms):
        # [thumb, index, middle, ring, pinky]
        up = [lms[4][0] < lms[3][0]]
        for tip in [8, 12, 16, 20]:
            up.append(lms[tip][1] < lms[tip-2][1])
        return up

    def is_pinch(self, lms):
        return dist2(lms[4], lms[8]) < PINCH_THR

    def two_finger_spread(self, lms0, lms1):
        """Distance between index tips of two hands — for thickness control."""
        return dist2(lms0[8], lms1[8])

    def is_thumbs_up(self, up):
        # thumb up, all fingers down
        return up[0] and not up[1] and not up[2] and not up[3] and not up[4]

    def draw_hand(self, frame, hlms):
        self.du.draw_landmarks(frame, hlms, self.conn,
            self.du.DrawingSpec((200,220,255),1,3),
            self.du.DrawingSpec((100,160,255),2))

# ─────────────────────────────────────────
#  RENDERER
# ─────────────────────────────────────────
class Renderer:
    def render_stroke(self, overlay, mask, s: Stroke,
                      highlight=False, live_end=None):
        """Draw stroke with neon glow effect into overlay/mask."""
        pts = apply_tf(s.points, s.offset, s.scale, s.rotation)
        if not pts: return
        th = max(1, int(s.thickness * s.scale))
        c  = s.color

        # ── collect draw calls as lambdas ────────────────────────────────
        def draw_on(target, color, thickness):
            if s.stype == "free":
                for i in range(1, len(pts)):
                    p1=(int(pts[i-1][0]),int(pts[i-1][1]))
                    p2=(int(pts[i][0]),  int(pts[i][1]))
                    cv2.line(target,p1,p2,color,thickness,cv2.LINE_AA)

            elif s.stype == "line":
                start=(int(pts[0][0]),int(pts[0][1])) if pts else None
                end=live_end if live_end else ((int(pts[-1][0]),int(pts[-1][1])) if len(pts)>1 else None)
                if start and end:
                    cv2.line(target,start,end,color,thickness,cv2.LINE_AA)

            elif s.stype == "triangle":
                # pts = 3 corner points
                if len(pts) >= 3:
                    tri = np.array([(int(p[0]),int(p[1])) for p in pts[:3]], np.int32)
                    cv2.polylines(target,[tri],True,color,thickness,cv2.LINE_AA)

            elif s.stype == "rect":
                start=pts[0] if pts else None
                end=live_end if live_end else (pts[-1] if len(pts)>1 else None)
                if start and end:
                    x1,y1=int(start[0]),int(start[1])
                    x2,y2=int(end[0]),int(end[1])
                    cx2,cy2=(x1+x2)/2,(y1+y2)/2
                    corners=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                    rc=[(int(p[0]),int(p[1])) for p in
                        [rot_pt(px,py,cx2,cy2,s.rotation) for px,py in corners]]
                    cv2.polylines(target,[np.array(rc,np.int32)],True,color,thickness,cv2.LINE_AA)

            elif s.stype == "circle":
                start=pts[0] if pts else None
                end=live_end if live_end else (pts[-1] if len(pts)>1 else None)
                if start and end:
                    cx=int((start[0]+end[0])/2); cy=int((start[1]+end[1])/2)
                    r=int(dist2(start,end)/2)
                    if r>1:
                        cv2.circle(target,(cx,cy),r,color,thickness,cv2.LINE_AA)

        # ── SOFT NEON GLOW — like the yellow crown ───────────────────────
        # Just one soft outer glow + clean bright core — not overdone

        # Outer glow: slightly wider, softly blurred
        glow_outer = np.zeros_like(overlay)
        draw_on(glow_outer, c, th+6)
        glow_outer = cv2.GaussianBlur(glow_outer, (0,0), sigmaX=6, sigmaY=6)

        # Core: clean sharp line in the color
        draw_on(overlay, c, th+2)

        # Bright center: thin white-ish line on top
        bright = tuple(min(255, int(ch*0.6)+140) for ch in c)
        draw_on(overlay, bright, max(1, th-1))

        # Blend glow softly onto overlay
        cv2.add(overlay, (glow_outer * 0.55).astype(np.uint8), overlay)

        # Mask covers glow area
        draw_on(mask, 255, th+14)

        # ── Selection highlight ───────────────────────────────────────────
        if highlight:
            bb = bbox(s)
            if bb:
                x0,y0,x1_,y1_ = [int(v) for v in bb]
                pad=12
                cv2.rectangle(overlay,(x0-pad,y0-pad),(x1_+pad,y1_+pad),(65,140,255),2,cv2.LINE_AA)
                cv2.rectangle(mask,   (x0-pad,y0-pad),(x1_+pad,y1_+pad),200,2,cv2.LINE_AA)
                for hx,hy in [(x0-pad,y0-pad),(x1_+pad,y0-pad),
                               (x0-pad,y1_+pad),(x1_+pad,y1_+pad)]:
                    cv2.circle(overlay,(hx,hy),6,(65,140,255),-1)
                    cv2.circle(mask,   (hx,hy),6,200,-1)

    def nearest(self, strokes, px, py, radius=50):
        best_i, best_d = None, radius
        for i, s in enumerate(strokes):
            pts = apply_tf(s.points, s.offset, s.scale, s.rotation)
            if not pts: continue

            if s.stype == "free":
                for p in pts:
                    d = dist2((px,py), p)
                    if d < best_d: best_d=d; best_i=i
            else:
                # All shapes: bbox hit test
                xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
                x0,y0,x1_,y1_ = min(xs),min(ys),max(xs),max(ys)
                pad = max(20, s.thickness * s.scale)
                if (x0-pad <= px <= x1_+pad) and (y0-pad <= py <= y1_+pad):
                    cx2=(x0+x1_)/2; cy2=(y0+y1_)/2
                    d = dist2((px,py),(cx2,cy2))
                    if d < best_d: best_d=d; best_i=i
        return best_i

# ─────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────
class State:
    def __init__(self):
        self.mode      = "DRAW"       # DRAW | ERASE | GRAB | TRANSFORM
        self.color     = PALETTE["Black"]
        self.thickness = 5
        self.shape     = "free"       # toolbar selection for NEW strokes

        self.strokes: List[Stroke]   = []
        self.undo_stack: List[list]  = []
        self.redo_stack: List[list]  = []

        self.cur:      Optional[Stroke] = None
        self.live_end  = None            # shape preview tip position
        self.sel_idx:  Optional[int]    = None
        self.grab_off  = (0,0)

        # mode stability
        self._cand="DRAW"; self._cnt=0
        self.announce_until=0.0

        # transform state (two-hand pinch zoom)
        self.tf_prev_dist:  Optional[float] = None
        self.tf_prev_angle: Optional[float] = None

    def try_mode(self, m):
        if m == self._cand: self._cnt += 1
        else: self._cand=m; self._cnt=1
        if self._cnt >= CONFIRM_FRAMES and m != self.mode:
            self.mode=m; self._cnt=0
            self.announce_until=time.time()+1.6
            self.cur=None; self.live_end=None
            self.tf_prev_dist=None; self.tf_prev_angle=None

    def force_mode(self, m):
        self.mode=m; self._cand=m; self._cnt=CONFIRM_FRAMES
        self.announce_until=time.time()+1.4
        self.cur=None; self.live_end=None

    def push_undo(self):
        self.undo_stack.append(deepcopy(self.strokes))
        self.redo_stack.clear()
        if len(self.undo_stack)>60: self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(deepcopy(self.strokes))
            self.strokes=self.undo_stack.pop(); self.sel_idx=None

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(deepcopy(self.strokes))
            self.strokes=self.redo_stack.pop(); self.sel_idx=None

# ─────────────────────────────────────────
#  TOOLBAR
# ─────────────────────────────────────────
class Toolbar:
    PAL_R=14; PAL_Y=42; PAL_X0=210; PAL_GAP=37
    SLD_W=110; SLD_Y=42
    SHP_W=54;  SHP_H=26; SHP_Y=16

    def __init__(self):
        self.f=cv2.FONT_HERSHEY_SIMPLEX
        self.b=cv2.FONT_HERSHEY_DUPLEX
        self.SLD_X=self.PAL_X0+len(PAL_NAMES)*self.PAL_GAP+48
        self.SHP_X0=self.SLD_X+self.SLD_W+18

    def _rr(self,img,x,y,w,h,r,col,a=0.82):
        ov=img.copy()
        cv2.rectangle(ov,(x+r,y),(x+w-r,y+h),col,-1)
        cv2.rectangle(ov,(x,y+r),(x+w,y+h-r),col,-1)
        for cx,cy in[(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(ov,(cx,cy),r,col,-1)
        cv2.addWeighted(ov,a,img,1-a,0,img)

    def _t(self,img,t,x,y,sc=0.41,col=(255,255,255),b=False):
        cv2.putText(img,t,(x,y),self.b if b else self.f,sc,col,1,cv2.LINE_AA)

    def _btn(self,img,x,y,w,h,label,active=False):
        self._rr(img,x,y,w,h,7,(60,130,250) if active else (40,42,60),0.90)
        tw,th=cv2.getTextSize(label,self.f,0.37,1)[0]
        self._t(img,label,x+(w-tw)//2,y+(h+th)//2-1,0.37)

    def draw(self,frame,st:State):
        self._rr(frame,32,4,WIN_W-64,TOOLBAR_H-8,12,(16,18,32),0.90)

        mc={"DRAW":(75,205,95),"ERASE":(40,40,228),
            "GRAB":(55,158,248),"TRANSFORM":(195,140,48)}.get(st.mode,(190,190,190))
        self._t(frame,st.mode,50,30,sc=0.54,col=mc,b=True)
        self._t(frame,"MODE",  50,52,sc=0.36,col=mc)

        sub={"DRAW":"☝ index finger","ERASE":"✌ two fingers",
             "GRAB":"🤏 pinch to select","TRANSFORM":"🤏🤏 pinch both hands"}.get(st.mode,"")
        self._t(frame,sub,50,66,sc=0.31,col=(120,132,168))

        for i,name in enumerate(PAL_NAMES):
            cx=self.PAL_X0+i*self.PAL_GAP; cy=self.PAL_Y
            bgr=PALETTE[name]; sel=(st.color==bgr)
            if sel: cv2.circle(frame,(cx,cy),self.PAL_R+4,(255,255,255),-1)
            cv2.circle(frame,(cx,cy),self.PAL_R,bgr,-1)
            cv2.circle(frame,(cx,cy),self.PAL_R,(0,0,0),1,cv2.LINE_AA)

        # Size slider
        sx,sy,sw=self.SLD_X,self.SLD_Y,self.SLD_W; sh=8
        cv2.rectangle(frame,(sx,sy-sh//2),(sx+sw,sy+sh//2),(50,52,72),-1,cv2.LINE_AA)
        fill=int(sw*(st.thickness-1)/29)
        cv2.rectangle(frame,(sx,sy-sh//2),(sx+fill,sy+sh//2),(60,130,250),-1,cv2.LINE_AA)
        kx=sx+fill
        cv2.circle(frame,(kx,sy),10,(205,220,255),-1)
        cv2.circle(frame,(kx,sy),10,(60,130,250),2,cv2.LINE_AA)
        self._t(frame,f"Size {st.thickness}",sx,sy-13,sc=0.35,col=(145,170,220))

        # Shape buttons
        for i,(stype,lbl) in enumerate([("free","Pen"),("line","Line"),("rect","Rect"),("circle","Circ")]):
            bx=self.SHP_X0+i*(self.SHP_W+4)
            self._btn(frame,bx,self.SHP_Y,self.SHP_W,self.SHP_H,lbl,active=(st.shape==stype))

        ux=WIN_W-108
        self._btn(frame,ux,   self.SHP_Y,50,self.SHP_H,"Undo")
        self._btn(frame,ux+53,self.SHP_Y,50,self.SHP_H,"Redo")

        now=time.time()
        if st.announce_until>now:
            a=min(1.0,(st.announce_until-now)*2.5)
            self._banner(frame,f"{st.mode} MODE",mc,a)

        self._t(frame,
            "☝ Draw  ✌ Erase  🤏 Grab  🤏🤏 Zoom/Rotate  ✌✌ Thickness  |  S Save  R Rec  Z Undo  Y Redo  Q Quit",
            38, WIN_H-10, sc=0.34, col=(115,128,165))

    def _banner(self,frame,text,color,alpha):
        ov=frame.copy()
        tw,th=cv2.getTextSize(text,self.b,1.0,2)[0]
        bx=(WIN_W-tw)//2-20; by=WIN_H//2-38
        self._rr(ov,bx,by,tw+40,th+28,12,(14,16,30),0.93)
        cv2.putText(ov,text,(bx+20,by+th+8),self.b,1.0,color,2,cv2.LINE_AA)
        cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

    # hit-tests
    def pal_hit(self,px,py):
        for i,n in enumerate(PAL_NAMES):
            if dist2((px,py),(self.PAL_X0+i*self.PAL_GAP,self.PAL_Y))<=self.PAL_R+5:
                return PALETTE[n]
        return None

    def sld_hit(self,px,py):
        if self.SLD_X-5<=px<=self.SLD_X+self.SLD_W+5 and abs(py-self.SLD_Y)<=18:
            return max(1,min(30,int((px-self.SLD_X)/self.SLD_W*29)+1))
        return None

    def shp_hit(self,px,py):
        for i,st in enumerate(["free","line","rect","circle"]):
            bx=self.SHP_X0+i*(self.SHP_W+4)
            if bx<=px<=bx+self.SHP_W and self.SHP_Y<=py<=self.SHP_Y+self.SHP_H: return st
        return None

    def undo_hit(self,px,py):
        ux=WIN_W-108
        return ux<=px<=ux+50 and self.SHP_Y<=py<=self.SHP_Y+self.SHP_H

    def redo_hit(self,px,py):
        ux=WIN_W-108+53
        return ux<=px<=ux+50 and self.SHP_Y<=py<=self.SHP_Y+self.SHP_H

# ─────────────────────────────────────────
#  APP
# ─────────────────────────────────────────
class App:
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_H)
        self.gest   =GestureDetector()
        self.tb     =Toolbar()
        self.rend   =Renderer()
        self.st     =State()
        self._sdrag =False
        self._ep    =None
        self._eup   =False
        self._tf_i0 =None
        self._tf_i1 =None
        self._shape_toast = 0.0
        self._toast_msg   = ""

        # Recording
        self._recorder   = None
        self._recording  = False
        self._rec_start  = 0.0

        # Thickness gesture state
        self._thick_ref  = None   # reference spread distance

        cv2.namedWindow("Gesture Drawing",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Drawing",WIN_W,WIN_H)
        cv2.setMouseCallback("Gesture Drawing",self._mouse)

    # ── MOUSE ─────────────────────────────
    def _mouse(self,ev,x,y,f,_):
        if y>TOOLBAR_H: return
        st=self.st
        if ev==cv2.EVENT_LBUTTONDOWN:
            h=self.tb.pal_hit(x,y)
            if h: st.color=h; return
            t=self.tb.sld_hit(x,y)
            if t: st.thickness=t; self._sdrag=True; return
            sh=self.tb.shp_hit(x,y)
            if sh:
                if sh != "free":
                    # Place a default shape in center of canvas
                    self._place_shape(sh, st)
                else:
                    st.shape=sh
                return
            if self.tb.undo_hit(x,y): st.undo(); return
            if self.tb.redo_hit(x,y): st.redo(); return
        elif ev==cv2.EVENT_MOUSEMOVE and self._sdrag:
            t=self.tb.sld_hit(x,y)
            if t: st.thickness=t
        elif ev==cv2.EVENT_LBUTTONUP:
            self._sdrag=False

    def _place_shape(self, stype, st):
        """Place shape in center — auto-selected, ready to grab & zoom."""
        st.push_undo()
        cx, cy = WIN_W//2, WIN_H//2 + 50   # slightly below center (avoid toolbar)
        size   = 150
        if stype == "line":
            pts = [(cx-size, cy), (cx+size, cy)]
        elif stype == "rect":
            pts = [(cx-size, cy-80), (cx+size, cy+80)]
        elif stype == "circle":
            pts = [(cx-size, cy), (cx+size, cy)]
        else:
            return
        s = Stroke(stype=stype, points=pts, color=st.color, thickness=max(st.thickness, 4))
        st.strokes.append(s)
        st.sel_idx = len(st.strokes)-1   # auto-select
        # Show a toast hint
        self._shape_toast = time.time() + 2.5

    # ── MAIN LOOP ─────────────────────────
    def run(self):
        print("Gesture Drawing | S=Save  R=Record  Z=Undo  Y=Redo  Q=Quit")
        while True:
            ok, frame = self.cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WIN_W, WIN_H))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = self.gest.process(rgb)

            hands = []
            if res.multi_hand_landmarks:
                for hlms in res.multi_hand_landmarks:
                    self.gest.draw_hand(frame, hlms)
                    lms   = self.gest.lms_px(hlms, WIN_W, WIN_H)
                    up    = self.gest.fingers_up(lms)
                    pinch = self.gest.is_pinch(lms)
                    hands.append((lms, up, pinch))

            # ── Thickness gesture: both hands ✌✌ (index+middle up) ──────
            if len(hands) == 2:
                lms0,up0,_ = hands[0]; lms1,up1,_ = hands[1]
                both_two = (up0[1] and up0[2] and not up0[3] and not up0[4] and
                            up1[1] and up1[2] and not up1[3] and not up1[4])
                if both_two:
                    spread = self.gest.two_finger_spread(lms0, lms1)
                    if self._thick_ref is None:
                        self._thick_ref = (spread, self.st.thickness)
                    else:
                        ref_d, ref_t = self._thick_ref
                        new_t = int(ref_t * (spread / max(ref_d, 1)))
                        self.st.thickness = max(1, min(30, new_t))
                else:
                    self._thick_ref = None
            else:
                self._thick_ref = None

            self._decide_mode(hands)
            self._act(hands, frame)

            display = self._composite(frame)
            self.tb.draw(display, self.st)

            # Recording indicator
            if self._recording:
                self._recorder.write(display)
                elapsed = int(time.time() - self._rec_start)
                cv2.circle(display, (WIN_W-30, 30), 10, (0,0,255), -1)
                cv2.putText(display, f"REC {elapsed}s", (WIN_W-100, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)

            cv2.imshow("Gesture Drawing", display)

            k = cv2.waitKey(1) & 0xFF
            if   k == ord('q'): break
            elif k == ord('z'): self.st.undo()
            elif k == ord('y'): self.st.redo()
            elif k == ord('s'): self._save_png(display)
            elif k == ord('r'): self._toggle_record()

        if self._recording:
            self._recorder.release()
        self.cap.release()
        cv2.destroyAllWindows()

    # ── SAVE PNG ──────────────────────────
    def _save_png(self, display):
        os.makedirs("exports", exist_ok=True)
        fname = f"exports/drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(fname, display)
        self._toast("Saved: " + fname, 2.5)
        print(f"[Saved] {fname}")

    # ── RECORD VIDEO ──────────────────────
    def _toggle_record(self):
        if not self._recording:
            os.makedirs("exports", exist_ok=True)
            fname = f"exports/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self._recorder = cv2.VideoWriter(fname, fourcc, 20.0, (WIN_W, WIN_H))
            self._recording = True
            self._rec_start = time.time()
            self._toast("Recording started...", 1.5)
            print(f"[Recording] {fname}")
        else:
            self._recorder.release()
            self._recorder  = None
            self._recording = False
            self._toast("Recording saved!", 2.0)
            print("[Recording] Stopped & saved")

    def _toast(self, msg, duration=2.0):
        self._toast_msg        = msg
        self._shape_toast      = time.time() + duration
    def _decide_mode(self, hands):
        if not hands: return
        lms0, up0, pinch0 = hands[0]
        st = self.st

        # Two hands both pinching → TRANSFORM (instant, no delay)
        if len(hands) >= 2:
            _, up1, pinch1 = hands[1]
            if pinch0 and pinch1:
                st.mode = "TRANSFORM"
                st._cand = "TRANSFORM"; st._cnt = CONFIRM_FRAMES
                return

        # 🤏 Single hand pinch → GRAB (INSTANT — no confirm delay)
        if pinch0:
            st.mode = "GRAB"
            st._cand = "GRAB"; st._cnt = CONFIRM_FRAMES
            return

        # ☝ One finger (index only) → DRAW
        if up0[1] and not up0[2] and not up0[3] and not up0[4]:
            st.try_mode("DRAW"); return

        # ✌ Two fingers (index+middle) → ERASE
        if up0[1] and up0[2] and not up0[3] and not up0[4]:
            st.try_mode("ERASE"); return

        # Open palm → stop drawing, keep mode
        if all(up0):
            self._end_stroke()

    # ── ACT PER MODE ──────────────────────
    def _act(self,hands,frame):
        st=self.st
        if not hands:
            self._end_stroke(); self._ep=None; return

        lms0,up0,pinch0=hands[0]
        tip=lms0[8]

        # Cursor colour per mode
        cc={"DRAW":(75,205,95),"ERASE":(40,40,228),
            "GRAB":(55,158,248),"TRANSFORM":(195,140,48)}.get(st.mode,(80,200,255))
        cv2.circle(frame,tip,10,cc,2)
        cv2.circle(frame,tip,3,(255,255,255),-1)

        if st.mode=="DRAW":
            # In DRAW mode, draw with index finger (two fingers up activates,
            # but once in DRAW mode we only need the index tip to draw)
            # Allow drawing when at least index is up
            if up0[1]:
                self._do_draw(tip)
                st.live_end=tip
            else:
                self._end_stroke()

        elif st.mode=="ERASE":
            # Erase with index finger tip
            if up0[1]:
                self._do_erase(tip)
                self._ep=tip
            else:
                self._ep=None
                if st.mode!="ERASE": self._eup=False

        elif st.mode=="GRAB":
            mid=((lms0[4][0]+lms0[8][0])//2,
                 (lms0[4][1]+lms0[8][1])//2)
            self._do_grab(pinch0,mid)

        elif st.mode=="TRANSFORM":
            self._do_transform(hands)

        else:
            self._end_stroke()

        if st.mode!="ERASE":
            self._eup=False; self._ep=None

    # ── DRAW ──────────────────────────────
    def _do_draw(self, tip):
        st = self.st
        if st.cur is None:
            st.push_undo()
            st.cur = Stroke(stype="free", color=st.color, thickness=st.thickness)
        if not st.cur.points or dist2(tip, st.cur.points[-1]) >= DRAW_MIN_DIST:
            st.cur.points.append(tip)

    def _end_stroke(self):
        st = self.st
        if st.cur and len(st.cur.points) > 1:
            # ── Auto shape detection ──────────────────────────────────
            detected, simplified = detect_shape(st.cur.points)
            if detected != "free":
                st.cur.stype  = detected
                st.cur.points = simplified
                self._toast(f"Auto-detected: {detected.upper()}!", 1.8)
            st.strokes.append(st.cur)
        st.cur = None; st.live_end = None

    # ── ERASE ─────────────────────────────
    def _do_erase(self, tip):
        self._ep = tip
        tx, ty = tip
        r2 = ERASER_R * ERASER_R

        new_strokes = []
        changed = False

        for s in self.st.strokes:
            pts = apply_tf(s.points, s.offset, s.scale, s.rotation)

            if s.stype == "free":
                # ── Freehand: remove only the touched points ──────────────
                # Split stroke into segments — keep runs of un-erased points
                # Each continuous un-erased run becomes its own stroke
                segment = []
                for i, p in enumerate(s.points):
                    tp = pts[i] if i < len(pts) else p
                    inside = (tp[0]-tx)**2 + (tp[1]-ty)**2 <= r2
                    if inside:
                        # Flush current segment if long enough
                        if len(segment) >= 2:
                            ns = Stroke(stype="free", points=list(segment),
                                        color=s.color, thickness=s.thickness,
                                        offset=list(s.offset), scale=s.scale,
                                        rotation=s.rotation)
                            new_strokes.append(ns)
                        segment = []
                        changed = True
                    else:
                        segment.append(p)
                # Flush final segment
                if len(segment) >= 2:
                    ns = Stroke(stype="free", points=list(segment),
                                color=s.color, thickness=s.thickness,
                                offset=list(s.offset), scale=s.scale,
                                rotation=s.rotation)
                    new_strokes.append(ns)
                elif len(segment) == 1 and not changed:
                    # Stroke untouched (single leftover point edge case)
                    new_strokes.append(s)

            else:
                # ── Shapes: delete whole shape if eraser touches bbox ─────
                if pts:
                    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
                    pad = max(ERASER_R, s.thickness * s.scale)
                    x0,y0,x1_,y1_ = min(xs),min(ys),max(xs),max(ys)
                    hit = (x0-pad <= tx <= x1_+pad) and (y0-pad <= ty <= y1_+pad)
                    if hit:
                        changed = True
                    else:
                        new_strokes.append(s)
                else:
                    new_strokes.append(s)

        if changed:
            if not self._eup:
                self.st.push_undo(); self._eup = True
            self.st.strokes = new_strokes
            if self.st.sel_idx is not None and self.st.sel_idx >= len(self.st.strokes):
                self.st.sel_idx = None

    # ── GRAB (pinch → select + move) ─────────
    def _do_grab(self, pinch, mid):
        st = self.st
        if pinch:
            if st.sel_idx is None:
                # Find nearest stroke to pinch midpoint
                idx = self.rend.nearest(st.strokes, mid[0], mid[1])
                if idx is not None:
                    st.sel_idx  = idx
                    sel = st.strokes[idx]
                    st.grab_off = (mid[0] - sel.offset[0],
                                   mid[1] - sel.offset[1])
                    st.push_undo()
            else:
                # Drag currently selected stroke
                if st.sel_idx < len(st.strokes):
                    sel = st.strokes[st.sel_idx]
                    sel.offset[0] = mid[0] - st.grab_off[0]
                    sel.offset[1] = mid[1] - st.grab_off[1]
        else:
            # Pinch released → always clear so next pinch selects fresh
            st.sel_idx       = None
            st.tf_prev_dist  = None
            st.tf_prev_angle = None

    # ── TRANSFORM (two hand pinch → zoom + rotate selected) ───
    def _do_transform(self, hands):
        """
        రెండు hands తో pinch చేయి:
        - selected box మీద రెండు hands పెట్టి pinch చేయి
        - దూరం పెంచు → ZOOM IN
        - దూరం తగ్గించు → ZOOM OUT
        - angle మార్చు → ROTATE
        """
        st = self.st
        if len(hands) < 2: return

        lms0, up0, pinch0 = hands[0]
        lms1, up1, pinch1 = hands[1]

        # Pinch midpoint of each hand (thumb+index midpoint)
        mid0 = ((lms0[4][0]+lms0[8][0])//2, (lms0[4][1]+lms0[8][1])//2)
        mid1 = ((lms1[4][0]+lms1[8][0])//2, (lms1[4][1]+lms1[8][1])//2)

        # Store fingertip positions for visual feedback
        self._tf_i0 = mid0
        self._tf_i1 = mid1

        cur_dist  = dist2(mid0, mid1)
        cur_angle = math.atan2(mid1[1]-mid0[1], mid1[0]-mid0[0])

        # If no object selected yet, auto-select nearest to midpoint
        overall_mid = ((mid0[0]+mid1[0])//2, (mid0[1]+mid1[1])//2)
        if st.sel_idx is None:
            idx = self.rend.nearest(st.strokes, overall_mid[0], overall_mid[1], radius=500)
            if idx is not None:
                st.sel_idx = idx
                st.push_undo()
            st.tf_prev_dist  = cur_dist
            st.tf_prev_angle = cur_angle
            return

        # Apply zoom + rotate delta
        if st.tf_prev_dist is not None and st.sel_idx < len(st.strokes):
            sel = st.strokes[st.sel_idx]

            # Zoom — ratio of current distance to previous distance
            if st.tf_prev_dist > 0:
                ratio = cur_dist / st.tf_prev_dist
                sel.scale = max(0.05, min(15.0, sel.scale * ratio))

            # Rotate — angle delta
            d_angle = cur_angle - st.tf_prev_angle
            if d_angle >  math.pi: d_angle -= 2*math.pi
            if d_angle < -math.pi: d_angle += 2*math.pi
            sel.rotation += d_angle

        st.tf_prev_dist  = cur_dist
        st.tf_prev_angle = cur_angle

    # ── COMPOSITE ─────────────────────────
    def _composite(self,frame):
        st=self.st
        overlay=np.zeros_like(frame)
        mask=np.zeros((WIN_H,WIN_W),dtype=np.uint8)

        for i,s in enumerate(st.strokes):
            self.rend.render_stroke(overlay,mask,s,highlight=(i==st.sel_idx))

        if st.cur:
            self.rend.render_stroke(overlay,mask,st.cur)

        # Eraser circle
        if self._ep:
            ex,ey=self._ep
            cv2.circle(overlay,(ex,ey),ERASER_R,(200,200,255),2,cv2.LINE_AA)
            cv2.circle(mask,   (ex,ey),ERASER_R,160,2,cv2.LINE_AA)

        # Transform fingertip line feedback
        if st.mode=="TRANSFORM" and self._tf_i0 and self._tf_i1:
            cv2.line(overlay, self._tf_i0, self._tf_i1, (195,140,48), 2, cv2.LINE_AA)
            cv2.circle(overlay, self._tf_i0, 8, (195,140,48), -1)
            cv2.circle(overlay, self._tf_i1, 8, (195,140,48), -1)
            cv2.line(mask, self._tf_i0, self._tf_i1, 200, 2, cv2.LINE_AA)
            cv2.circle(mask, self._tf_i0, 8, 200, -1)
            cv2.circle(mask, self._tf_i1, 8, 200, -1)
        else:
            self._tf_i0=None; self._tf_i1=None

        # Alpha blend over camera
        a3=cv2.merge([mask,mask,mask]).astype(np.float32)/255.0
        out=frame.astype(np.float32)*(1-a3)+overlay.astype(np.float32)*a3
        result = np.clip(out,0,255).astype(np.uint8)

        # Toast notification (save / record / shape detected)
        now = time.time()
        if self._shape_toast > now:
            alpha = min(1.0, (self._shape_toast - now) * 2.5)
            ov2 = result.copy()
            msg = self._toast_msg or "Done!"
            f   = cv2.FONT_HERSHEY_DUPLEX
            tw,th2 = cv2.getTextSize(msg, f, 0.65, 1)[0]
            bx = (WIN_W-tw)//2 - 16
            by = WIN_H - 80
            cv2.rectangle(ov2,(bx-12,by-th2-10),(bx+tw+12,by+12),(18,20,38),-1)
            cv2.putText(ov2, msg, (bx, by), f, 0.65, (80,210,100), 1, cv2.LINE_AA)
            cv2.addWeighted(ov2, alpha, result, 1-alpha, 0, result)

        return result


if __name__=="__main__":
    App().run()