"""
PMC Tool AI-2  v4
=================
Automated Minecraft controller with real computer-vision AI.

Vision layers (all run in background threads):
  1. MOG2 background subtractor (OpenCV) — fast pixel-level motion mask
  2. Lucas-Kanade Optical Flow (OpenCV) — tracks feature points, gives
     motion direction & magnitude so we know WHERE things are moving
  3. HSV colour heuristic (OpenCV) — detects Minecraft mob skin colours
     (zombie green, skeleton off-white, creeper green, pig pink, etc.)
  4. YOLOv8n (Ultralytics) — optional, auto-loaded if ultralytics is
     installed; detects "person" class as proxy for players / humanoid mobs

Reactions:
  • Motion detected        → left click (attack)
  • Motion on LEFT side    → strafe A + click
  • Motion on RIGHT side   → strafe D + click
  • Motion on TOP          → look-up adjust
  • Mob colour detected    → extra right-click (use item) + W toward it
  • YOLO person detected   → immediate attack burst

Controls:   W A S D E | Ctrl+W Shift+W | Space 1-9 | Mouse
Stop:       ESC
"""

import time
import random
import threading
import platform
from collections import deque

import cv2
import numpy as np
import mss
from PIL import Image
from pynput import keyboard, mouse
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

# ── optional YOLO ────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _yolo_model = _YOLO("yolov8n.pt")   # downloads ~6 MB first run
    YOLO_AVAILABLE = True
except Exception:
    _yolo_model = None
    YOLO_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
#  Window detection
# ─────────────────────────────────────────────────────────────────

def get_active_window_title() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value
        elif system == "Darwin":
            from AppKit import NSWorkspace  # type: ignore
            app = NSWorkspace.sharedWorkspace().activeApplication()
            return app.get("NSApplicationName", "")
        elif system == "Linux":
            import subprocess
            r = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                               capture_output=True, text=True)
            return r.stdout.strip()
    except Exception:
        pass
    return ""

def is_minecraft_focused() -> bool:
    title = get_active_window_title().lower()
    return any(kw in title for kw in ["minecraft", "java edition", "bedrock", "mc"])

# ─────────────────────────────────────────────────────────────────
#  Controllers & shared state
# ─────────────────────────────────────────────────────────────────

kb         = KeyboardController()
mse        = MouseController()
stop_event = threading.Event()
_log_lock  = threading.Lock()

# Vision events — set by watcher, cleared after reaction
class VisionEvent:
    NONE       = 0
    MOTION     = 1   # generic motion
    MOB_COLOUR = 2   # mob-skin HSV match
    YOLO_HIT   = 3   # YOLO detected humanoid

_vision_queue: deque = deque(maxlen=5)   # (event_type, side, magnitude)
_vision_lock  = threading.Lock()
_vision_cooldown = 0.0

def _push_vision(ev_type: int, side: str = "centre", magnitude: float = 0.0):
    global _vision_cooldown
    if time.time() < _vision_cooldown:
        return
    with _vision_lock:
        _vision_queue.append((ev_type, side, magnitude))
    _vision_cooldown = time.time() + 0.8

def _pop_vision():
    with _vision_lock:
        return _vision_queue.popleft() if _vision_queue else None

# ─────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────

RESET   = "\033[0m";  CYAN    = "\033[96m";  GREEN   = "\033[92m"
YELLOW  = "\033[93m"; RED     = "\033[91m";  MAGENTA = "\033[95m"
BLUE    = "\033[94m"; ORANGE  = "\033[33m";  BOLD    = "\033[1m"

def log(msg: str, color: str = CYAN) -> None:
    ts = time.strftime("%H:%M:%S")
    with _log_lock:
        print(f"{BOLD}{color}[PMC AI-2 | {ts}]{RESET} {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────
#  HSV mob-colour ranges  (BGR→HSV in OpenCV)
# ─────────────────────────────────────────────────────────────────
#  Each entry: (name, lower_hsv, upper_hsv)
MOB_HSV_RANGES = [
    ("zombie",   np.array([35,  60,  40]),  np.array([75,  255, 180])),  # muted green
    ("creeper",  np.array([55,  80,  60]),  np.array([85,  255, 210])),  # bright green
    ("skeleton", np.array([0,   0,  170]),  np.array([180, 30,  255])),  # near-white
    ("pig",      np.array([0,   60, 180]),  np.array([10,  140, 255])),  # pink
    ("spider",   np.array([0,   0,   20]),  np.array([180, 40,   80])),  # dark grey
    ("enderman", np.array([0,   0,    0]),  np.array([180, 255,  30])),  # near-black
]
MOB_COLOUR_THRESHOLD = 0.025   # fraction of frame pixels matching = mob present

# ─────────────────────────────────────────────────────────────────
#  Vision watcher thread
# ─────────────────────────────────────────────────────────────────

def _vision_watcher() -> None:
    """
    Runs in a daemon thread. Continuously grabs the centre 40% of screen,
    applies three analysis layers and pushes events to the queue.
    """
    log(f"🧠  Vision AI starting  (YOLO={'ON' if YOLO_AVAILABLE else 'OFF'})", YELLOW)

    # OpenCV background subtractor — learns the "still" Minecraft background
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=40, detectShadows=False
    )

    # Lucas-Kanade optical flow params
    lk_params = dict(
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    feature_params = dict(maxCorners=60, qualityLevel=0.25,
                          minDistance=8, blockSize=7)

    prev_grey  = None
    prev_pts   = None
    frame_count = 0
    yolo_every  = 8   # run YOLO every N frames (expensive)

    with mss.mss() as sct:
        mon = sct.monitors[1]
        sw, sh = mon["width"], mon["height"]
        rw = int(sw * 0.40);  rh = int(sh * 0.40)
        rx = mon["left"] + (sw - rw) // 2
        ry = mon["top"]  + (sh - rh) // 2
        region = {"left": rx, "top": ry, "width": rw, "height": rh}

        log(f"👁  Capture region {rw}×{rh} px (centre 40%)", YELLOW)

        while not stop_event.is_set():
            if not is_minecraft_focused():
                time.sleep(0.15)
                prev_grey = None;  prev_pts = None
                continue

            try:
                raw   = sct.grab(region)
                frame = np.array(raw)[:, :, :3]          # H×W×3 BGR
                grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_count += 1

                # ── Layer 1: MOG2 foreground mask ────────────────
                fg_mask  = bg_sub.apply(grey)
                fg_ratio = np.count_nonzero(fg_mask) / fg_mask.size

                if fg_ratio > 0.018:
                    # Determine which horizontal third the motion is in
                    w3 = rw // 3
                    left_sum   = np.count_nonzero(fg_mask[:, :w3])
                    centre_sum = np.count_nonzero(fg_mask[:, w3:2*w3])
                    right_sum  = np.count_nonzero(fg_mask[:, 2*w3:])
                    top_sum    = np.count_nonzero(fg_mask[:rh//2, :])

                    mx = max(left_sum, centre_sum, right_sum)
                    if mx == left_sum:       side = "left"
                    elif mx == right_sum:    side = "right"
                    elif top_sum > fg_mask.size * 0.012: side = "top"
                    else:                   side = "centre"

                    _push_vision(VisionEvent.MOTION, side, fg_ratio)

                # ── Layer 2: Lucas-Kanade optical flow ───────────
                if prev_grey is not None and prev_pts is not None and len(prev_pts) > 0:
                    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_grey, grey, prev_pts, None, **lk_params
                    )
                    if new_pts is not None:
                        good_new = new_pts[status == 1]
                        good_old = prev_pts[status == 1]
                        if len(good_new) > 3:
                            flow = good_new - good_old
                            mag  = np.linalg.norm(flow, axis=1).mean()
                            if mag > 3.5:
                                # Average direction → side classification
                                avg_dx = flow[:, 0].mean()
                                side = "left" if avg_dx < -2 else "right" if avg_dx > 2 else "centre"
                                _push_vision(VisionEvent.MOTION, side, float(mag))

                # Refresh feature points every 12 frames or when lost
                if prev_grey is None or frame_count % 12 == 0 or \
                   prev_pts is None or len(prev_pts) < 5:
                    prev_pts = cv2.goodFeaturesToTrack(grey, mask=None, **feature_params)

                prev_grey = grey

                # ── Layer 3: HSV mob colour detection ───────────
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                for mob_name, lo, hi in MOB_HSV_RANGES:
                    mask = cv2.inRange(hsv, lo, hi)
                    ratio = np.count_nonzero(mask) / mask.size
                    if ratio > MOB_COLOUR_THRESHOLD:
                        # Find centroid of detection
                        cnts, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if cnts:
                            c = max(cnts, key=cv2.contourArea)
                            M = cv2.moments(c)
                            if M["m00"] > 0:
                                cx_mob = M["m10"] / M["m00"]
                                side = "left" if cx_mob < rw/3 else \
                                       "right" if cx_mob > 2*rw/3 else "centre"
                                log(f"🎯  Mob colour [{mob_name}] detected  "
                                    f"side={side}  coverage={ratio:.3f}", ORANGE)
                                _push_vision(VisionEvent.MOB_COLOUR, side, ratio)
                        break

                # ── Layer 4: YOLOv8 (every N frames) ────────────
                if YOLO_AVAILABLE and frame_count % yolo_every == 0:
                    results = _yolo_model(
                        frame, imgsz=320, conf=0.40, verbose=False,
                        classes=[0]   # class 0 = person (humanoid proxy)
                    )
                    if results and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        # Pick largest box
                        areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
                        best  = boxes[np.argmax(areas)]
                        cx_b  = (best[0] + best[2]) / 2
                        side  = "left" if cx_b < rw/3 else \
                                "right" if cx_b > 2*rw/3 else "centre"
                        conf  = float(results[0].boxes.conf[np.argmax(areas)])
                        log(f"🤖  YOLO hit  conf={conf:.2f}  side={side}", RED)
                        _push_vision(VisionEvent.YOLO_HIT, side, conf)

            except Exception as exc:
                log(f"⚠  Vision error: {exc}", RED)

            time.sleep(0.06)   # ~16 fps analysis


# ─────────────────────────────────────────────────────────────────
#  Mouse movement engine  (same as v3 — no spinning, drift-corrected)
# ─────────────────────────────────────────────────────────────────

_cam_drift_x = 0.0
_CAM_MAX_DRIFT = 90
_drift_lock = threading.Lock()

def _reg_dx(dx: int):
    global _cam_drift_x
    with _drift_lock:
        _cam_drift_x += dx

def _correction_dx() -> int:
    with _drift_lock:
        d = _cam_drift_x
    if abs(d) < _CAM_MAX_DRIFT:
        return 0
    return int(-d * random.uniform(0.30, 0.55))

MOUSE_PROFILES = [
    ("micro",        (-12, 12),  (-22,  22),  (28, 50),   (0.28, 0.58)),
    ("look-updown",  (-10, 10),  (-130, 130), (75, 115),  (0.80, 1.40)),
    ("look-up",      (-8,   8),  (-170, -50), (80, 120),  (0.85, 1.50)),
    ("look-down",    (-8,   8),  ( 50,  170), (80, 120),  (0.85, 1.50)),
    ("small-side",   (-40, 40),  (-28,  28),  (50,  85),  (0.55, 1.00)),
    ("medium-side",  (-55, 55),  (-40,  40),  (65, 100),  (0.70, 1.25)),
]
_PROFILE_W = np.array([20, 22, 14, 12, 16, 16], dtype=float)
_PROFILE_W /= _PROFILE_W.sum()

def _bezier_move(dx: int, dy: int, steps: int, duration: float) -> None:
    cx, cy = mse.position
    jit = max(abs(dx), abs(dy)) * 0.20 + 5
    cp1 = np.array([cx + random.uniform(-jit, jit), cy + random.uniform(-jit, jit)])
    cp2 = np.array([cx+dx + random.uniform(-jit, jit), cy+dy + random.uniform(-jit, jit)])
    p0  = np.array([cx, cy]);  p3 = np.array([cx+dx, cy+dy])
    t   = (1 - np.cos(np.linspace(0, 1, steps) * np.pi)) / 2
    pts = ((1-t)**3*p0[:,None] + 3*(1-t)**2*t*cp1[:,None]
           + 3*(1-t)*t**2*cp2[:,None] + t**3*p3[:,None])
    delay = duration / steps
    for i in range(steps):
        if stop_event.is_set() or not is_minecraft_focused():
            return
        mse.position = (int(pts[0, i]), int(pts[1, i]))
        time.sleep(delay)

def fire_mouse_move(tag: str = "", override_dx: int = None) -> threading.Thread:
    idx  = int(np.random.choice(len(MOUSE_PROFILES), p=_PROFILE_W))
    name, dx_r, dy_r, step_r, dur_r = MOUSE_PROFILES[idx]
    corr = _correction_dx()
    if override_dx is not None:
        dx = override_dx
    elif corr != 0:
        dx = int(corr*0.60 + random.uniform(*dx_r)*0.40)
    else:
        dx = int(random.uniform(*dx_r))
    dy       = int(random.uniform(*dy_r))
    steps    = random.randint(*step_r)
    duration = random.uniform(*dur_r)
    _reg_dx(dx)
    prefix = f"[{tag}] " if tag else ""
    log(f"🖱  {prefix}[{name}] Δ({dx:+d},{dy:+d}) {steps}pts {duration:.2f}s", GREEN)
    t = threading.Thread(target=_bezier_move, args=(dx, dy, steps, duration), daemon=True)
    t.start()
    return t

# ─────────────────────────────────────────────────────────────────
#  Vision reaction handler
# ─────────────────────────────────────────────────────────────────

def handle_vision_event(ev_type: int, side: str, magnitude: float) -> None:
    """
    Dispatches the right controller response based on what the AI saw.
    """
    if ev_type == VisionEvent.YOLO_HIT:
        log(f"⚔  YOLO ATTACK  side={side}  mag={magnitude:.2f}", RED)
        # Aim toward side then burst-click
        if side == "left":
            fire_mouse_move("yolo-aim", override_dx=-random.randint(15, 40))
        elif side == "right":
            fire_mouse_move("yolo-aim", override_dx=random.randint(15, 40))
        time.sleep(0.12)
        for _ in range(random.randint(2, 4)):
            dur = random.uniform(0.15, 0.40)
            mse.press(Button.left);  time.sleep(dur);  mse.release(Button.left)
            time.sleep(random.uniform(0.05, 0.15))
        return

    if ev_type == VisionEvent.MOB_COLOUR:
        log(f"🎯  MOB COLOUR REACT  side={side}", ORANGE)
        if side == "left":
            kb.press('a');  time.sleep(random.uniform(0.2, 0.5));  kb.release('a')
        elif side == "right":
            kb.press('d');  time.sleep(random.uniform(0.2, 0.5));  kb.release('d')
        fire_mouse_move("mob-aim")
        time.sleep(0.18)
        lc = random.uniform(0.3, 0.9)
        mse.press(Button.left);  time.sleep(lc);  mse.release(Button.left)
        log(f"✔  Mob attack LClick {lc:.2f}s", CYAN)
        # Walk toward it
        if random.random() < 0.55:
            wd = random.uniform(1.0, 4.0)
            log(f"⌨  Walk toward mob  W {wd:.1f}s", BLUE)
            kb.press('w');  time.sleep(wd);  kb.release('w')
        if random.random() < 0.30:
            rc = random.uniform(0.15, 0.50)
            mse.press(Button.right);  time.sleep(rc);  mse.release(Button.right)
        return

    # Generic MOTION
    log(f"⚡  MOTION  side={side}  mag={magnitude:.3f}", ORANGE)
    if side == "left":
        fire_mouse_move("react", override_dx=-random.randint(10, 30))
    elif side == "right":
        fire_mouse_move("react", override_dx=random.randint(10, 30))
    elif side == "top":
        fire_mouse_move("react", override_dx=0)
    time.sleep(0.10)
    lc = random.uniform(0.25, 0.80)
    mse.press(Button.left);  time.sleep(lc);  mse.release(Button.left)
    log(f"✔  Motion LClick {lc:.2f}s", CYAN)
    if random.random() < 0.30:
        rc = random.uniform(0.12, 0.45)
        mse.press(Button.right);  time.sleep(rc);  mse.release(Button.right)

# ─────────────────────────────────────────────────────────────────
#  Key hold helper
# ─────────────────────────────────────────────────────────────────

def _hold_with_mouse(key_char_or_key, label: str, duration: float,
                     modifier=None, click_chance: float = 0.15) -> None:
    log(f"⌨  Press [{label}] {duration:.1f}s | mouse+clicks={int(click_chance*100)}%", BLUE)
    if modifier:
        kb.press(modifier)
    kb.press(key_char_or_key)

    deadline = time.time() + duration
    threads  = []

    while time.time() < deadline and not stop_event.is_set():
        if not is_minecraft_focused():
            time.sleep(0.1);  continue

        # Vision event takes priority
        ev = _pop_vision()
        if ev:
            handle_vision_event(*ev)
            continue

        remaining = deadline - time.time()
        if remaining <= 0:
            break

        gap = random.uniform(0.30, 0.70)
        time.sleep(min(gap, remaining))
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        t = fire_mouse_move(label)
        threads.append(t)

        if random.random() < click_chance:
            btn  = Button.left if random.random() < 0.65 else Button.right
            hold = random.uniform(0.10, min(0.35, remaining))
            log(f"🖱  [{label}+{'L' if btn==Button.left else 'R'}Click] {hold:.2f}s", MAGENTA)
            mse.press(btn);  time.sleep(hold);  mse.release(btn)

    kb.release(key_char_or_key)
    if modifier:
        kb.release(modifier)
    log(f"✔  Release [{label}]", CYAN)
    for t in threads:
        t.join(timeout=0.5)

# ─────────────────────────────────────────────────────────────────
#  Individual actions
# ─────────────────────────────────────────────────────────────────

def action_w():
    _hold_with_mouse('w', 'W', random.uniform(5, 30), click_chance=0.18)

def action_a():
    _hold_with_mouse('a', 'A', random.uniform(1, 7),  click_chance=0.12)

def action_s():
    _hold_with_mouse('s', 'S', random.uniform(1, 6),  click_chance=0.08)

def action_d():
    _hold_with_mouse('d', 'D', random.uniform(1, 7),  click_chance=0.12)

def action_e():
    dur = random.uniform(0.3, 1.5)
    log(f"⌨  Press [E] {dur:.1f}s", BLUE)
    kb.press('e');  time.sleep(dur);  kb.release('e')
    log("✔  Release [E]", CYAN)

def action_ctrl_w():
    _hold_with_mouse('w', 'Ctrl+W', random.uniform(1, 10),
                     modifier=Key.ctrl, click_chance=0.15)

def action_shift_w():
    _hold_with_mouse('w', 'Shift+W', random.uniform(1, 10),
                     modifier=Key.shift, click_chance=0.22)

def action_space():
    log("⌨  Press [Space] 0.7s", BLUE)
    kb.press(Key.space);  time.sleep(0.7);  kb.release(Key.space)
    log("✔  Release [Space]", CYAN)

def action_number():
    n = random.randint(1, 9)
    log(f"⌨  Press [{n}] 0.5s", BLUE)
    kb.press(str(n));  time.sleep(0.5);  kb.release(str(n))
    log(f"✔  Release [{n}]", CYAN)

def action_mouse_only():
    fire_mouse_move("solo")
    time.sleep(random.uniform(0.20, 0.55))

def action_look_then_walk():
    turns = random.randint(1, 2)
    log(f"👁  Look-then-walk: {turns} turn(s) → W", YELLOW)
    for _ in range(turns):
        if stop_event.is_set():
            return
        fire_mouse_move("look").join(timeout=2.5)
        time.sleep(random.uniform(0.15, 0.40))
    time.sleep(random.uniform(0.15, 0.40))
    if not stop_event.is_set() and is_minecraft_focused():
        _hold_with_mouse('w', 'W→look', random.uniform(4, 18), click_chance=0.15)

def action_left_click():
    dur = random.uniform(0.5, 4.0)
    log(f"🖱  Left click {dur:.1f}s", MAGENTA)
    mse.press(Button.left);  time.sleep(dur);  mse.release(Button.left)
    log("✔  Release LClick", CYAN)

def action_right_click():
    dur = random.uniform(0.5, 4.0)
    log(f"🖱  Right click {dur:.1f}s", MAGENTA)
    mse.press(Button.right);  time.sleep(dur);  mse.release(Button.right)
    log("✔  Release RClick", CYAN)

# ─────────────────────────────────────────────────────────────────
#  Action table
# ─────────────────────────────────────────────────────────────────

ACTION_TABLE = [
    (action_w,              14),
    (action_a,               5),
    (action_s,               3),
    (action_d,               5),
    (action_e,               2),
    (action_ctrl_w,          3),
    (action_shift_w,         5),
    (action_space,           8),
    (action_number,          4),
    (action_mouse_only,     16),
    (action_look_then_walk, 22),
    (action_left_click,      4),
    (action_right_click,     4),
]
_act_fns, _act_raw = zip(*ACTION_TABLE)
_act_w = np.array(_act_raw, dtype=float);  _act_w /= _act_w.sum()

def pick_and_run() -> None:
    ev = _pop_vision()
    if ev:
        handle_vision_event(*ev)
        return
    np.random.choice(_act_fns, p=_act_w)()

# ─────────────────────────────────────────────────────────────────
#  ESC listener
# ─────────────────────────────────────────────────────────────────

def _on_press(key):
    if key == Key.esc:
        log("ESC — stopping.", RED)
        stop_event.set()
        return False

# ─────────────────────────────────────────────────────────────────
#  Banner & main
# ─────────────────────────────────────────────────────────────────

BANNER = f"""{BOLD}{CYAN}
 ██████╗ ███╗   ███╗ ██████╗    ████████╗ ██████╗  ██████╗ ██╗      █████╗ ██╗      ██████╗
 ██╔══██╗████╗ ████║██╔════╝    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██╔══██╗██║     ╚════██╗
 ██████╔╝██╔████╔██║██║            ██║   ██║   ██║██║   ██║██║     ███████║██║      █████╔╝
 ██╔═══╝ ██║╚██╔╝██║██║            ██║   ██║   ██║██║   ██║██║     ██╔══██║██║     ██╔═══╝
 ██║     ██║ ╚═╝ ██║╚██████╗       ██║   ╚██████╔╝╚██████╔╝███████╗██║  ██║███████╗███████╗
 ╚═╝     ╚═╝     ╚═╝ ╚═════╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝
                    PMC Tool AI-2  v4  |  CV/AI-Powered Minecraft Controller
{RESET}"""

def main() -> None:
    print(BANNER)
    log("PMC Tool AI-2  v4  starting…", YELLOW)
    log(f"Vision: MOG2 + OpticalFlow + HSV mobs + YOLO={'ON ✔' if YOLO_AVAILABLE else 'OFF (pip install ultralytics)'}", YELLOW)
    log("Keys : W A S D E | Ctrl+W Shift+W | Space 1-9", YELLOW)
    log("Press [ESC] to stop.", YELLOW)
    print()

    threading.Thread(target=_vision_watcher, daemon=True, name="VisionAI").start()
    listener = keyboard.Listener(on_press=_on_press)
    listener.daemon = True;  listener.start()

    wait_shown = False
    while not stop_event.is_set():
        if not is_minecraft_focused():
            if not wait_shown:
                log("⏳  Waiting for Minecraft to be focused…", YELLOW)
                wait_shown = True
            time.sleep(0.8);  continue

        wait_shown = False
        try:
            pick_and_run()
        except Exception as exc:
            log(f"⚠  Error: {exc}", RED)

        time.sleep(random.uniform(0.08, 0.40))

    listener.stop()
    log("PMC Tool AI-2 stopped. Goodbye!", RED)

if __name__ == "__main__":
    main()