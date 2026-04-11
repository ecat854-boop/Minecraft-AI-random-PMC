import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import random
import json
import os
from collections import deque

import numpy as np
import cv2
import mss

from pynput.keyboard import Controller as KB, Key
from pynput.mouse import Controller as MS, Button

# -----------------------
# setup
# -----------------------
kb = KB()
ms = MS()

os.makedirs("mlrun", exist_ok=True)

running = False

ACTIONS = [
    "w","a","s","d",
    "space","shift",
    "1","2","3","4","5","6","7","8","9",
    "left_click","right_click",
    "mouse_move"
]

brain = {a: 0.0 for a in ACTIONS}

# load brain
if os.path.exists("mlrun/brain.json"):
    with open("mlrun/brain.json","r") as f:
        brain = json.load(f)

# history
last_actions = deque(maxlen=6)
last_combos = set()

# -----------------------
# logging
# -----------------------
def log(msg):
    box.insert(tk.END, msg + "\n")
    box.see(tk.END)

# -----------------------
# expression
# -----------------------
def get_expression():
    best = max(brain, key=brain.get)
    score = brain[best]

    if score > 10:
        return "😎 Smart"
    elif score > 5:
        return "🙂 Learning"
    else:
        return "🤖 Random"

# -----------------------
# choose action
# -----------------------
def choose_action():
    if random.random() < 0.55:
        return random.choice(ACTIONS)
    return max(brain, key=brain.get)

# -----------------------
# perform action
# -----------------------
def do_action(action):
    try:
        if action in ["w","a","s","d","1","2","3","4","5","6","7","8","9"]:
            kb.press(action)
            time.sleep(0.12)
            kb.release(action)

        elif action == "space":
            kb.press(Key.space)
            time.sleep(0.15)
            kb.release(Key.space)

        elif action == "shift":
            kb.press(Key.shift)
            time.sleep(0.15)
            kb.release(Key.shift)

        elif action == "left_click":
            ms.click(Button.left)

        elif action == "right_click":
            ms.click(Button.right)

        elif action == "mouse_move":
            # smoother movement
            for _ in range(3):
                ms.move(random.randint(-15,15), random.randint(-15,15))
                time.sleep(0.02)

    except:
        pass

# -----------------------
# CV REWARD SYSTEM
# -----------------------
last_frame = None

def get_cv_reward(frame):
    global last_frame

    if last_frame is None:
        last_frame = frame
        return 0

    diff = abs(frame.mean() - last_frame.mean())
    last_frame = frame

    if diff > 2:
        return 1
    else:
        return -0.3

# -----------------------
# COMBO + PUNISH
# -----------------------
def get_reward(frame, action):
    reward = 0

    # CV reward
    reward += get_cv_reward(frame)

    # combo reward
    if len(last_actions) >= 2:
        combo = tuple(list(last_actions)[-2:] + [action])

        if combo not in last_combos:
            last_combos.add(combo)
            reward += 2

    # spam punishment
    count = last_actions.count(action)
    if count >= 2:
        reward -= count * 0.7

    # variety bonus
    if len(set(last_actions)) > 3:
        reward += 0.3

    last_actions.append(action)

    return reward

# -----------------------
# learning
# -----------------------
def update_brain(action, reward):
    for k in brain:
        brain[k] *= 0.995

    brain[action] += 0.2 * reward

# -----------------------
# save
# -----------------------
def save_brain():
    with open("mlrun/brain.json","w") as f:
        json.dump(brain, f)

# -----------------------
# MAIN LOOP (THREAD SAFE MSS)
# -----------------------
def loop():
    global running

    with mss.mss() as sct:   # ✅ FIXED (inside thread)
        monitor = sct.monitors[1]

        while True:
            if running:
                frame = np.array(sct.grab(monitor))[:, :, :3]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                action = choose_action()
                do_action(action)

                reward = get_reward(gray, action)
                update_brain(action, reward)

                log(f"{action} | r={round(reward,2)} | {get_expression()}")

                save_brain()

            time.sleep(0.3)

# -----------------------
# buttons
# -----------------------
def start():
    global running
    running = True
    log("▶ Started")

def stop():
    global running
    running = False
    log("⏹ Stopped")

def show_brain():
    log("🧠 Learned:")
    for k,v in brain.items():
        log(f"{k} = {round(v,2)}")

# -----------------------
# UI
# -----------------------
root = tk.Tk()
root.title("AI Combo Learner (CV + RL)")

tk.Button(root, text="Start", command=start, bg="green").pack()
tk.Button(root, text="Stop", command=stop, bg="red").pack()
tk.Button(root, text="Show Learning", command=show_brain).pack()

box = scrolledtext.ScrolledText(root, width=65, height=20)
box.pack()

threading.Thread(target=loop, daemon=True).start()

log("🤖 Ready (Thonny Safe + CV + Combo AI)")

root.mainloop()