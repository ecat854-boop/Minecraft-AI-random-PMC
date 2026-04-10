import random
import time
import threading
from pynput import keyboard, mouse

kb = keyboard.Controller()
mouse_ctrl = mouse.Controller()

running = True


def log(msg):
    print(f"[BOT] {msg}")


# ---------------- MOVEMENT ----------------
def movement_loop():
    while running:
        duration = random.uniform(0.8, 2.5)

        log(f"W ➜ holding {duration:.2f}s")

        kb.press('w')
        time.sleep(duration)
        kb.release('w')

        time.sleep(random.uniform(0.5, 1.0))


# ---------------- SMOOTH CAMERA ----------------
def camera_loop():
    global running

    while running:
        # VERY small movement = smoother camera
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)

        x, y = mouse_ctrl.position

        # move in tiny increments (not teleporting cursor)
        mouse_ctrl.position = (x + dx, y + dy)

        log(f"CAMERA ➜ smooth drift dx={dx}, dy={dy}")

        time.sleep(0.06)  # slower updates = smoother feel


# ---------------- CHAT (40% chance) ----------------
def chat_loop():
    sentences = [
        "gg", "nice", "lol", "wp", "bro what", "easy", "crazy game"
    ]

    while running:
        if random.random() < 0.4:

            msg = random.choice(sentences)

            log(f"CHAT ➜ {msg}")

            kb.press('t')
            kb.release('t')

            time.sleep(0.25)
            kb.type(msg)

            kb.press(keyboard.Key.enter)
            kb.release(keyboard.Key.enter)

        time.sleep(random.uniform(3, 6))


# ---------------- STOP (ESC) ----------------
def on_press(key):
    global running
    if key == keyboard.Key.esc:
        running = False
        log("STOPPED 🛑")
        return False


print("Smooth Minecraft bot started 🎮✨")

threading.Thread(target=movement_loop, daemon=True).start()
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=chat_loop, daemon=True).start()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

print("Exited 👍")