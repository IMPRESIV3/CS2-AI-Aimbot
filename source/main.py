import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import gc
import time
import signal
import threading
from collections import deque
import ctypes
from ctypes import Structure, Union, POINTER, windll, byref, sizeof, pointer
from ctypes.wintypes import LONG, DWORD, ULONG

import torch
import cupy as cp
import numpy as np
import cv2
import win32api
import win32con
from ultralytics import YOLO

from config import (
    aaMovementAmp, useMask, maskHeight, maskWidth, aaQuitKey,
    confidence, headshot_mode, cpsDisplay, visuals, centerOfScreen, screenShotWidth
)
import gameSelection


# ==========================
# === ctypes mouse input ===
# ==========================
class MOUSEINPUT(Structure):
    _fields_ = [
        ('dx', LONG),
        ('dy', LONG),
        ('mouseData', DWORD),
        ('dwFlags', DWORD),
        ('time', DWORD),
        ('dwExtraInfo', POINTER(ULONG)),
    ]


class INPUT_UNION(Union):
    _fields_ = [('mi', MOUSEINPUT)]


class INPUT(Structure):
    _fields_ = [('type', DWORD), ('union', INPUT_UNION)]


def send_relative_mouse_move(dx: int, dy: int):
    """Send smooth relative mouse movement using SendInput."""
    if dx == 0 and dy == 0:
        return
    extra = ctypes.c_ulong(0)
    ii_ = INPUT()
    ii_.type = 0  # INPUT_MOUSE
    ii_.union.mi = MOUSEINPUT(dx, dy, 0, win32con.MOUSEEVENTF_MOVE, 0, pointer(extra))
    windll.user32.SendInput(1, byref(ii_), sizeof(ii_))


def mouse_click():
    """Simulate a single left mouse click."""
    windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.01)
    windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


# ==========================
# === Display Threading ====
# ==========================
class DisplayThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            if self.frame is not None:
                cv2.imshow("Live Feed", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        cv2.destroyAllWindows()


# ==========================
# ===== Helper Routines ====
# ==========================
def apply_mask_on_gpu(npImg_cp):
    """Apply left/right mask on the GPU array (cupy) if enabled."""
    if not useMask:
        return npImg_cp
    from config import maskSide
    side = maskSide.lower()
    if side == "right":
        npImg_cp[:, -maskHeight:, -maskWidth:, :] = 0
    elif side == "left":
        npImg_cp[:, -maskHeight:, :maskWidth, :] = 0
    return npImg_cp


def pick_closest_target_to_center(targets, cX, cY):
    """Pick target with minimal distance to screen center."""
    return min(targets, key=lambda t: t[5])  # t[5] = dist


# ==========================
# ===== Main Execution =====
# ==========================
def main():
    # ---- graceful exit handler ----
    def handle_exit(signum, frame):
        print("\n[EXIT] Graceful shutdown requested...")
        try:
            disp.running = False
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            camera.stop()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    gc.disable()
    torch.backends.cudnn.benchmark = True

    print("Starting YOLOv8 TensorRT AI Aim + Triggerbot...")

    # --- Select game region ---
    camera, cWidth, cHeight = gameSelection.gameSelection()

    # --- Load TensorRT Engine ---
    engine_path = "yolov8m.engine"
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    print("Loading YOLOv8 TensorRT engine...")
    model = YOLO(engine_path)
    print("Model loaded successfully!")

    # --- Core aim params ---
    deadzone = 1.5
    max_speed = 10.0
    ema_smooth = 0.2
    ema_dx, ema_dy = 0.0, 0.0

    # --- No-recoil (Alt + LMB) ---
    recoil_strength = 1.8  # set strength; applied only when Alt & LMB are pressed

    # --- Triggerbot params (CapsLock toggle) ---
    triggerbot_enabled = False
    trigger_threshold = 12.0
    trigger_cooldown = 0.08
    trigger_last_shot = 0.0
    last_caps_pressed = False

    # --- CPS / perf ---
    count, sTime = 0, time.time()
    cps_window = deque(maxlen=30)
    target_fps = 180

    # --- Display thread ---
    disp = DisplayThread()
    if visuals:
        disp.start()
    else:
        disp.running = True  # so main loop condition is valid

    with torch.no_grad():
        while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0 and disp.running:
            loop_start = time.time()

            # === CapsLock toggle for triggerbot ===
            caps_pressed = (win32api.GetAsyncKeyState(0x14) & 0x8000) != 0
            if caps_pressed and not last_caps_pressed:
                triggerbot_enabled = not triggerbot_enabled
                print(f"[Triggerbot] {'ON' if triggerbot_enabled else 'OFF'}")
                time.sleep(0.15)  # debounce
            last_caps_pressed = caps_pressed

            # === Frame capture (cupy) ===
            npImg_cp = cp.array([camera.get_latest_frame()])
            if npImg_cp.shape[3] == 4:
                npImg_cp = npImg_cp[:, :, :, :3]
            npImg_cp = apply_mask_on_gpu(npImg_cp)

            # === Convert to RGB numpy for YOLOv8 ===
            frame_bgr = cp.asnumpy(npImg_cp[0])
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # === YOLO inference ===
            results = model.predict(
                frame_rgb,
                conf=confidence,
                verbose=False,
                half=True,
                imgsz=320,         # keep fixed; divisible by 32
                device=0,
                classes=[0],       # person only
            )

            result = results[0]
            boxes = result.boxes

            # === Collect targets ===
            targets = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                dist = np.hypot(mid_x - cWidth, mid_y - cHeight)
                targets.append((mid_x, mid_y, width, height, conf, dist))

            # === Aim logic ===
            if targets:
                # choose closest to center
                xMid, yMid, width, height, conf, _ = pick_closest_target_to_center(targets, cWidth, cHeight)

                # headshot adjustment
                yMid -= height * (0.38 if headshot_mode else 0.2)

                # movement vector
                dx = (xMid - cWidth) * aaMovementAmp
                dy = (yMid - cHeight) * aaMovementAmp

                # deadzone & clamping + EMA smooth
                distance = np.hypot(dx, dy)
                if distance > deadzone:
                    dx = np.clip(dx, -max_speed, max_speed)
                    dy = np.clip(dy, -max_speed, max_speed)
                    ema_dx = ema_dx * (1 - ema_smooth) + dx * ema_smooth
                    ema_dy = ema_dy * (1 - ema_smooth) + dy * ema_smooth

                    # Left Alt = master key for aim + recoil
                    alt_pressed = win32api.GetKeyState(0xA4) < 0
                    if alt_pressed:
                        # add recoil only if shooting while aiming
                        if win32api.GetKeyState(0x01) < 0:
                            ema_dy += recoil_strength

                        send_relative_mouse_move(int(ema_dx), int(ema_dy))
                else:
                    ema_dx, ema_dy = 0.0, 0.0

                # === Triggerbot ===
                if triggerbot_enabled:
                    now = time.time()
                    size_factor = (width + height) / 2
                    dynamic_threshold = max(10, min(25, trigger_threshold + size_factor * 0.02))
                    predicted_distance = np.hypot(dx * 0.8, dy * 0.8)

                    if predicted_distance < dynamic_threshold and (now - trigger_last_shot) > trigger_cooldown:
                        mouse_click()
                        trigger_last_shot = now

            # === Visual Debug (throttled) ===
            if visuals and (count % 10 == 0):
                cp.cuda.Stream.null.synchronize()
                annotated = result.plot()

                disp.frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

            # === CPS / Performance ===
            count += 1
            if (time.time() - sTime) > 1:
                cps_window.append(count)
                avg_cps = sum(cps_window) / len(cps_window)
                if cpsDisplay:
                    print(f"CPS: {count} | avg: {avg_cps:.1f}")
                count, sTime = 0, time.time()

            # === FPS limit ===
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / target_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # cleanup
    try:
        camera.stop()
    except Exception:
        pass
    disp.running = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exception(e)
        cv2.destroyAllWindows()