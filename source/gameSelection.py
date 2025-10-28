import pygetwindow
import time
import bettercam
import pyautogui
from typing import Union

# Import config values
from config import screenShotHeight, screenShotWidth

def gameSelection() -> (bettercam.BetterCam, int, Union[int, None]):
    # List available windows for the user
    try:
        videoGameWindows = pygetwindow.getAllWindows()
        print("=== All Windows ===")
        for index, window in enumerate(videoGameWindows):
            if window.title.strip() != "":
                print(f"[{index}]: {window.title}")
        try:
            userInput = int(input("Please enter the number corresponding to the window you'd like to select: "))
        except ValueError:
            print("You didn't enter a valid number. Please try again.")
            return
        videoGameWindow = videoGameWindows[userInput]
    except Exception as e:
        print(f"Failed to select game window: {e}")
        return None

    # Try to activate that window
    activationRetries = 30
    activationSuccess = False
    while activationRetries > 0:
        try:
            videoGameWindow.activate()
            activationSuccess = True
            break
        except pygetwindow.PyGetWindowException as we:
            print(f"Failed to activate game window: {str(we)}")
            print("Trying again... (you should switch to the game now)")
        except Exception as e:
            print(f"Failed to activate game window: {str(e)}")
            print("Read the relevant restrictions here: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow")
            activationSuccess = False
            activationRetries = 0
            break
        time.sleep(3.0)
        activationRetries -= 1

    if not activationSuccess:
        return None

    print("Successfully activated the game window...")

    # ======================
    # === CENTERED FOV ====
    # ======================
    # Always capture the center of the primary screen
    screen_w, screen_h = pyautogui.size()
    region_left = int((screen_w // 2) - (screenShotWidth // 2))
    region_top = int((screen_h // 2) - (screenShotHeight // 2))
    region_right = region_left + screenShotWidth
    region_bottom = region_top + screenShotHeight
    region = (region_left, region_top, region_right, region_bottom)

    print(f"[INFO] Capture region set to {region} (center FOV around crosshair)")

    # Center point (used by aimbot)
    cWidth: int = screenShotWidth // 2
    cHeight: int = screenShotHeight // 2

    # Initialize camera (capture only central area)
    camera = bettercam.create(region=region, output_color="BGRA", max_buffer_len=512)
    if camera is None:
        return
    camera.start(target_fps=120, video_mode=True)

    return camera, cWidth, cHeight