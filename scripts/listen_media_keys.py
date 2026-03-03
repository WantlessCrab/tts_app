from pynput import keyboard
from datetime import datetime


def describe_key(key):
    info = {
        "repr": repr(key),
        "type": type(key).__name__,
    }

    try:
        info["vk"] = key.vk
    except AttributeError:
        info["vk"] = None

    try:
        info["scan"] = key.scan
    except AttributeError:
        info["scan"] = None

    return info


def on_press(key):
    timestamp = datetime.now().isoformat()
    print(f"{timestamp} | PRESSED  | {describe_key(key)}", flush=True)

    # Stop listener when ESC is pressed
    if key == keyboard.Key.esc:
        print("ESC detected. Stopping listener.", flush=True)
        return False


if __name__ == "__main__":
    print("Listener started. Press media keys. Press ESC to stop.", flush=True)
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()