"""Print every five minutes while a Codespace demo is running."""

from datetime import datetime
import time


def main():
    print("Keeping the demo terminal active. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            print(f"Demo running: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}", flush=True)
            time.sleep(300)
    except KeyboardInterrupt:
        print("\nKeep-alive stopped.", flush=True)


if __name__ == "__main__":
    main()
