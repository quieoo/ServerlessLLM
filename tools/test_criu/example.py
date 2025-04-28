# example.py
import time

def main():
    counter = 0
    while True:
        print(f"Counter: {counter}")
        counter += 1
        time.sleep(1)

if __name__ == "__main__":
    main()