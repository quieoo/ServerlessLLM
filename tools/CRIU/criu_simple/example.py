# example.py
import time

def main():
    counter = 0
    for _ in range(10):
        print(f"Counter: {counter}")
        counter += 1
        time.sleep(1)

if __name__ == "__main__":
    main()