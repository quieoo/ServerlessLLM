sudo mkdir -p /mnt/tmpfs
sudo mount -t tmpfs -o size=1G tmpfs /mnt/tmpfs
python3 test_io.py
