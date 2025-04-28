# rebuild the rpc proto files
./rebuild_rpc.sh
rm -rf build && pip install . -v
