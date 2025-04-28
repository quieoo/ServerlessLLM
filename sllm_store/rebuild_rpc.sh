cd sllm_store/proto
python -m grpc_tools.protoc -I../../proto --python_out=. --grpc_python_out=. ../../proto/storage.proto

FILE="storage_pb2_grpc.py"
if [ -f "$FILE" ]; then
    sed -i '5s|^import storage_pb2 as storage__pb2$|from . import storage_pb2 as storage__pb2|' "$FILE"
fi

cd ../..