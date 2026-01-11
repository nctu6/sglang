export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=/usr/bin/g++-12

CUDA_VISIBLE_DEVICES=2,3 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
python3 -m sglang.launch_server --model /workspace/nctu6/model/meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm PEARL \
    --speculative-draft-model-path /workspace/nctu6/model/meta-llama/Llama-3.2-1B-Instruct \
    --speculative-draft-gpu-id 3 \
    --speculative-num-steps -1 \
    --speculative-auto-steps max=8,mult=4.0 \
    --cuda-graph-max-bs 1 --mem-fraction 0.8 --dtype float16 --port 30000 2>&1 | tee run.log
