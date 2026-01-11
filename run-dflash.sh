export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=/usr/bin/g++-12

CUDA_VISIBLE_DEVICES=2,3 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
python3 -m sglang.launch_server --model /workspace/nctu6/model/Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /workspace/nctu6/model/z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-num-draft-tokens 16 \
    --speculative-draft-attention-backend fa3 \
    --mem-fraction 0.8 --dtype float16 --port 30000 2>&1 | tee run.log
