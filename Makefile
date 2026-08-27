MODEL ?= stevebottos/qwen3.5-0.8b-find-and-interpret
DTYPE ?= bfloat16

.PHONY: vllm-up vllm-down

# make vllm-up MODEL=... DTYPE=bfloat16
# mounts the host's HF cache so the container picks up your `huggingface-cli
# login` credentials -- needed since the model repo is private
vllm-up:
	docker run -d --name vllm-seek --gpus all --ipc=host \
		-p 8000:8000 \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		vllm/vllm-openai:latest \
		--model $(MODEL) --served-model-name seek \
		--max-model-len 2048 --gpu-memory-utilization 0.8 --dtype $(DTYPE) \
		--enable-prefix-caching

vllm-down:
	docker stop vllm-seek && docker rm vllm-seek
