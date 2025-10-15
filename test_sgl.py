import torch, runpy, os, sys

os.environ["SGLANG_FLASHINFER"] = "1"
os.environ["SGLANG_ATTENTION_BACKEND"] = "flashinfer"

print(">>> [Profiler] Starting torch.profiler with FlashInfer backend ...")

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(".")
) as prof:
    sys.argv = [
        "sglang.bench_one_batch",
        "--model-path", "Qwen/Qwen3-8B",       
        "--batch-size", "1",                  
        "--input-len", "256",               
        "--output-len", "128",              
        "--decode-attention-backend", "flashinfer",  
        "--prefill-attention-backend", "flashinfer", 
    ]

    runpy.run_module("sglang.bench_one_batch", run_name="__main__")
    torch.cuda.synchronize()
    prof.step()

print(">>> [Profiler] Done! Trace saved to ./")
