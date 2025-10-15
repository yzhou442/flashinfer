"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch

import flashinfer


def benchmark_prefill_paged_kv():
    """性能测试: 比较 FA2 vs FA3 的 Batch Prefill with Paged KV Cache"""
    
    # 测试配置
    batch_size = 1
    seq_len = 500  # 长序列测试
    num_qo_heads = 4
    num_kv_heads = 1
    head_dim = 128
    page_size = 4096
    kv_layout = "HND"
    
    print("=" * 80)
    print("Benchmark: BatchPrefillWithPagedKVCacheWrapper (FA2 vs FA3)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  num_qo_heads: {num_qo_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  page_size: {page_size}")
    print(f"  kv_layout: {kv_layout}")
    print("=" * 80)
    
    # 准备数据
    q = torch.randn(batch_size * seq_len, num_qo_heads, head_dim).to(0).half()
    
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    
    kv_data = torch.randn(
        total_num_pages, 2, num_kv_heads, page_size, head_dim
    ).to(0).half()
    
    # 索引数据
    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int().to(0)
    kv_indptr = torch.arange(0, total_num_pages + 1, num_pages_per_seq).int().to(0)
    kv_indices = torch.arange(0, total_num_pages + 256).int().to(0)  # +256 padding
    last_page_len = torch.full(
        (batch_size,), 
        (seq_len - 1) % page_size + 1, 
        dtype=torch.int32
    ).to(0)
    
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8).to(0)
    warmup_times = 16
    profiling_times = 1000
    # ========== 测试 1: FA2 Backend ==========
    print("\n" + "=" * 80)
    print("Test 1: FA2 Backend (FlashAttention-2)")
    print("=" * 80)
    
    wrapper_fa2 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fa2.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
    )
    
    # Warm-up
    print("Warming up FA2...")
    for _ in range(warmup_times):
        o_fa2, lse_fa2 = wrapper_fa2.run_return_lse(q, kv_data)
    torch.cuda.synchronize()
    
    # Profiling
    print("Profiling FA2...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = profiling_times
    starter.record()
    for _ in range(repetitions):
        o_fa2, lse_fa2 = wrapper_fa2.run_return_lse(q, kv_data)
    ender.record()
    torch.cuda.synchronize()
    total_time_fa2 = starter.elapsed_time(ender)
    avg_time_fa2 = total_time_fa2 / repetitions
    print(f"✅ FA2 Average time: {avg_time_fa2:.6f} ms ({repetitions} runs)")
    
    # ========== 测试 2: FA3 Backend ==========
    print("\n" + "=" * 80)
    print("Test 2: FA3 Backend (FlashAttention-3 / Hopper)")
    print("=" * 80)
    
    wrapper_fa3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa3"
    )
    wrapper_fa3.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
    )
    
    # Warm-up
    print("Warming up FA3...")
    for _ in range(warmup_times):
        o_fa3, lse_fa3 = wrapper_fa3.run_return_lse(q, kv_data)
    torch.cuda.synchronize()
    
    # Profiling
    print("Profiling FA3...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = profiling_times
    starter.record()
    for _ in range(repetitions):
        o_fa3, lse_fa3 = wrapper_fa3.run_return_lse(q, kv_data)
    ender.record()
    torch.cuda.synchronize()
    total_time_fa3 = starter.elapsed_time(ender)
    avg_time_fa3 = total_time_fa3 / repetitions
    print(f"✅ FA3 Average time: {avg_time_fa3:.6f} ms ({repetitions} runs)")
    
    # ========== 性能对比 ==========
    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print(f"FA2 (FlashAttention-2):  {avg_time_fa2:.6f} ms")
    print(f"FA3 (FlashAttention-3):  {avg_time_fa3:.6f} ms")
    
    speedup = avg_time_fa2 / avg_time_fa3
    if speedup > 1:
        print(f"🚀 FA3 is {speedup:.2f}x faster than FA2")
    else:
        print(f"⚠️  FA2 is {1/speedup:.2f}x faster than FA3")
    
    # ========== 验证结果一致性 ==========
    print("\n" + "=" * 80)
    print("Correctness Verification")
    print("=" * 80)
    lse_diff = torch.abs(lse_fa2 - lse_fa3).max().item()
    output_diff = torch.abs(o_fa2 - o_fa3).max().item()
    print(f"LSE max diff: {lse_diff:.6e}")
    print(f"Output max diff: {output_diff:.6e}")
    
    try:
        torch.testing.assert_close(lse_fa2, lse_fa3, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(o_fa2, o_fa3, rtol=1e-3, atol=1e-3)
        print("✅ Results match within tolerance!")
    except AssertionError as e:
        print(f"⚠️  Results differ: {e}")
    
    print("=" * 80)
    print("Benchmark completed!")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_prefill_paged_kv()
