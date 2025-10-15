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


def test_mlc_failed_case():
    num_requests = 1
    num_tokens = 8
    kv_indptr_1 = torch.tensor([0, 1]).int().to(0)
    kv_indices_1 = torch.tensor([0]).int().to(0)
    kv_last_page_len_1 = torch.tensor([600]).int().to(0)

    
    kv_layout = "HND"
    # kv_indptr_1 = torch.tensor([0, 0, 9]).int().to(0)
    # kv_indices_1 = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11]).int().to(0)
    # kv_last_page_len_1 = torch.tensor([0, 1]).int().to(0)
    num_qo_heads = 4
    num_kv_heads = 1
    page_size = 4096
    head_dim = 128
    q = torch.randn(num_requests, num_qo_heads, head_dim).to(0).half()
    kv_data = torch.randn(12, 2, num_kv_heads, page_size, head_dim).to(0).half()

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    # wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    # wrapper.plan(
    #     kv_indptr_1,
    #     kv_indices_1,
    #     kv_last_page_len_1,
    #     num_qo_heads,
    #     num_kv_heads,
    #     head_dim,
    #     page_size,
    #     pos_encoding_mode="NONE",
    #     data_type=torch.float16,
    #     q_data_type=torch.float16,
    # )
    # o_1, lse_1 = wrapper.run_return_lse(q, kv_data)
    print("kv_indptr_1: ", kv_indptr_1)
    print("kv_indices_1: ", kv_indices_1)
    print("kv_last_page_len_1: ", kv_last_page_len_1)
    print("num_qo_heads: ", num_qo_heads)
    print("num_kv_heads: ", num_kv_heads)
    print("head_dim: ", head_dim)
    print("page_size: ", page_size)
    
    print("\n=== test 1: not use tensor cores ===")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.plan(
        kv_indptr_1,
        kv_indices_1,
        kv_last_page_len_1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    
    # Warm-up
    for _ in range(0):
        o_1, lse_1 = wrapper.run_return_lse(q, kv_data)
    torch.cuda.synchronize()
    
    # Profiling
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1
    starter.record()
    for _ in range(repetitions):
        o_1, lse_1 = wrapper.run_return_lse(q, kv_data)
    ender.record()
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions
    print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")

    print("\n=== test 2: use tensor cores ===")
    wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    wrapper_tensor_cores.plan(
        kv_indptr_1,
        kv_indices_1,
        kv_last_page_len_1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    
    # Warm-up
    for _ in range(0):
        o_1_tc, lse_1_tc = wrapper_tensor_cores.run_return_lse(q, kv_data)
    torch.cuda.synchronize()
    
    # Profiling
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1
    starter.record()
    for _ in range(repetitions):
        o_1_tc, lse_1_tc = wrapper_tensor_cores.run_return_lse(q, kv_data)
    ender.record()
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions
    print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")

    print("lse diff:", torch.abs(lse_1 - lse_1_tc).max().item())
    print("output diff:", torch.abs(o_1 - o_1_tc).max().item())

    torch.testing.assert_close(lse_1, lse_1_tc, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_1, o_1_tc, rtol=1e-3, atol=1e-3)
    print("✅ all tests passed!")


if __name__ == "__main__":
    test_mlc_failed_case()
