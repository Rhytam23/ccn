# 3-minute pitch outline

1. **Problem (20 s).** Real-time decoding is a bottleneck for fault-tolerant quantum computers; most syndromes are easy, but we pay for the hardest case every time.
2. **Idea (30 s).** Hierarchical decoding: a GPU pre-decoder removes the easy local structure; only the residual reaches the exact decoder (MWPM / BP+OSD). Show the architecture diagram from the README.
3. **What we built (40 s).** Local GPU pre-decoder, batched GPU BP with OSD fallback, learned gate with a calibrated threshold, one benchmark harness, Stim data, BB qLDPC codes, optional CUDA-Q QEC backend, reproduction of NVIDIA's Ising baseline.
4. **Evidence (50 s).** Open the website: LER parity (error bars), throughput speed-up vs PyMatching by distance, fast-path fraction, p50/p95/p99 latency. State the hardware on screen.
5. **Honesty slide (15 s).** MLP gate does not scale beyond d=5 (negative result, why local models win); qLDPC uses code-capacity noise; CPU runs are slower than C++ baselines by design.
6. **Ask / next steps (15 s).** Swap in NVIDIA Ising CNN as the pre-decoder, TensorRT/CUDA graphs for batch-1 latency, circuit-level BB codes, real-time streaming.

Backup: record the Colab run as a 2 minute video and embed it in the README in case live access fails.
