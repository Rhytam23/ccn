# 3-minute pitch outline

1. **Problem (20 s).** Real-time decoding is a bottleneck for fault-tolerant quantum computers; most syndromes are easy, but we pay for the hardest case every time.
2. **Idea (30 s).** Hierarchical decoding: a GPU pre-decoder removes the easy local structure; only the residual reaches the global decoder (MWPM / BP+OSD). Show the architecture diagram from the README.
3. **What we built (40 s).** A batched GPU BP decoder (equal logical-error counts to `ldpc` on the same shots, GPU/CPU batch-throughput ratio 0.98-1.58 on a T4), a local GPU pre-decoder with an accuracy/throughput dial, a benchmark harness, and a head-to-head against NVIDIA's trained Ising pre-decoder.
4. **Evidence (50 s).** Open the website: LER parity (error bars), GPU/CPU throughput ratio vs PyMatching by distance (state that a ratio below 1 means slower), fast-path fraction, p50/p95/p99 latency. State the hardware on screen.
5. **Honesty slide (15 s).** The historical T4 radius-2 quick run had one extra logical error at one point; its modest throughput ratios need a corrected timing rerun. The corrected local T600 quick run showed no surface throughput advantage. The learned Ising model cleared ~97 % of the syndrome in one case, but valid full-pipeline timing remains pending. GPU paths lose at batch size 1, so we do not claim real-time latency. qLDPC uses code-capacity noise. Show accuracy/throughput trade-offs and quote only timings with their protocol and hardware.
6. **Ask / next steps (15 s).** Valid Ising end-to-end timings and more distances, TensorRT/CUDA graphs for batch-1 latency, circuit-level BB codes, real-time streaming.

Backup: record the Colab run as a 2 minute video and embed it in the README in case live access fails.
