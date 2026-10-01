# GPU runbook (Google Colab, free tier)

1. **Push the repo** to a public GitHub repository (see `PUBLISHING.md`).
2. Colab > *File > Open notebook > GitHub* > your repo > `notebooks/01_run_benchmarks_colab.ipynb`.
3. *Runtime > Change runtime type > T4 GPU*. Edit `GITHUB_REPO` in the first cell.
4. *Runtime > Run all*. Time: ~5 min with `--profile quick`, ~20-40 min with `full` (edit the benchmark cell).
5. The last cell downloads `results_colab.zip`. Unzip into the repo (`results/colab-gpu/`), then locally:
   ```
   python scripts/make_report.py
   git add results docs && git commit -m "Add Colab GPU results" && git push
   ```

## Troubleshooting
| Symptom | Fix |
|---|---|
| `CUDA: False` | Runtime type is CPU; switch to GPU and re-run all |
| Out of memory in BP | lower `chunk` in `BatchedMinSumBP` (default 2048) or use the [[72,12,6]] code |
| Out of memory for d=13 gate training | reduce `train` in `PROFILES["full"]` |
| `torch.sparse` error in `local_predecoder.py` | on CUDA the adjacency is a sparse CSR tensor; if your torch version lacks CSR x dense, change `_sp` to return a dense `torch.tensor(A.toarray())` (fine for d<=13) |
| `cudaq_qec` import fails | `pip install cudaq-qec`, restart runtime; otherwise the harness skips that backend and says so in `meta.json` |
| Session disconnects | results are written per configuration to `results/<tag>/*.csv`; re-run with `--only surface` / `--only qldpc` |
| Free quota exhausted | use Kaggle Notebooks (T4/P100, ~30 GPU h/week) with the same commands |

## Profiling for the demo (Nsight Systems)
Colab does not expose Nsight easily; for a timeline figure use PyTorch's profiler instead:
```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    hybrid.decode_batch(test_dets)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")   # open in chrome://tracing or ui.perfetto.dev
```
