# Colab T4: NVIDIA Ising head-to-head, run 1 (trained weights)

Only one case reached the file: d=9, p=0.003, 20,000 shots, NVIDIA's circuit and noise model, Tesla T4.
`meta.json` recorded `"ising_weights": "trained"`.

| decoder | logical errors | stage 1 | PyMatching stage | syndrome weight left | shots fully resolved |
|---|---|---|---|---|---|
| PyMatching alone | 22 | - | 0.39 s | 100 % | 0 % |
| NVIDIA Ising + PyMatching | 26 | 15.6 s (see below) | **0.10 s** | **2.9 %** | **47 %** |
| ours, local pre-decoder (old radius-1 rule) | 36 | 0.08 s | 0.19 s | 48 % | 0.4 % |

What this shows
* The trained Ising model removes ~97 % of the syndrome weight and fully resolves 47 % of shots, so PyMatching's job is ~3.8x faster, with 26 vs 22 logical errors in the respective runs (no statistical comparison was made).
  Our classical local rule removes about half the weight (and, in its old aggressive form, added errors: 36 vs 22). This is the evidence for using a *learned* pre-decoder.
* **The 15.6 s stage-1 time is not a valid speed measurement.** The adapter warmed the compiled model up on a 256-shot batch but timed a 20,000-shot batch, so torch.compile re-specialised
  inside the timed region. Fixed afterwards (warm-up on the full shape); needs a re-run. Do not quote the 1,273 shots/s from this run.
* The remaining cases (d=9 p=0.005, d=13) are missing from this run.
