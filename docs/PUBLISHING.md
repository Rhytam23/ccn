# Publishing checklist (public repo + shareable link)

## Before the first push
- [ ] `git status` shows no secrets: no tokens, `.env`, Drive paths or personal emails (`.gitignore` already excludes `.env`, keys, weights, `.venv`).
- [ ] Notebook outputs are cleared of anything personal (Runtime > Clear all outputs) unless you want the outputs visible.
- [ ] NVIDIA model weights are **not** in the repo (`*.safetensors` is git-ignored). Link to the Hugging Face model pages instead.
- [ ] Replace `Rhytam23/ccn` in the notebooks (`scripts/make_notebooks.py`, then re-run it) and in the README.
- [ ] Add team names to `CITATION.cff` and `LICENSE`.

## Commands
```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

## Shareable website (GitHub Pages)
1. Repo > *Settings > Pages* > Source: *Deploy from a branch* > `main` / `/docs` > Save.
2. After ~1 minute the site is at `https://<user>.github.io/<repo>/` (it serves `docs/index.html`).
3. Put that link at the top of the README and in your hackathon submission.

## Submission package
- Public repo link, website link, 2-3 minute demo video (Colab run + website), the pitch outline in `PITCH.md`.
- State the hardware (e.g. Colab T4) next to every speed-up number.
