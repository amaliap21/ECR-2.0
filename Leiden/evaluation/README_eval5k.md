# 5k-per-topic time-based evaluation (all 6 topics)

Re-evaluates all **6 topics** (vaksin, indonesia_gelap, boikot, korupsi, ijazah,
mbg) with **5,000 time-based gold samples each = 30,000 rows**. Replaces the old
180 error-sample evaluation.

Models evaluated (5): Fine-tuned XLM-RoBERTa, IndoBERT-API (No Limit),
RoBERTa-API (cached), GPT (`gpt-5.4-mini`), IndoBERT (`mdhugol/...`).

## Run order

```bash
cd Leiden/evaluation

# 1. Build the time-based pool (30,000 rows, evenly spaced over each timeline).
python eval5k_build_pool.py            # -> outputs/eval5k/pool.csv

# 2. Run model inference (all resumable; stop/restart any time).
python eval5k_infer.py --models finetuned indobert roberta_api   # local + free HTTP
python eval5k_infer.py --models gpt                              # OpenAI (costs $)
#   -> outputs/eval5k/pred_*.csv   (30,000 rows each when done)

# 3. Enter the gold labels manually (interactive, real-time saved, resumable).
python eval5k_label.py                 # all 6 topics
python eval5k_label.py --topic vaksin  # one topic at a time
#   keys: 1=neg  2=neu  3=pos  |  s=skip  u=undo  q=quit
#   -> outputs/eval5k/gold_labels.csv  (saved after every single answer)

# 4. Assemble pool + gold + predictions into one table.
python eval5k_assemble.py              # -> outputs/summary/eval_all6.csv

# 5. Compute F1 across all 6 topics + overall.
python 05_compute_f1.py                # -> outputs/summary/model_comparison.{csv,png}
```

Steps 2 and 3 are independent — label while inference runs. You can run step 4+5
at any point (even mid-labeling): F1 is computed only on rows you've labeled, and
any model whose inference is unfinished is scored only on the rows it has
(`n_scored` column).

## Output schema (`eval_all6.csv`)
```
cm_cell,cleaned_text,indobert_label,roberta_api_cached_label,manual_label,topic,
gpt_label,roberta_api_cached_confidence,text,indobert_api_label,
indobert_api_confidence,finetuned_label,finetuned_confidence
```

## Notes
- `pool.csv` row id (`uid`) = `<topic>_<00000>`; all caches join on it.
- Time sampling = sort each topic by `date_created`, take evenly-spaced rows.
- Labeling is **blind** by default (model predictions hidden) to avoid biasing
  your gold. Use `python eval5k_label.py --show-preds` to reveal them.
- GPT full run ≈ 1,500 batched API calls (batch of 20). RoBERTa-API uses the
  self-hosted endpoint `103.67.43.247:9443` (must be reachable).
