# Evaluation Monitoring Guide

## Evaluation Status

**Started**: $(date)
**Command**: 
```bash
python3 scripts/run_evaluation_with_monitoring.py \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --config configs/atlas/crm_harness.yaml \
  --output-dir artifacts/evaluation/atlas_400 \
  --sample 400 \
  --seed 42
```

## Monitor Progress

### Watch Live Log
```bash
tail -f artifacts/evaluation/atlas_400/evaluation.log
```

### Check Process Status
```bash
ps aux | grep run_evaluation_with_monitoring
```

### Check Progress (Session Count)
```bash
wc -l artifacts/evaluation/atlas_400/atlas/sessions.jsonl 2>/dev/null || echo "No sessions yet"
```

### Check for Errors
```bash
grep -i error artifacts/evaluation/atlas_400/evaluation.log | tail -20
```

### Check Learning Growth
```bash
psql "$STORAGE__DATABASE_URL" -c "
  SELECT learning_key, 
         LENGTH(student_learning) as student_len,
         LENGTH(teacher_learning) as teacher_len,
         updated_at
  FROM learning_registry
  WHERE learning_key LIKE 'crm-benchmark-full-evaluation%'
  ORDER BY updated_at DESC LIMIT 1;
"
```

## Auto-Resume Features

- ✅ Automatic retry on crash (up to 10 attempts)
- ✅ Exponential backoff between retries
- ✅ Progress preservation (resumes from completed sessions)
- ✅ Detailed logging with timestamps

## Expected Duration

~2-3 hours for 400 conversations

## Manual Resume

If the process stops, you can manually resume by running the same command:
```bash
python3 scripts/run_evaluation_with_monitoring.py \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --config configs/atlas/crm_harness.yaml \
  --output-dir artifacts/evaluation/atlas_400 \
  --sample 400 \
  --seed 42
```

The script will automatically detect completed sessions and resume from where it left off.
