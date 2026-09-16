import re
import sys
import csv
import os

def parse_ssf_log(log_path):
    with open(log_path, 'r') as f:
        content = f.read()

    seeds = re.split(r'Random seed set to: \d+', content)[1:]
    seed_nums = re.findall(r'Random seed set to: (\d+)', content)

    results = []
    for seed, block in zip(seed_nums, seeds):
        before = parse_metrics_block(block, 'performance_before_continual_training')
        after  = parse_metrics_block(block, 'performance_after_continual_training')
        results.append({
            'seed': int(seed),
            'before_accuracy':  before.get('accuracy'),
            'before_precision': before.get('precision'),
            'before_recall':    before.get('recall'),
            'before_f1':        before.get('f1'),
            'after_accuracy':   after.get('accuracy'),
            'after_precision':  after.get('precision'),
            'after_recall':     after.get('recall'),
            'after_f1':         after.get('f1'),
        })

    return results

def parse_metrics_block(text, section_name):
    pattern = rf'performance_{section_name}.*?(?=performance_|Random seed|$)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return {}
    block = match.group()
    metrics = {}
    for key, pat in [
        ('accuracy',  r'Accuracy\s+([\d.]+)'),
        ('precision', r'Precision\s+([\d.]+)'),
        ('recall',    r'Recall\s+([\d.]+)'),
        ('f1',        r'F1 score\s+([\d.]+)'),
    ]:
        m = re.search(pat, block)
        if m:
            metrics[key] = float(m.group(1))
    return metrics

def summarize(results):
    import statistics
    keys = ['accuracy', 'precision', 'recall', 'f1']
    print(f"\n{'':=<70}")
    for timing in ['before', 'after']:
        print(f"\n[{timing.upper()} continual training]")
        print(f"{'Metric':<12} {'Mean':>10} {'Std':>10} {'Values'}")
        print('-' * 60)
        for k in keys:
            vals = [r[f'{timing}_{k}'] for r in results if r[f'{timing}_{k}'] is not None]
            if vals:
                mean = statistics.mean(vals)
                std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
                vals_str = ', '.join(f'{v:.4f}' for v in vals)
                print(f"{k:<12} {mean:>10.4f} {std:>10.4f}  [{vals_str}]")

def save_csv(results, out_path):
    if not results:
        return
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {out_path}")

if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'nsl_run.log'
    out_csv  = os.path.splitext(log_path)[0] + '_parsed.csv'

    results = parse_ssf_log(log_path)
    if not results:
        print("결과를 찾지 못했습니다. 실험이 아직 완료되지 않았거나 로그 경로를 확인하세요.")
        sys.exit(1)

    print(f"파싱된 seed 수: {len(results)} / 5")
    summarize(results)
    save_csv(results, out_csv)
