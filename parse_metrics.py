"""
Parse metrics từ log file của main_experiments_hetero1.py.

Cách dùng:
    python parse_metrics.py <log_file> <output_json>

Ví dụ:
    python parse_metrics.py logs/training_full_*.log results/baseline_v2.0_metrics.json
    python parse_metrics.py logs/ablation_no_cl.log results/ablation_no_cl.json
"""
import json
import re
import sys
from pathlib import Path


# Patterns lấy từ output của main_experiments_hetero1.py
PATTERNS = {
    'AUC':         re.compile(r'AUC\s*:\s*([\d.]+)'),
    'AUPR':        re.compile(r'AUPR\s*:\s*([\d.]+)'),
    'F1':          re.compile(r'^\s*F1\s*:\s*([\d.]+)', re.MULTILINE),
    'Accuracy':    re.compile(r'Accuracy\s*:\s*([\d.]+)'),
    'Recall':      re.compile(r'^\s*Recall\s*:\s*([\d.]+)', re.MULTILINE),
    'Specificity': re.compile(r'Specificity\s*:\s*([\d.]+)'),
    'Precision':   re.compile(r'^\s*Precision\s*:\s*([\d.]+)', re.MULTILINE),
    'top1_precision': re.compile(r'(?:Top-1 Precision|top1_precision)\s*:\s*([\d.]+)'),
    'top1_recall':    re.compile(r'(?:Top-1 Recall|top1_recall)\s*:\s*([\d.]+)'),
    'top1_f1':        re.compile(r'(?:Top-1 F1|top1_f1)\s*:\s*([\d.]+)'),
}

FOLD_TIME_PATTERN = re.compile(r'Fold (\d+) training time:\s*([\d.]+)\s*seconds')
EXEC_TIME_PATTERN = re.compile(r'Total execution time:\s*([\d.]+)\s*seconds')


def _read_log(log_path: Path) -> str:
    """Đọc log file với auto-detect encoding (UTF-8 hoặc UTF-16 LE từ PowerShell Tee)."""
    raw = log_path.read_bytes()
    # PowerShell Tee-Object mặc định UTF-16 LE với BOM ff fe
    if raw[:2] == b'\xff\xfe':
        return raw.decode('utf-16-le', errors='replace')
    if raw[:2] == b'\xfe\xff':
        return raw.decode('utf-16-be', errors='replace')
    if raw[:3] == b'\xef\xbb\xbf':
        return raw[3:].decode('utf-8', errors='replace')
    return raw.decode('utf-8', errors='replace')


def parse_log(log_path: Path) -> dict:
    """Đọc log file và trích xuất các metric.

    Lấy giá trị NẰM SAU heading "FINAL COMPREHENSIVE RESULTS" — đó là
    average across folds.
    """
    text = _read_log(log_path)

    # Tìm phần FINAL block — chỉ extract từ đó trở đi
    final_marker = 'FINAL COMPREHENSIVE RESULTS'
    idx = text.find(final_marker)
    if idx == -1:
        # Fallback: dùng toàn bộ log nhưng lấy LAST match
        section = text
    else:
        section = text[idx:]

    out = {}
    for key, pattern in PATTERNS.items():
        matches = pattern.findall(section)
        if matches:
            # Lấy match cuối — thường nằm trong FINAL block
            try:
                out[key] = float(matches[-1])
            except ValueError:
                out[key] = matches[-1]
        else:
            # Fallback: search full text
            matches_full = pattern.findall(text)
            if matches_full:
                try:
                    out[key] = float(matches_full[-1])
                except ValueError:
                    out[key] = matches_full[-1]
            else:
                out[key] = None

    # Fold timings
    fold_times = FOLD_TIME_PATTERN.findall(text)
    if fold_times:
        out['fold_times_sec'] = [float(t) for _, t in fold_times]
        out['avg_fold_time_sec'] = sum(out['fold_times_sec']) / len(out['fold_times_sec'])

    exec_match = EXEC_TIME_PATTERN.findall(text)
    if exec_match:
        out['total_execution_sec'] = float(exec_match[-1])

    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    log_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        # Try glob
        candidates = sorted(Path().glob(sys.argv[1]))
        if not candidates:
            print(f"❌ No log file found matching: {sys.argv[1]}")
            sys.exit(1)
        log_path = candidates[-1]
        print(f"Using latest match: {log_path}")

    metrics = parse_log(log_path)
    metrics['_source'] = str(log_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved metrics to: {output_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
