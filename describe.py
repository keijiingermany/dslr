import sys
from lib.utils import resolve_dataset_path, read_csv_dicts, is_number, safe_float
from lib.stats import mean, std_sample, compute_min, compute_max, percentile_linear


def describe(path: str) -> None:
    fieldnames, rows = read_csv_dicts(path)

    # collect numeric columns (only numeric cells)
    cols = {k: [] for k in fieldnames}
    for r in rows:
        for k in fieldnames:
            v = r.get(k, "")
            if is_number(v):
                cols[k].append(float(v))

    # keep columns with numeric data
    numeric_cols = [k for k in fieldnames if len(cols[k]) > 0]
    if not numeric_cols:
        print(f"ERROR: No numeric columns found: {path}")
        sys.exit(1)

    stats_header = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    print(f"Dataset: {path}")
    print("\t".join(["Feature"] + stats_header))

    for col in numeric_cols:
        data = cols[col]
        n = len(data)
        mu = mean(data)
        sd = std_sample(data, mu)
        mn = compute_min(data)
        mx = compute_max(data)

        data_sorted = data[:]
        data_sorted.sort()

        p25 = percentile_linear(data_sorted, 0.25)
        p50 = percentile_linear(data_sorted, 0.50)
        p75 = percentile_linear(data_sorted, 0.75)

        print(
            "\t".join(
                [
                    col,
                    f"{n}",
                    f"{mu:.6f}",
                    f"{sd:.6f}",
                    f"{mn:.6f}",
                    f"{p25:.6f}",
                    f"{p50:.6f}",
                    f"{p75:.6f}",
                    f"{mx:.6f}",
                ]
            )
        )


def main():
    path = resolve_dataset_path(sys.argv, prefer="datasets/dataset_train.csv")
    describe(path)


if __name__ == "__main__":
    main()