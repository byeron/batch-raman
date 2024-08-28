import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_output_dir(path, output_dir):
    filename = path.split("/")[-1].split(".")[0]
    output_dir = f"{output_dir}/{filename}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_batch(batch, output_dir, index, suffix=None):
    if suffix is not None:
        path = f"{output_dir}/batch_{index}_{suffix}.csv"
    else:
        path = f"{output_dir}/batch_{index}.csv"
    batch.to_csv(path)
    print(f"Saved to {path}")


def create_batch(df, batch_size, shuffle, random_seed, label) -> list:
    n_splits = len(df) // batch_size  # batchsizeをもとに分割数を計算
    if n_splits < 2:
        click.echo("Warning: batch size must be smaller than the half of the data.")
        click.echo("Skip this process. No batch is created.")
        return

    if shuffle:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_seed
        )
    else:
        click.echo("Note: shuffle is False, random_seed is ignored.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)

    click.echo(df)

    batches = []
    # train_index : test_index = n_splits - 1 : 1  の比率で分割される
    for i, (_, test_index) in enumerate(skf.split(df, df.index)):
        batch = df.iloc[test_index]
        batches.append(batch)
        click.echo(f"batch_{i}, shape: {batch.shape}")
        click.echo(batch)

    return batches


def accumulate_statistics(batches) -> tuple:
    means = []
    stds = []
    for i, _ in enumerate(batches, 1):
        tmp = pd.concat(batches[:i])
        std = tmp.std().mean()
        mean = tmp.mean().mean()
        means.append(mean)
        stds.append(std)

    for i, (mean, std) in enumerate(zip(means, stds)):
        if i == 0:
            continue
        delta_mean = abs(mean - means[i - 1])
        delta_std = abs(std - stds[i - 1])

        # 有効桁数を指定して出力
        k = 5
        mean = round(mean, k)
        std = round(std, k)
        delta_mean = round(delta_mean, k)
        delta_std = round(delta_std, k)
        # click.echo(f"batch_{i}\tmean: {mean}\tstd: {std}\tdelta1: {delta_mean}\tdelta2: {delta_std}")

    return (means, stds)


def calc_welford(batches):
    mean = 0
    m2 = 0
    statistics = []
    for i, batch in enumerate(batches, 1):
        mean, m2 = update_welford(batch.values.flatten(), i, mean, m2)
        variance = m2 / i
        statistics.append(
            {
                "batch": i,
                "mean": mean,
                "variance": variance,
                "std": np.sqrt(variance),
            }
        )
    return statistics


def update_welford(batch, n, mean, m2):
    """Welford's online algorithm"""
    for x in batch:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
    return mean, m2


def each_batch_statistics(batches):
    avg_means = []
    avg_stds = []
    avg_maxs = []
    avg_mins = []
    for i, batch in enumerate(batches):
        # 各バッチの全特徴量の平均、標準偏差、最大値、最小値を計算
        summary = batch.describe().T
        avg_means.append(summary.loc[:, ["mean"]].mean())
        avg_stds.append(summary.loc[:, ["std"]].mean())
        avg_maxs.append(summary.loc[:, ["max"]].mean())
        avg_mins.append(summary.loc[:, ["min"]].mean())

    return (avg_means, avg_stds)


def line_plot(means, stds, output_path):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(211)
    ax.plot(means, label="mean", marker="o", color="tab:blue")
    ax.legend()
    ax.set_xlabel("batch index")
    ax.set_ylabel("value")
    ax.set_title("mean")
    ax = fig.add_subplot(212)
    ax.plot(stds, label="std", marker="o", color="tab:orange")
    ax.legend()
    ax.set_xlabel("batch index")
    ax.set_ylabel("value")
    ax.set_title("std")
    plt.tight_layout()
    fig.savefig(output_path)


@click.group()
def cmd():
    pass


@cmd.command()
@click.argument("path", type=click.Path(exists=True), nargs=1)
@click.option("--batch_size", "-bs", type=int, default=10)
@click.option("--shuffle", "-sf", is_flag=True, default=False)
@click.option(
    "--random_seed",
    "-rs",
    type=int,
    default=None,
    help="[default: None (change seed every time)]",
)
@click.option("--output-dir", "-od", type=click.Path(), default="output")
@click.option("--label", "-l", type=str, default=None)
def run(path, batch_size, shuffle, output_dir, random_seed, label):
    # 出力ディレクトリの作成
    output_dir = make_output_dir(path, output_dir)

    df = pd.read_csv(path, index_col=0, header=0)

    suffix = None
    if label is not None:
        df = df[df.index == label]
        suffix = label

    batches = create_batch(df, batch_size, shuffle, random_seed, label)

    for i, batch in enumerate(batches):
        save_batch(batch, output_dir, i, suffix=suffix)

    # batchにラベルがすべて含まれなければ警告
    if len(batch.index.value_counts()) != len(df.index.value_counts()):
        click.echo(
            "\nWarning: batch does not contain all labels, please rerun with a larger batch size.\n"
        )


@cmd.command()
@click.argument("path", type=click.Path(exists=True), nargs=1)
@click.option("--batch_size", "-bs", type=int, default=10)
@click.option("--shuffle", "-sf", is_flag=True, default=False)
@click.option(
    "--random_seed",
    "-rs",
    type=int,
    default=None,
    help="[default: None (change seed every time)]",
)
@click.option("--output-dir", "-od", type=click.Path(), default="img")
@click.option("--label", "-l", type=str, default=None)
def benchmark(path, batch_size, shuffle, output_dir, random_seed, label):
    # 出力ディレクトリの作成
    output_dir = make_output_dir(path, output_dir)

    df = pd.read_csv(path, index_col=0, header=0)

    suffix = ""
    if label is not None:
        df = df[df.index == label]
        suffix = f"_{label}"

    batches = create_batch(df, batch_size, shuffle, random_seed, label)

    mean, std = each_batch_statistics(batches)
    line_plot(mean, std, f"{output_dir}/each_batch_statistics{suffix}.png")

    # Welford online algorithm
    # 大規模データの統計量を逐次計算するアルゴリズム
    welford = calc_welford(batches)
    line_plot(
        [s["mean"] for s in welford],
        [s["std"] for s in welford],
        f"{output_dir}/welford{suffix}.png",
    )

    means, stds = accumulate_statistics(batches)
    line_plot(means, stds, f"{output_dir}/accumulate_statistics{suffix}.png")

    # batchにラベルがすべて含まれなければ警告
    if len(batches[-1].index.value_counts()) != len(df.index.value_counts()):
        click.echo(
            "\nWarning: batch does not contain all labels, please rerun with a larger batch size.\n"
        )

    # benchmark 終了メッセージ
    click.echo("Benchmark finished.")
    click.echo(
        f"Minibatch is NOT created in this command. Please run 'run' command to create minibatch."
    )


@cmd.command()
@click.option("--row", type=int, default=100)
@click.option("--col", type=int, default=10)
@click.option("--path", type=click.Path(), default="data/test.csv")
def testdata(row, col, path):
    df = pd.DataFrame(
        np.random.rand(row, col),
        columns=[f"col_{i}" for i in range(col)],
        index=np.random.choice(["A", "B", "C"], row, p=[0.15, 0.35, 0.5]),
    )
    # index をA, B, Cの順にする
    df = df.sort_index()
    click.echo(df)
    df.to_csv(path)
    click.echo(f"Saved to {path}")


def main():
    cmd()


if __name__ == "__main__":
    main()
