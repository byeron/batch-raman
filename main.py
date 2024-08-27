import os

import click
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

    # train_index : test_index = n_splits - 1 : 1  の比率で分割される
    for i, (_, test_index) in enumerate(skf.split(df, df.index)):
        batch = df.iloc[test_index]
        click.echo(f"batch_{i}, shape: {batch.shape}")
        click.echo(batch)
        save_batch(batch, output_dir, i, suffix=suffix)

    # batchにラベルがすべて含まれなければ警告
    if len(batch.index.value_counts()) != len(df.index.value_counts()):
        click.echo(
            "\nWarning: batch does not contain all labels, please rerun with a larger batch size.\n"
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
