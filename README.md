# batch-raman
このツールは、csvデータからミニバッチの作成と各バッチの統計量を計算するツールです。

主にラマンスペクトルのデータを想定していますが、他のデータでも使用可能です。

## 実行環境
- Python 3.10
- poetry(パッケージ管理)
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

## 動作確認
```bash
$ git clone https://github.com/byeron/batch-raman.git
$ cd batch-raman

$ docker pull byeron/poetry
$ docker run -it --rm -v $(pwd):/workspace byeron/poetry bash

# docker内
$ cd /workspace
$ poetry install
$ poetry run python main.py --help
```

## 使い方
```bash
# create test data (default output: data/test.csv)
$ poetry run python main.py testdata

# Output minibatch
$ poetry run python main.py run data/test.csv --batch_size 10 --shuffle

# Output statistics (without output minibatch)
$ poetry run python main.py benchmark data/test.csv --batch_size 10 --shuffle
```
