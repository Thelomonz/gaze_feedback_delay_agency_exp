# Gaze-Tracking Experiment Program

本プロジェクトは、視線制御における視覚フィードバックの遅延が運動主体感に与える影響を検討する、2段階構成の視線追跡実験プログラムです。

## Attribution / Credits

本プロジェクトは、以下のプロジェクトを基に改変したものです。

原プロジェクト:[acercyc/soa_eye_tracking_exp](https://github.com/acercyc/soa_eye_tracking_exp)
原作者:[acercyc](https://github.com/acercyc)

本研究の運動主体感に関する視線追跡実験に合わせて、実験条件、刺激制御、およびデータ記録処理を変更しています。

## Overview

実験は、相互に関連する次の2段階で構成されます。

1. **視線制御体験段階**：1つの動物画像が、参加者の現在または過去の視線位置に追従します。3種類のフィードバック遅延条件を繰り返し提示します。
2. **制御率評価段階**：各遅延条件を再度1回ずつ提示し、参加者が感じた制御の程度を0～100%で回答します。

すべての条件で、刺激は視線に基づいて移動します。独立変数は、参加者の視線入力が刺激位置へ反映されるまでの遅延時間です。本実験は、制御可能な運動と制御不可能な運動を比較するものではありません。

## Features

### Feedback Delay Conditions

実験全体を通して、次の3種類の視線追従条件を使用します。

- **`locking_a`**：刺激が最新の視線位置へ追従する0.0秒遅延条件
- **`locking_b`**：刺激が約0.5秒前の視線位置へ追従する条件
- **`locking_c`**：刺激が約2.0秒前の視線位置へ追従する条件

各条件には異なる画像をランダムに割り当てます。標準設定では、第1段階で各条件を6回ずつ提示します。試行系列の生成時には、同じ条件が連続しないように試みます。

### Technical Features

- `psychopy_tobii_controller` を介した Tobii アイトラッカーとの連携
- アイトラッカーがない環境で動作確認を行うためのマウス入力
- 5点または9点の Tobii キャリブレーション
- バックグラウンドサンプリングを使用した任意の移動平均平滑化
- Tobii モードにおいて、視線欠損時間を除外する有効提示時間の計測
- 遅延条件、試行時間、評価範囲、画面表示の設定変更
- `.png`、`.jpg`、`.jpeg`、`.bmp` 画像の自動検出
- 参加者ごとの遅延条件と画像のランダムな対応付け
- フレーム単位の CSV、Tobii TSV、実行時設定 JSON の保存

### Control Rating Phase Features

- 各遅延条件を1回ずつ追加提示
- 空白時間と刺激提示時間を個別に設定可能
- 最小値、最大値、初期値、変更量を設定できる評価スライダー
- 左右矢印キーによる評価値の調整
- Space または Enter キーによる回答確定
- 評価値と反応時間の記録

## Requirements

- Python 3
- PsychoPy
- NumPy
- Tobii モードで使用する Tobii Research SDK（`tobii_research`）
- Tobii モードで使用する対応アイトラッカー
- 本リポジトリに同梱されている `psychopy_tobii_controller`
- `GazeStabilizer.py` の任意の GMM 平滑化を使用する場合のみ scikit-learn

## Configuration

### Experimental Design Settings

主な試行条件と評価条件は `src/main.py` で定義します。

```python
config = {
    "design": {
        "blank_duration": 0,
        "movement_duration": 10,
        "trials_per_condition": 6,
        "second_half_blank_duration": 2,
        "second_half_movement_duration": 10,
        "control_rate_initial_percent": 50,
        "control_rate_min_percent": 0,
        "control_rate_max_percent": 100,
        "control_rate_step_percent": 1,
        "control_rate_repeat_interval": 0.11,
        "locking_delays": {
            "locking_a": 0.0,
            "locking_b": 0.5,
            "locking_c": 2.0,
        },
    }
}
```

### Display, Input, and Tobii Settings

画面表示、入力方式、キャリブレーション、平滑化、刺激の設定も同じ辞書内で定義します。

```python
config = {
    "experiment": {
        "fullscreen": True,
        "show_pos_indicator": False,
        "screen_margin": 0.02,
        "background_color": (100, 100, 100),
        "calibration_intro_duration": 2,
        "experiment_start_delay": 2,
        "pos_indicator_radius": 0.005,
        "pos_indicator_color": "red",
        "pos_indicator_opacity": 0.7,
    },
    "controller": {
        "type": "mouse",  # "mouse" or "tobii"
    },
    "tobii": {
        "calibration": True,
        "calibration_points": 5,
        "calibration_range_x": 0.6,
        "calibration_range_y": 0.4,
        "stabilizer_type": "moving_average",
        "stabilizer_moving_average": {
            "buffer_size": 50,
            "sampling_rate": 500,
        },
    },
    "stimulus": {
        "target_type": "image",
        "scale": 0.05,
        "flash": False,
        "images": discover_images(),
    },
    "runtime": {
        "condition_to_image": {},
    },
}
```

## Usage

### Running with Mouse Input

1. `config["controller"]["type"]` を `"mouse"` に設定します
2. `src/images` に3枚以上の画像ファイルを配置します
3. リポジトリのルートディレクトリで実験を実行します

```
python src/main.py
```

4. 入力を求められたら、空でない参加者 ID を入力します
5. 説明画面で Space キーを押します
6. 評価段階では左右矢印キーで値を調整します
7. Space または Enter キーで各回答を確定します
8. 実験を中断または終了する場合は Escape キーを押します

### Running with Tobii Input

1. 対応する Tobii アイトラッカーを接続します
2. PsychoPy と `tobii_research` からデバイスを認識できることを確認します
3. `config["controller"]["type"]` を `"tobii"` に設定します
4. 必要に応じて `config["tobii"]["calibration"]` を設定します
5. 実験を実行します

```
python src/main.py
```

6. 空でない参加者 ID を入力します
7. キャリブレーションが有効な場合、左右の視線マーカーを確認し、Space キーを押してキャリブレーションを完了します
8. 画面上の説明に従って両段階を完了します

## Data

データは、リポジトリ直下に自動作成される `data` ディレクトリへ保存されます。日時には `YYYYMMDDHHMM` 形式を使用します。

- `{subject_id}_exp_{date}.csv`：視線制御体験段階のフレーム単位データ
- `{subject_id}_exp_second_half_{date}.csv`：制御率評価段階のフレーム単位データと回答
- `{subject_id}_tobii_exp_{date}.tsv`：Tobii の生視線データとイベント。Tobii モードでのみ作成
- `{subject_id}_config_exp_{date}.json`：実行時設定、試行系列、条件と画像の対応関係

CSV の主な項目には、視線位置、遅延視線位置、刺激位置、視線欠損状態、試行番号、フレーム番号、時刻、条件名、遅延時間、フェーズ、画像パス、制御率、反応時間が含まれます。

## Project Structure

- `src/main.py`：実験のメインスクリプト
- `src/images/`：自動的に読み込まれる動物画像
- `src/GazeStabilizer.py`：視線座標の平滑化アルゴリズム
- `src/psychopy_tobii_controller/`：PsychoPy と Tobii の連携モジュール
- `src/readme.md`：日本語の使用説明
- `data/`：実行時に作成される実験データの保存先
- `test/`：平滑化処理と並列サンプリングのテスト
- `.github/instructions/`：詳細な実験仕様書

## Classes

本実験では、主に次のクラスで構成されるオブジェクト指向設計を使用しています。

### Core Classes

- `MovingMode`：刺激運動の抽象基底クラス
  - `MovingMode_locking`：刺激を現在または遅延後の視線／マウス位置へ移動
- `Target`：視覚刺激の抽象基底クラス
  - `Target_image`：画像刺激の表示と位置設定
- `ControllerBase`：入力コントローラーの抽象基底クラス
  - `TobiiController`：Tobii 入力、記録、任意の平滑化を管理
  - `MouseController`：マウス位置を入力信号として使用

### Experimental Design Class

- `DesignExp`：制約付き遅延条件系列を生成し、各条件へ画像を割り当てるクラス

### Data Collection Class

- `DataManagerExp`：出力パスの作成、実行時設定の保存、2つの CSV への書き込み、データファイルの終了処理を行うクラス
