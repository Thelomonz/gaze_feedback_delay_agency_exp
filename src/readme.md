# 本プロジェクトは、以下のプロジェクトを基に改変したものです。  
https://github.com/acercyc/soa_eye_tracking_exp  
原作者：acercyc  
Modified for a sense of agency eye-tracking experiment.

# 視線追従・制御感評価実験

PsychoPy と Tobii アイトラッカーを使用し、視線に追従する画像刺激の遅延が主観的な制御感に与える影響を測定する実験プログラムです。Tobii を接続していない環境では、マウスを使って表示や操作を確認できます。

## 実験の概要

実験は前半と後半で構成されます。

### 前半：遅延条件の提示

視線またはマウス位置に追従する画像を、標準設定では次の 3 条件で提示します。

| 条件 | 遅延時間 |
|---|---:|
| `locking_a` | 0.0 秒 |
| `locking_b` | 0.5 秒 |
| `locking_c` | 2.0 秒 |

各条件には異なる画像がランダムに割り当てられます。同じ条件が連続しないように試行系列を生成し、標準設定では各条件を 6 回ずつ提示します。

### 後半：制御率の評価

各遅延条件を 1 回ずつ再提示した後、参加者が刺激をどの程度制御できたと感じたかを 0～100% で回答します。

- `←` / `→`：値を変更
- `Space` / `Enter`：回答を確定
- `Esc`：実験を中断または終了

## ファイル構成

```text
soa_eye_tracking_exp/
├── data/                              # 実験データの保存先（実行時に自動作成）
└── src/
    ├── main.py                        # 実験プログラム
    ├── GazeStabilizer.py              # 視線座標の平滑化処理
    ├── images/                        # 刺激画像
    ├── psychopy_tobii_controller/     # PsychoPy–Tobii 接続モジュール
    └── readme.md
```

## 動作環境

- Python 3
- PsychoPy
- NumPy
- Tobii Pro SDK (`tobii_research`、Tobii モードのみ)
- Tobii 対応アイトラッカー（Tobii モードのみ）
- scikit-learn（GMM ベースの平滑化機能を使用する場合のみ）

PsychoPy Standalone に付属する Python 環境の使用を推奨します。

## セットアップ

PowerShell で `src` ディレクトリへ移動します。

```powershell
cd C:\Users\phorc\Desktop\soa_eye_tracking_exp\src
```

必要なパッケージが環境にない場合はインストールします。

```powershell
python -m pip install numpy psychopy tobii-research
python -m pip install -e .\psychopy_tobii_controller
```

GMM ベースの平滑化機能も使用する場合は、次を追加します。

```powershell
python -m pip install scikit-learn
```

## 刺激画像

画像ファイルを `src/images` に配置してください。次の拡張子が自動的に読み込まれます。

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

各遅延条件に異なる画像を割り当てるため、画像数は条件数以上である必要があります。標準設定では少なくとも 3 枚必要です。

## 設定

実験条件は `main.py` 冒頭の `config` で変更します。

### 実験デザイン

| 設定項目 | 説明 |
|---|---|
| `design.blank_duration` | 前半の試行間に表示する空白時間（秒）。`0` で省略 |
| `design.movement_duration` | 前半の刺激提示時間（秒） |
| `design.trials_per_condition` | 前半における各遅延条件の試行数 |
| `design.second_half_blank_duration` | 後半の空白時間（秒） |
| `design.second_half_movement_duration` | 後半の刺激提示時間（秒） |
| `design.control_rate_initial_percent` | 制御率回答の初期値（設定した範囲内） |
| `design.control_rate_min_percent` | 制御率回答の最小値（%） |
| `design.control_rate_max_percent` | 制御率回答の最大値（%） |
| `design.control_rate_step_percent` | キー入力 1 回あたりの変更量 |
| `design.control_rate_repeat_interval` | 矢印キー長押し時の連続入力間隔（秒） |
| `design.locking_delays` | 条件名と遅延時間（秒）の対応 |

### 表示と入力

| 設定項目 | 説明 |
|---|---|
| `experiment.fullscreen` | フルスクリーン表示の有効・無効 |
| `experiment.show_pos_indicator` | 現在の視線／マウス位置を表示するか |
| `experiment.screen_margin` | 刺激と画面端の最小距離 |
| `experiment.background_color` | ウィンドウの背景色（RGB） |
| `experiment.calibration_intro_duration` | キャリブレーション案内の表示時間（秒） |
| `experiment.experiment_start_delay` | 実験開始前の待機時間（秒） |
| `experiment.pos_indicator_radius` | 位置インジケーターの半径 |
| `experiment.pos_indicator_color` | 位置インジケーターの色 |
| `experiment.pos_indicator_opacity` | 位置インジケーターの不透明度（0～1） |
| `controller.type` | 入力方式。`"mouse"` または `"tobii"` |
| `stimulus.scale` | 刺激画像の大きさ |

### Tobii

| 設定項目 | 説明 |
|---|---|
| `tobii.calibration` | 実験開始前にキャリブレーションを行うか |
| `tobii.calibration_points` | キャリブレーション点数。`5` または `9` |
| `tobii.calibration_range_x` | キャリブレーション点の水平方向範囲 |
| `tobii.calibration_range_y` | キャリブレーション点の垂直方向範囲 |
| `tobii.stabilizer_type` | 平滑化方式。`None` または `"moving_average"` |
| `tobii.stabilizer_moving_average.buffer_size` | 移動平均に使用するサンプル数 |
| `tobii.stabilizer_moving_average.sampling_rate` | バックグラウンド取得周波数（Hz） |

`stimulus.flash` は現在の locking モードでは使用されません。

## 実行方法

### マウスで動作確認する場合

`config` を次のように設定します。

```python
"controller": {
    "type": "mouse",
},
```

`src` ディレクトリで実行します。

```powershell
python main.py
```

### Tobii を使用する場合

入力方式を変更します。

```python
"controller": {
    "type": "tobii",
},
```

アイトラッカーを接続し、Tobii Pro SDK から認識できることを確認してから `main.py` を実行してください。`tobii.calibration` が `True` の場合は、視線位置の確認画面に続いてキャリブレーションが開始されます。

## 実行時の流れ

1. ターミナルで参加者 ID を入力します。空の ID は使用できません。
2. Tobii モードでは、必要に応じて視線確認とキャリブレーションを行います。
3. 説明画面で `Space` を押して前半を開始します。
4. 各遅延条件の刺激を観察します。
5. 後半で各条件を再度観察し、制御率を回答します。
6. 終了画面で `Esc` を押します。

## 出力データ

データはプロジェクト直下の `data` に保存されます。日時は `YYYYMMDDHHMM` 形式です。

| ファイル | 内容 |
|---|---|
| `{subject_id}_exp_{date}.csv` | 前半のフレーム単位データ |
| `{subject_id}_exp_second_half_{date}.csv` | 後半のフレームデータと制御率回答 |
| `{subject_id}_tobii_exp_{date}.tsv` | Tobii の生データ |
| `{subject_id}_config_exp_{date}.json` | 実行時設定、試行系列、画像割り当て |

マウスモードでは Tobii の TSV ファイルは作成されません。

CSV には主に次の項目が記録されます。

- 視線またはマウス座標：`eye_x`, `eye_y`
- 遅延適用後の座標：`delayed_eye_x`, `delayed_eye_y`
- 刺激座標：`stim_x`, `stim_y`
- 試行番号とフレーム番号：`trial_num`, `frame_num`
- 時刻：`time`, `time_trial`
- 条件名と遅延時間：`condition_name`, `delay_seconds`
- フェーズ：`phase`
- 視線欠損フラグ：`no_eye_data`
- 画像ファイル：`image_file`
- 制御率と反応時間：`control_rate`, `control_rate_percent`, `response_time`

## 注意事項

- 本番前にマウスモードで画像表示、キー操作、試行時間、データ保存を確認してください。
- Tobii モードでは、接続状態と設定したサンプリング周波数を事前に確認してください。
- 実験中はウィンドウを強制終了せず、可能な限り `Esc` で終了してください。強制終了するとデータファイルが正常に閉じられない場合があります。
- `psychopy_tobii_controller` は非公式の補助モジュールであり、Tobii 社の公式製品ではありません。
