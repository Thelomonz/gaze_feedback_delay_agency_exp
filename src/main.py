"""Run the experiment with mouse or Tobii gaze input."""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
import importlib
import json
import os
import sys
import threading
import time
import numpy as np
from psychopy import core, event, visual
import GazeStabilizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOBII_CONTROLLER_DIR = os.path.join(BASE_DIR, "psychopy_tobii_controller")
if TOBII_CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, TOBII_CONTROLLER_DIR)

psychopy_tobii_controller = importlib.import_module("psychopy_tobii_controller")
print(f"Using psychopy_tobii_controller from: {psychopy_tobii_controller.__file__}")


IMAGE_DIR = os.path.join(BASE_DIR, "images")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def discover_images():
    return [
        os.path.join(IMAGE_DIR, fname)
        for fname in sorted(os.listdir(IMAGE_DIR))
        if fname.lower().endswith(IMAGE_EXTENSIONS)
    ]


config = {
    "design": {
        "blank_duration": 0,  # 試行間の空白時間（秒）
        "movement_duration": 10,  # 刺激の移動時間（秒）
        "trials_per_condition": 6,  # 各遅延条件の試行数
        "second_half_blank_duration": 2,  # 後半試行の空白時間（秒）
        "second_half_movement_duration": 10,  # 後半試行の刺激提示時間（秒）
        "control_rate_initial_percent": 50,  # 制御率の初期値
        "control_rate_min_percent": 0,  # 制御率の最小値（%）
        "control_rate_max_percent": 100,  # 制御率の最大値（%）
        "control_rate_step_percent": 1,  # キー入力1回あたりの制御率変更量（%）
        "control_rate_repeat_interval": 0.11,  # 矢印キー長押し時の連続入力間隔（秒）
        "locking_delays": {
            "locking_a": 0.0,  # 条件Aの遅延時間（秒）
            "locking_b": 0.5,  # 条件Bの遅延時間（秒）
            "locking_c": 2.0,  # 条件Cの遅延時間（秒）
        },
    },
    "experiment": {
        "fullscreen": True,
        "show_pos_indicator": False,  # 視点位置の表示
        "screen_margin": 0.02,  # 画面端から刺激までの最小余白（height単位）
        "background_color": (100, 100, 100),  # ウィンドウの背景色（RGB: 0～255）
        "calibration_intro_duration": 2,  # キャリブレーション案内の表示時間（秒）
        "experiment_start_delay": 2,  # 実験開始前の待機時間（秒）
        "pos_indicator_radius": 0.005,  # 位置インジケーターの半径（height単位）
        "pos_indicator_color": "red",  # 位置インジケーターの色
        "pos_indicator_opacity": 0.7,  # 位置インジケーターの不透明度（0～1）
    },
    "controller": {
        "type": "mouse",  # 入力方式（'mouse' or 'tobii'）
    },
    "tobii": {
        "calibration": True, 
        "calibration_points": 5,  # キャリブレーション点数（5 or 9）
        "calibration_range_x": 0.6,  # 水平方向範囲
        "calibration_range_y": 0.4,  # 垂直方向範囲
        "stabilizer_type": "moving_average",  # 平滑化（None or 'moving_average'）
        "stabilizer_moving_average": {
            "buffer_size": 50,  # 移動平均に使用するサンプル数
            "sampling_rate": 500,  # バックグラウンドでの視線取得周波数（Hz）
        },
    },
    "stimulus": {
        "target_type": "image",
        "scale": 0.05,  # 刺激画像の大きさ（height）
        "flash": False,  # 刺激の点滅
        "images": discover_images(),
    },
    "runtime": {
        "condition_to_image": {},
    },
}


def create_data_directory():
    data_folder = os.path.abspath(os.path.join(BASE_DIR, "../data"))
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Data directory created: {data_folder}")
    else:
        print(f"Data directory already exists: {data_folder}")
    return data_folder


def get_images():
    return config["stimulus"]["images"]


def bring_back_to_screen(pos, horizontal_limit, vertical_limit):
    if pos[0] < -horizontal_limit:
        pos[0] = -horizontal_limit
    elif pos[0] > horizontal_limit:
        pos[0] = horizontal_limit

    if pos[1] < -vertical_limit:
        pos[1] = -vertical_limit
    elif pos[1] > vertical_limit:
        pos[1] = vertical_limit

    return pos


def constrain_to_screen(pos, win):
    horizontal_limit = 0.5 * win.aspect - config["experiment"]["screen_margin"]
    vertical_limit = 0.5 - config["experiment"]["screen_margin"]
    return bring_back_to_screen(
        np.array(pos, dtype=np.float64), horizontal_limit, vertical_limit
    )


def run_gaze_position_preview(tobii_controller):
    win = tobii_controller.win
    status_bg = visual.Rect(
        win,
        size=(0.9, 0.65),
        pos=(0, 0),
        lineColor="white",
        fillColor="black",
        units="height",
        autoLog=False,
    )
    message = visual.TextStim(
        win,
        text="",
        pos=(0, -0.4),
        color="white",
        height=0.025,
        units="height",
        autoLog=False,
    )
    left_marker = visual.Circle(
        win,
        radius=0.018,
        fillColor=None,
        lineColor="green",
        lineWidth=4,
        units="height",
        autoLog=False,
    )
    right_marker = visual.Circle(
        win,
        radius=0.011,
        fillColor="red",
        lineColor="red",
        lineWidth=2,
        units="height",
        autoLog=False,
    )

    tobii_controller.gaze_data = []
    tobii_controller.eyetracker.subscribe_to(
        psychopy_tobii_controller.tobii_research.EYETRACKER_GAZE_DATA,
        tobii_controller.on_gaze_data,
    )
    try:
        while True:
            gaze_pos = np.array(
                tobii_controller.get_current_gaze_position(), dtype=np.float64
            )
            left_pos = gaze_pos[0:2]
            right_pos = gaze_pos[2:4]
            left_valid = np.all(np.isfinite(left_pos))
            right_valid = np.all(np.isfinite(right_pos))

            status_bg.draw()
            if left_valid:
                left_marker.setPos(left_pos)
                left_marker.draw()
            if right_valid:
                right_marker.setPos(right_pos)
                right_marker.draw()

            message.setText(
                "Tobii gaze preview: green=left, red=right. Press space to continue.\n"
                f"left={left_pos}, valid={left_valid}\n"
                f"right={right_pos}, valid={right_valid}"
            )
            message.draw()
            win.flip()

            keys = event.getKeys(keyList=["space", "escape"])
            if "escape" in keys:
                return "abort"
            if "space" in keys:
                return "success"
    finally:
        tobii_controller.eyetracker.unsubscribe_from(
            psychopy_tobii_controller.tobii_research.EYETRACKER_GAZE_DATA,
        )
        tobii_controller.gaze_data = []


def run_tobii_calibration(tobii_controller, num_points=5):
    event.clearEvents(eventType="keyboard")
    preview_result = run_gaze_position_preview(tobii_controller)
    if preview_result == "abort":
        return "abort"
    event.clearEvents(eventType="keyboard")

    x = config["tobii"]["calibration_range_x"]
    y = config["tobii"]["calibration_range_y"]

    num_points = int(num_points)
    if num_points == 5:
        calibration_points = [(-x, y), (x, y), (0, 0), (-x, -y), (x, -y)]
    elif num_points == 9:
        calibration_points = [
            (-x, y),
            (0, y),
            (x, y),
            (-x, 0),
            (0, 0),
            (x, 0),
            (-x, -y),
            (0, -y),
            (x, -y),
        ]
    else:
        raise ValueError("Invalid number of calibration points. Use '5' or '9'.")

    ret = tobii_controller.run_calibration(calibration_points)

    if ret == "abort":
        print("Calibration aborted.")
        return "abort"

    print("Calibration completed successfully.")
    return "success"


def process_gaze_position(current_gaze_position, last_gaze_position):
    num_of_nan_in_gaze = np.count_nonzero(np.isnan(current_gaze_position))
    no_eye_data = False

    if num_of_nan_in_gaze == 0:
        current_gaze_position[0] = (
            current_gaze_position[0] + current_gaze_position[2]
        ) / 2
        current_gaze_position[1] = (
            current_gaze_position[1] + current_gaze_position[3]
        ) / 2
        eye_x_loc = current_gaze_position[0]
        eye_y_loc = current_gaze_position[1]

    elif num_of_nan_in_gaze == 4:
        no_eye_data = True
        eye_x_loc = last_gaze_position[0]
        eye_y_loc = last_gaze_position[1]

    elif num_of_nan_in_gaze == 2:
        if not np.isnan(current_gaze_position[0]) and not np.isnan(
            current_gaze_position[1]
        ):
            eye_x_loc = current_gaze_position[0]
            eye_y_loc = current_gaze_position[1]
        else:
            eye_x_loc = current_gaze_position[2]
            eye_y_loc = current_gaze_position[3]

    else:
        no_eye_data = True
        eye_x_loc = last_gaze_position[0]
        eye_y_loc = last_gaze_position[1]

    current_gaze_position = np.array((eye_x_loc, eye_y_loc), dtype=np.float64)
    return current_gaze_position, no_eye_data


class DesignExp:
    def __init__(self, locking_delays, trials_per_condition=6):
        self.locking_delays = locking_delays
        self.trials_per_condition = trials_per_condition
        self.trial_sequence = None
        self.condition_to_image = None

    def generate_design(self):
        condition_names = list(self.locking_delays.keys())
        counts = {
            condition_name: self.trials_per_condition
            for condition_name in condition_names
        }

        sequence = self._build_constrained_sequence(counts)
        self.trial_sequence = [
            {
                "condition_type": "locking",
                "condition_name": condition_name,
                "delay_seconds": self.locking_delays[condition_name],
            }
            for condition_name in sequence
        ]
        config["runtime"]["trial_sequence"] = self.trial_sequence
        return self.trial_sequence

    def _build_constrained_sequence(self, counts):
        for _ in range(1000):
            remaining = counts.copy()
            sequence = []
            last_condition = None

            while sum(remaining.values()) > 0:
                candidates = [
                    condition_name
                    for condition_name, count in remaining.items()
                    if count > 0 and condition_name != last_condition
                ]
                if not candidates:
                    break

                weights = np.array(
                    [remaining[name] for name in candidates], dtype=np.float64
                )
                weights = weights / weights.sum()
                next_condition = np.random.choice(candidates, p=weights)

                sequence.append(next_condition)
                remaining[next_condition] -= 1
                last_condition = next_condition

            if len(sequence) == sum(counts.values()):
                return sequence

        sequence = []
        for condition_name, count in counts.items():
            sequence.extend([condition_name] * count)
        np.random.shuffle(sequence)
        return sequence

    def assign_condition_to_image(self):
        condition_names = list(self.locking_delays.keys())
        image_files = config["stimulus"]["images"]

        if len(image_files) < len(condition_names):
            raise ValueError(
                f"Experiment needs at least {len(condition_names)} image files, "
                f"but only {len(image_files)} were found."
            )

        image_indexes = np.random.choice(
            range(len(image_files)), size=len(condition_names), replace=False
        )
        self.condition_to_image = {
            condition_name: image_files[image_indexes[i]]
            for i, condition_name in enumerate(condition_names)
        }
        config["runtime"]["condition_to_image"] = self.condition_to_image
        return self.condition_to_image


class MovingMode(ABC):
    def __init__(self):
        self.pos = np.array((0.0, 0.0), dtype=np.float64)

    @abstractmethod
    def update(self):
        pass

    def reset(self, pos):
        self.pos = np.array(pos, dtype=np.float64)


class MovingMode_locking(MovingMode):
    def __init__(self, win):
        super().__init__()
        self.win = win

    def update(self, position=None):
        if position is None:
            return
        self.pos = constrain_to_screen(position, self.win)

    def reset(self, pos=None):
        if pos is not None:
            super().reset(pos)


class Target(ABC):
    @abstractmethod
    def set_stim(self, stim):
        pass

    @abstractmethod
    def set_pos(self, pos):
        pass

    @abstractmethod
    def draw(self):
        pass


class Target_image(Target):
    def __init__(self, win, images=None, scale=0.1):
        if images is None:
            self.images = get_images()
        else:
            self.images = images
        self.win = win
        self.scale = scale
        self.image = visual.ImageStim(
            win, image=self.images[0] if self.images else None, size=scale
        )

    def set_stim(self, image_path):
        self.image.setImage(image_path)
        self.image.setSize(self.scale)

    def set_pos(self, pos):
        self.image.setPos(pos)

    def draw(self):
        self.image.draw()


class ControllerBase(ABC):
    @abstractmethod
    def get_pos(self):
        pass

    @abstractmethod
    def record_event(self, event):
        pass


class TobiiController(ControllerBase):
    def __init__(self, win, stabilizer_type=None):
        self.tobii_controller = psychopy_tobii_controller.tobii_controller(win=win)
        self.last_pos = np.array([0, 0], dtype=np.float64)
        self.isNoData = False
        self.stabilizer_type = stabilizer_type
        self.win = win

        if stabilizer_type == "moving_average":
            self._bg_on = threading.Event()
            self.bg_sampling_rate = config["tobii"]["stabilizer_moving_average"][
                "sampling_rate"
            ]
            self.bg_buffer_size = config["tobii"]["stabilizer_moving_average"][
                "buffer_size"
            ]
            self.stabilizer = GazeStabilizer.MovingAverageStabilizer(
                self.bg_buffer_size
            )
            self._bg_thread = threading.Thread(
                target=self._run_background_sampling, daemon=True
            )
        else:
            self.stabilizer = None
            self._bg_thread = None
            self.bg_sampling_rate = None
            self.bg_buffer_size = None

    def subscribe(self, data_path_tobii=None):
        self.tobii_controller.open_datafile(data_path_tobii, embed_events=False)
        self.tobii_controller.subscribe()
        if self._bg_thread:
            self._bg_on.set()
            self._bg_thread.start()

    def unsubscribe(self):
        self.tobii_controller.unsubscribe()
        self.tobii_controller.close_datafile()
        if self._bg_thread:
            self._bg_on.clear()
            self._bg_thread.join()

    def _run_background_sampling(self):
        period = 1.0 / self.bg_sampling_rate
        while self._bg_on.is_set():
            pos = self.tobii_controller.get_current_gaze_position()
            pos = np.array(pos, dtype=np.float64)
            pos, no_eye_data = process_gaze_position(pos, self.last_pos)
            self.isNoData = no_eye_data
            self.last_pos = pos

            if self.stabilizer:
                self.stabilizer.stabilize(pos[0], pos[1])

            time.sleep(period)

    def get_pos(self):
        pos = self.tobii_controller.get_current_gaze_position()
        pos = np.array(pos, dtype=np.float64)
        pos, no_eye_data = process_gaze_position(pos, self.last_pos)
        self.isNoData = no_eye_data
        self.last_pos = pos

        if self.stabilizer:
            pos = self.stabilizer.stabilize(pos[0], pos[1])
            pos = np.array(pos, dtype=np.float64)
        return pos

    def record_event(self, event):
        self.tobii_controller.record_event(event)
        print(f"Event recorded: {event}")


class MouseController(ControllerBase):
    def __init__(self, win=None):
        self.mouse = event.Mouse(win=win)
        self.isNoData = False

    def get_pos(self):
        return np.array(self.mouse.getPos(), dtype=np.float64)

    def record_event(self, event):
        print(f"Event recorded: {event}")


class DataManagerExp:
    def __init__(self, data_folder):
        self.data_path_exp = None
        self.data_path_second_half = None
        self.data_path_tobii = None
        self.data_path_config = None
        self.data_folder = data_folder
        self.date = datetime.today().strftime("%Y%m%d%H%M")
        self.iDataEntry = 0
        self.iSecondHalfDataEntry = 0
        self.file = None
        self.second_half_file = None

    def enter_subj_id(self):
        subject_id = input("Enter subject ID: ")
        if not subject_id:
            raise ValueError("Subject ID cannot be empty.")
        self.data_path_exp = os.path.join(
            self.data_folder, f"{subject_id}_exp_{self.date}.csv"
        )
        self.data_path_second_half = os.path.join(
            self.data_folder, f"{subject_id}_exp_second_half_{self.date}.csv"
        )
        self.data_path_tobii = os.path.join(
            self.data_folder, f"{subject_id}_tobii_exp_{self.date}.tsv"
        )
        self.data_path_config = os.path.join(
            self.data_folder, f"{subject_id}_config_exp_{self.date}.json"
        )
        print(
            f"Data paths set: {self.data_path_exp}, {self.data_path_second_half}, "
            f"{self.data_path_tobii}, {self.data_path_config}"
        )

    def save_config(self, config_to_save):
        try:
            with open(self.data_path_config, "w") as f:
                json.dump(config_to_save, f, indent=4)
            print(f"Configuration saved to: {self.data_path_config}")
        except Exception as exc:
            print(f"Error saving configuration: {exc}")

    @staticmethod
    def _serialize_row(frame_data):
        values = []
        for value in frame_data.values():
            if value is None:
                values.append("None")
            elif isinstance(value, str):
                values.append(f'"{value}"')
            else:
                values.append(str(value))
        return ",".join(values)

    def log_data(self, frame_data):
        if self.file is None:
            try:
                self.file = open(self.data_path_exp, "w")
                print(f"Opened data file for writing: {self.data_path_exp}")
            except Exception as exc:
                print(f"Error opening data file: {exc}")
                return

        if self.iDataEntry == 0:
            header = ",".join(frame_data.keys())
            self.file.write(header + "\n")

        row = self._serialize_row(frame_data)
        self.file.write(row + "\n")
        self.file.flush()
        self.iDataEntry += 1

    def log_second_half_data(self, frame_data):
        if self.second_half_file is None:
            try:
                self.second_half_file = open(self.data_path_second_half, "w")
                print(
                    f"Opened second half data file for writing: {self.data_path_second_half}"
                )
            except Exception as exc:
                print(f"Error opening second half data file: {exc}")
                return

        if self.iSecondHalfDataEntry == 0:
            header = ",".join(frame_data.keys())
            self.second_half_file.write(header + "\n")

        row = self._serialize_row(frame_data)
        self.second_half_file.write(row + "\n")
        self.second_half_file.flush()
        self.iSecondHalfDataEntry += 1

    def close_file(self):
        if self.file is not None:
            self.file.close()
            print(f"Data file closed: {self.data_path_exp}")
            self.file = None
        if self.second_half_file is not None:
            self.second_half_file.close()
            print(f"Second half data file closed: {self.data_path_second_half}")
            self.second_half_file = None


def append_gaze_history(gaze_history, timestamp, controller_pos, no_eye_data):
    gaze_history.append(
        {
            "timestamp": timestamp,
            "eye_x": controller_pos[0],
            "eye_y": controller_pos[1],
            "no_eye_data": no_eye_data,
        }
    )


def get_delayed_gaze_position(gaze_history, current_time, delay_seconds):
    if not gaze_history:
        return np.array([0.0, 0.0], dtype=np.float64)

    if delay_seconds <= 0:
        sample = gaze_history[-1]
        return np.array([sample["eye_x"], sample["eye_y"]], dtype=np.float64)

    target_time = current_time - delay_seconds
    closest_sample = min(
        gaze_history,
        key=lambda sample: abs(sample["timestamp"] - target_time),
    )
    return np.array(
        [closest_sample["eye_x"], closest_sample["eye_y"]], dtype=np.float64
    )


def make_frame_data(
    controller_pos,
    delayed_pos,
    stim_pos,
    no_eye_data,
    trial_num,
    frame_num,
    current_time,
    start_time,
    condition_name,
    delay_seconds,
    phase,
    image_file,
    response_time=None,
    control_rate=None,
    control_rate_percent=None,
):
    return {
        "eye_x": controller_pos[0],
        "eye_y": controller_pos[1],
        "delayed_eye_x": delayed_pos[0] if delayed_pos is not None else None,
        "delayed_eye_y": delayed_pos[1] if delayed_pos is not None else None,
        "stim_x": stim_pos[0] if stim_pos is not None else None,
        "stim_y": stim_pos[1] if stim_pos is not None else None,
        "no_eye_data": no_eye_data,
        "trial_num": trial_num,
        "frame_num": frame_num,
        "time": current_time,
        "time_trial": current_time - start_time,
        "condition_type": "locking",
        "condition_name": condition_name,
        "delay_seconds": delay_seconds,
        "phase": phase,
        "image_file": image_file,
        "response_time": response_time,
        "control_rate": control_rate,
        "control_rate_percent": control_rate_percent,
    }


def make_control_rate_data(
    trial_num,
    condition_name,
    delay_seconds,
    image_file,
    response_time,
    control_rate_percent,
):
    current_time = core.getTime()
    control_rate = round(control_rate_percent / 100.0, 4)
    return {
        "eye_x": None,
        "eye_y": None,
        "delayed_eye_x": None,
        "delayed_eye_y": None,
        "stim_x": None,
        "stim_y": None,
        "no_eye_data": None,
        "trial_num": trial_num,
        "frame_num": None,
        "time": current_time,
        "time_trial": response_time,
        "condition_type": "locking",
        "condition_name": condition_name,
        "delay_seconds": delay_seconds,
        "phase": "control_rating",
        "image_file": image_file,
        "response_time": response_time,
        "control_rate": f"{control_rate:.4f}",
        "control_rate_percent": control_rate_percent,
    }


def cleanup_and_quit(win, controller, data_manager, controller_type):
    data_manager.close_file()
    if controller_type == "tobii":
        controller.unsubscribe()
    win.close()
    core.quit()


def collect_controller_sample(controller, controller_type):
    controller_pos = controller.get_pos()
    no_eye_data = controller.isNoData if controller_type == "tobii" else False
    return controller_pos, no_eye_data


def run_effective_phase(
    win,
    controller,
    controller_type,
    data_manager,
    gaze_history,
    target,
    locking_mode,
    pos_indicator,
    trial_num,
    frame_num,
    condition_name,
    delay_seconds,
    image_file,
    phase,
    duration,
    draw_target,
    data_logger=None,
):
    if data_logger is None:
        data_logger = data_manager.log_data

    start_time = core.getTime()
    last_time = start_time
    effective_time = 0.0
    current_pos = None

    while effective_time < duration:
        current_time = core.getTime()
        dt = current_time - last_time
        last_time = current_time

        controller_pos, no_eye_data = collect_controller_sample(
            controller, controller_type
        )
        append_gaze_history(gaze_history, current_time, controller_pos, no_eye_data)

        if not no_eye_data:
            effective_time += dt

        delayed_pos = get_delayed_gaze_position(
            gaze_history, current_time, delay_seconds
        )
        if draw_target:
            locking_mode.update(delayed_pos)
            current_pos = locking_mode.pos
            target.set_pos(current_pos)
            target.draw()

        if pos_indicator is not None and draw_target:
            pos_indicator.setPos(controller_pos)
            pos_indicator.draw()

        keys = event.getKeys()
        if "escape" in keys:
            controller.record_event("Experiment interrupted by user")
            cleanup_and_quit(win, controller, data_manager, controller_type)
            return frame_num, current_pos, False

        frame_data = make_frame_data(
            controller_pos=controller_pos,
            delayed_pos=delayed_pos,
            stim_pos=current_pos,
            no_eye_data=no_eye_data,
            trial_num=trial_num,
            frame_num=frame_num,
            current_time=current_time,
            start_time=start_time,
            condition_name=condition_name,
            delay_seconds=delay_seconds,
            phase=phase,
            image_file=image_file if draw_target else None,
        )
        data_logger(frame_data)

        frame_num += 1
        win.flip()

    return frame_num, current_pos, True


def get_control_rate_response(
    win, initial_percent=None, step_percent=None, confirm_keys=("space", "return")
):
    if initial_percent is None:
        initial_percent = config["design"]["control_rate_initial_percent"]
    if step_percent is None:
        step_percent = config["design"]["control_rate_step_percent"]

    min_percent = config["design"]["control_rate_min_percent"]
    max_percent = config["design"]["control_rate_max_percent"]
    if min_percent >= max_percent:
        raise ValueError(
            "control_rate_min_percent must be less than control_rate_max_percent."
        )

    control_rate_percent = float(initial_percent)
    slider_y = -0.05
    slider_left = -0.35
    slider_right = 0.35

    question = visual.TextStim(
        win,
        text="制御率を選択してください。\n(左右キーで調整、スペースキーで確定)",
        pos=(0, 0.18),
        color="white",
        height=0.05,
    )
    left_label = visual.TextStim(
        win,
        text=f"{min_percent}%",
        pos=(slider_left, slider_y - 0.08),
        color="white",
        height=0.035,
    )
    right_label = visual.TextStim(
        win,
        text=f"{max_percent}%",
        pos=(slider_right, slider_y - 0.08),
        color="white",
        height=0.035,
    )
    value_label = visual.TextStim(
        win,
        text="",
        pos=(0, slider_y + 0.12),
        color="white",
        height=0.045,
    )
    slider_line = visual.Line(
        win,
        start=(slider_left, slider_y),
        end=(slider_right, slider_y),
        lineColor="white",
        lineWidth=4,
    )
    slider_arrow = visual.ShapeStim(
        win,
        vertices=[(-0.018, 0.025), (0.018, 0.025), (0, -0.02)],
        fillColor="blue",
        lineColor="blue",
        closeShape=True,
    )

    key_state = None
    pyglet_key = None
    try:
        from pyglet.window import key as pyglet_key

        key_state = pyglet_key.KeyStateHandler()
        win.winHandle.push_handlers(key_state)
    except Exception:
        key_state = None

    event.clearEvents(eventType="keyboard")
    response_start = core.getTime()
    next_adjust_time = response_start
    repeat_interval = config["design"]["control_rate_repeat_interval"]

    while True:
        control_rate_percent = min(max_percent, max(min_percent, control_rate_percent))
        arrow_x = slider_left + (slider_right - slider_left) * (
            (control_rate_percent - min_percent) / (max_percent - min_percent)
        )
        slider_arrow.setPos((arrow_x, slider_y + 0.045))
        value_label.setText(f"{control_rate_percent:.0f}%")

        question.draw()
        value_label.draw()
        slider_line.draw()
        left_label.draw()
        right_label.draw()
        slider_arrow.draw()
        win.flip()

        keys = event.getKeys(keyList=["left", "right", "escape"] + list(confirm_keys))
        if "escape" in keys:
            return None, None, None, False
        if key_state is not None:
            now = core.getTime()
            if now >= next_adjust_time:
                if key_state[pyglet_key.LEFT]:
                    control_rate_percent -= step_percent
                    next_adjust_time = now + repeat_interval
                elif key_state[pyglet_key.RIGHT]:
                    control_rate_percent += step_percent
                    next_adjust_time = now + repeat_interval
                else:
                    next_adjust_time = now
        else:
            if "left" in keys:
                control_rate_percent -= step_percent
            if "right" in keys:
                control_rate_percent += step_percent
        if any(key in keys for key in confirm_keys):
            response_time = core.getTime() - response_start
            control_rate = round(control_rate_percent / 100.0, 4)
            return control_rate, control_rate_percent, response_time, True


def run_control_rating_trial(
    win,
    controller,
    controller_type,
    data_manager,
    gaze_history,
    target,
    locking_mode,
    pos_indicator,
    trial_num,
    condition,
    image_file,
    current_pos,
):
    condition_name = condition["condition_name"]
    delay_seconds = condition["delay_seconds"]
    frame_num = 0

    controller.record_event(
        f"Second half No.{trial_num} {condition_name} trial started"
    )

    current_pos = constrain_to_screen(current_pos, win)
    locking_mode.reset(current_pos)

    frame_num, blank_pos, should_continue = run_effective_phase(
        win=win,
        controller=controller,
        controller_type=controller_type,
        data_manager=data_manager,
        gaze_history=gaze_history,
        target=target,
        locking_mode=locking_mode,
        pos_indicator=pos_indicator,
        trial_num=trial_num,
        frame_num=frame_num,
        condition_name=condition_name,
        delay_seconds=delay_seconds,
        image_file=None,
        phase="second_half_blank",
        duration=config["design"]["second_half_blank_duration"],
        draw_target=False,
        data_logger=data_manager.log_second_half_data,
    )
    if not should_continue:
        return current_pos, False

    target.set_stim(image_file)
    movement_start_time = core.getTime()
    initial_delayed_pos = get_delayed_gaze_position(
        gaze_history, movement_start_time, delay_seconds
    )
    locking_mode.reset(initial_delayed_pos)

    frame_num, current_pos, should_continue = run_effective_phase(
        win=win,
        controller=controller,
        controller_type=controller_type,
        data_manager=data_manager,
        gaze_history=gaze_history,
        target=target,
        locking_mode=locking_mode,
        pos_indicator=pos_indicator,
        trial_num=trial_num,
        frame_num=frame_num,
        condition_name=condition_name,
        delay_seconds=delay_seconds,
        image_file=image_file,
        phase="second_half_movement",
        duration=config["design"]["second_half_movement_duration"],
        draw_target=True,
        data_logger=data_manager.log_second_half_data,
    )
    if not should_continue:
        return current_pos, False

    if current_pos is None:
        current_pos = (
            blank_pos
            if blank_pos is not None
            else np.array([0.0, 0.0], dtype=np.float64)
        )

    controller.record_event(
        f"Second half No.{trial_num} {condition_name} rating started"
    )
    _, control_rate_percent, response_time, should_continue = get_control_rate_response(
        win
    )
    if not should_continue:
        controller.record_event("Second half interrupted by user")
        cleanup_and_quit(win, controller, data_manager, controller_type)
        return current_pos, False

    data_manager.log_second_half_data(
        make_control_rate_data(
            trial_num=trial_num,
            condition_name=condition_name,
            delay_seconds=delay_seconds,
            image_file=image_file,
            response_time=response_time,
            control_rate_percent=control_rate_percent,
        )
    )

    controller.record_event(
        f"Second half No.{trial_num} {condition_name} trial ended "
        f"control_rate={control_rate_percent / 100.0:.4f}"
    )
    return current_pos, True


def run_second_half_experiment(
    win,
    controller,
    controller_type,
    data_manager,
    gaze_history,
    target,
    locking_mode,
    pos_indicator,
    design,
    current_pos,
):
    controller.record_event("Second half started")

    instructions = visual.TextStim(
        win,
        text="これから各動物が1回ずつ提示されます。\n提示が終わったら、制御率を報告してください。\n\n準備ができましたら、スペースキーを押してください。",
        pos=(0, 0),
        color="white",
        height=0.05,
    )
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    second_half_trials = [
        {
            "condition_type": "locking",
            "condition_name": condition_name,
            "delay_seconds": config["design"]["locking_delays"][condition_name],
        }
        for condition_name in config["design"]["locking_delays"]
    ]

    for iTrial, condition in enumerate(second_half_trials):
        image_file = design.condition_to_image[condition["condition_name"]]
        current_pos, should_continue = run_control_rating_trial(
            win=win,
            controller=controller,
            controller_type=controller_type,
            data_manager=data_manager,
            gaze_history=gaze_history,
            target=target,
            locking_mode=locking_mode,
            pos_indicator=pos_indicator,
            trial_num=iTrial,
            condition=condition,
            image_file=image_file,
            current_pos=current_pos,
        )
        if not should_continue:
            return current_pos, False

    controller.record_event("Second half completed")
    return current_pos, True


def run_exp(controller_type="tobii"):
    target_type = config["stimulus"]["target_type"]
    if target_type != "image":
        raise ValueError("Experiment expects target_type='image'.")

    data_folder = create_data_directory()

    design = DesignExp(
        locking_delays=config["design"]["locking_delays"],
        trials_per_condition=config["design"]["trials_per_condition"],
    )
    trial_sequence = design.generate_design()
    design.assign_condition_to_image()

    data_manager = DataManagerExp(data_folder)
    data_manager.enter_subj_id()
    data_manager.save_config(config)

    win = visual.Window(
        units="height",
        monitor="default",
        fullscr=config["experiment"]["fullscreen"],
        colorSpace="rgb255",
        color=config["experiment"]["background_color"],
    )

    if config["experiment"]["fullscreen"]:
        win.mouseVisible = False
        event.Mouse(visible=False)

    target = Target_image(win, scale=config["stimulus"]["scale"])

    if controller_type == "tobii":
        controller = TobiiController(
            win=win, stabilizer_type=config["tobii"]["stabilizer_type"]
        )
    else:
        controller = MouseController(win=win)

    pos_indicator = None
    if config["experiment"]["show_pos_indicator"]:
        pos_indicator = visual.Circle(
            win,
            radius=config["experiment"]["pos_indicator_radius"],
            fillColor=config["experiment"]["pos_indicator_color"],
            lineColor=config["experiment"]["pos_indicator_color"],
            opacity=config["experiment"]["pos_indicator_opacity"],
            autoLog=False,
        )

    instructions = visual.TextStim(
        win,
        text="Press 'escape' to exit",
        pos=(0, 0),
        color="white",
        height=0.05,
    )

    locking_mode = MovingMode_locking(win)

    if controller_type == "tobii" and config["tobii"]["calibration"]:
        calib_msg = visual.TextStim(
            win,
            text="Eye tracker calibration will start now.\nFollow the dots with your eyes.",
            pos=(0, 0),
            color="white",
            height=0.05,
        )
        calib_msg.draw()
        win.flip()
        core.wait(config["experiment"]["calibration_intro_duration"])

        calib_result = run_tobii_calibration(
            controller.tobii_controller,
            config["tobii"]["calibration_points"],
        )

        if calib_result == "abort":
            win.close()
            core.quit()
            return

    if controller_type == "tobii":
        controller.subscribe(data_manager.data_path_tobii)

    controller.record_event("Experiment started")

    instructions.setText(
        "実験にご参加いただき、ありがとうございます!\n\n準備ができましたら、スペースキーを押してください。"
    )
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    win.flip()
    core.wait(config["experiment"]["experiment_start_delay"])

    gaze_history = deque()
    current_pos = np.array([0.0, 0.0], dtype=np.float64)

    for iTrial, trial in enumerate(trial_sequence):
        condition_name = trial["condition_name"]
        delay_seconds = trial["delay_seconds"]
        image_file = design.condition_to_image[condition_name]
        frame_num = 0

        controller.record_event(f"Exp No.{iTrial} {condition_name} trial started")

        current_pos = constrain_to_screen(current_pos, win)
        locking_mode.reset(current_pos)

        blank_pos = None
        if config["design"]["blank_duration"] > 0:
            frame_num, blank_pos, should_continue = run_effective_phase(
                win=win,
                controller=controller,
                controller_type=controller_type,
                data_manager=data_manager,
                gaze_history=gaze_history,
                target=target,
                locking_mode=locking_mode,
                pos_indicator=pos_indicator,
                trial_num=iTrial,
                frame_num=frame_num,
                condition_name=condition_name,
                delay_seconds=delay_seconds,
                image_file=None,
                phase="black",
                duration=config["design"]["blank_duration"],
                draw_target=False,
            )
            if not should_continue:
                return

        target.set_stim(image_file)

        movement_start_time = core.getTime()
        initial_delayed_pos = get_delayed_gaze_position(
            gaze_history, movement_start_time, delay_seconds
        )
        locking_mode.reset(initial_delayed_pos)

        frame_num, current_pos, should_continue = run_effective_phase(
            win=win,
            controller=controller,
            controller_type=controller_type,
            data_manager=data_manager,
            gaze_history=gaze_history,
            target=target,
            locking_mode=locking_mode,
            pos_indicator=pos_indicator,
            trial_num=iTrial,
            frame_num=frame_num,
            condition_name=condition_name,
            delay_seconds=delay_seconds,
            image_file=image_file,
            phase="movement",
            duration=config["design"]["movement_duration"],
            draw_target=True,
        )
        if not should_continue:
            return

        if current_pos is None:
            current_pos = (
                blank_pos
                if blank_pos is not None
                else np.array([0.0, 0.0], dtype=np.float64)
            )

        controller.record_event(f"Exp No.{iTrial} {condition_name} trial ended")

    current_pos, should_continue = run_second_half_experiment(
        win=win,
        controller=controller,
        controller_type=controller_type,
        data_manager=data_manager,
        gaze_history=gaze_history,
        target=target,
        locking_mode=locking_mode,
        pos_indicator=pos_indicator,
        design=design,
        current_pos=current_pos,
    )
    if not should_continue:
        return

    controller.record_event("Experiment completed")

    instructions.setText(
        "実験が終了しました!\n終了するには「Esc」キーを押してください。"
    )
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=["escape"])

    cleanup_and_quit(win, controller, data_manager, controller_type)


if __name__ == "__main__":
    run_exp(controller_type=config["controller"]["type"])
