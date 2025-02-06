import sys
import os
import time
import cv2
import numpy as np
import datetime

# ------------------------------
# .NET / Kinesis motion imports
# ------------------------------
import clr

kinesis_path = r"C:\Program Files\Thorlabs\Kinesis"
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.DCServoCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI, DeviceConfiguration, SimulationManager
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
from System import Decimal, Int32

# ------------------------------
# PyQt imports
# ------------------------------
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox, QGridLayout, QCheckBox, QLineEdit, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage

# ------------------------------
# pylablib / camera imports
# ------------------------------
import pylablib as pll
from pylablib.devices import Thorlabs

# ------------------------------
# ctypes imports
# ------------------------------
from ctypes import cdll, c_uint32, c_int, c_double, byref, create_string_buffer


###############################################################################
# MotionController: Manages 3 Axes with Kinesis using Absolute Moves
###############################################################################
class MotionController:
    def __init__(self):
        self.axis_info = {
            "X": "27601188",
            "Y": "27601181",
            "Z": "27265400",
        }
        self.devices = {}
        self._init_devices()

    def _init_devices(self):
        DeviceManagerCLI.BuildDeviceList()
        print("Kinesis devices found:", list(DeviceManagerCLI.GetDeviceList()))
        for axis, serial_no in self.axis_info.items():
            device = KCubeDCServo.CreateKCubeDCServo(serial_no)
            device.Connect(serial_no)
            time.sleep(0.25)
            device.StartPolling(250)
            time.sleep(0.25)
            device.EnableDevice()
            time.sleep(0.25)
            if not device.IsSettingsInitialized():
                device.WaitForSettingsInitialized(10000)
            if not device.IsSettingsInitialized():
                print(f"[{axis}] Settings failed to initialize!")
                continue
            m_config = device.LoadMotorConfiguration(serial_no,
                DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)
            m_config.UpdateCurrentConfiguration()
            device.SetSettings(device.MotorDeviceSettings, True, False)
            if device.Position == Decimal(0):
                print(f"Homing axis {axis} (serial {serial_no})...")
                device.Home(60000)
                print(f"[{axis}] Homing complete. Current pos:", device.Position)
            else:
                print(f"Already homed.  Current pos:", device.Position)
            self.devices[axis] = device

    def move_axis_relative(self, axis, step_mm):
        if axis not in self.devices:
            print(f"[{axis}] Axis not initialized.")
            return
        device = self.devices[axis]
        try:
            current_pos = float(str(device.Position))
        except Exception as e:
            print(f"[{axis}] Error reading current position: {e}")
            return
        new_pos = current_pos + step_mm
        print(f"[{axis}] Moving from {current_pos:.4f} mm to {new_pos:.4f} mm (requested {step_mm} mm)")
        try:
            device.MoveTo(Decimal(new_pos), Int32(20000))
        except Exception as e:
            print(f"[{axis}] MoveTo command failed: {e}")
        time.sleep(0.1)
        print(f"[{axis}] Move done. Current pos: {device.Position}")

    def disconnect_all(self):
        for axis, device in self.devices.items():
            if device is not None:
                print(f"Disconnecting axis {axis}...")
                device.StopPolling()
                device.DisableDevice()
                device.Disconnect()
        SimulationManager.Instance.UninitializeSimulations()


###############################################################################
# StageMoveWorker: Runs a stage move command in a separate thread.
###############################################################################
class StageMoveWorker(QThread):
    finished_move = pyqtSignal()

    def __init__(self, hardware, axis, step_mm, parent=None):
        super().__init__(parent)
        self.hardware = hardware  # MicroscopeHardware instance
        self.axis = axis
        self.step_mm = step_mm

    def run(self):
        self.hardware._move_axis_absolute(self.axis, self.step_mm)
        self.finished_move.emit()


###############################################################################
# AutoCaptureWorker: Runs an automatic capture sequence in a separate thread.
###############################################################################
class AutoCaptureWorker(QThread):
    finished_capture = pyqtSignal()
    # Emit: captures_completed, total_captures, estimated_time_remaining (in seconds)
    progress_update = pyqtSignal(int, int, float)

    def __init__(self, hardware, mode, overlap, num_stack, pattern_size_x, pattern_size_y, z_stack_height, parent=None):
        super().__init__(parent)
        self.hardware = hardware  # MicroscopeHardware instance
        self.mode = mode            # "Focus Stack", "XY Pattern", or "Combined"
        self.overlap = overlap
        self.num_stack = num_stack
        self.pattern_size_x = pattern_size_x
        self.pattern_size_y = pattern_size_y
        self.z_stack_height = z_stack_height

        # Define per-capture time estimates (in seconds) based on your sample timings.
        self.time_estimates = {
            "Focus Stack": 3.0,
            "XY Pattern": 7.0,
            "Combined": 4.7
        }

    def run(self):
        # Create a folder for this auto capture session.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        folder_name = os.path.join(downloads, f"AutoCapture_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)
        self.hardware.auto_capture_folder = folder_name

        # Pause continuous acquisition during auto capture.
        self.hardware.auto_capture_in_progress = True

        # Get current stage positions
        current_x = self._get_current_coordinate('X')
        current_y = self._get_current_coordinate('Y')
        current_z = self._get_current_coordinate('Z')
        print(f"AutoCapture starting at X:{current_x:.3f} Y:{current_y:.3f} Z:{current_z:.3f}")

        if self.mode == "Focus Stack":
            self._run_focus_stack(current_x, current_y, current_z)
        elif self.mode == "XY Pattern":
            self._run_xy_pattern(current_x, current_y, current_z)
        elif self.mode == "Combined":
            self._run_combined(current_x, current_y, current_z)

        # Restart live view.
        self.hardware.start_camera()
        self.hardware.auto_capture_in_progress = False
        self.finished_capture.emit()

    def _get_current_coordinate(self, axis):
        device = self.hardware.motion_controller.devices.get(axis, None)
        if device is not None:
            try:
                return float(str(device.Position))
            except Exception as e:
                print(f"Error reading {axis} position: {e}")
                return 0.0
        return 0.0

    def _run_focus_stack(self, x, y, z):
        if self.num_stack <= 1:
            z_positions = [z]
        else:
            z_positions = [z - self.z_stack_height/2 + i*(self.z_stack_height/(self.num_stack-1))
                           for i in range(self.num_stack)]
        total = len(z_positions)
        completed = 0
        per_capture = self.time_estimates["Focus Stack"]
        for z_target in z_positions:
            print(f"Moving Z to {z_target:.3f} for focus stack")
            self.hardware.move_to_absolute(z=z_target)
            time.sleep(0.5)
            filename = f"focusstack_X{x:.3f}_Y{y:.3f}_Z{z_target:.3f}.png"
            self.hardware.capture_image(filename=filename)
            completed += 1
            remaining = (total - completed) * per_capture
            self.progress_update.emit(completed, total, remaining)

    def _run_xy_pattern(self, x, y, z):
        fov_x = self.hardware.fov_x
        fov_y = self.hardware.fov_y
        step_x = fov_x * (1 - self.overlap/100.0)
        step_y = fov_y * (1 - self.overlap/100.0)
        half_range_x = self.pattern_size_x / 2.0
        half_range_y = self.pattern_size_y / 2.0
        start_x = x - half_range_x
        end_x = x + half_range_x
        start_y = y - half_range_y
        end_y = y + half_range_y
        positions_x = []
        pos = start_x
        while pos <= end_x:
            positions_x.append(pos)
            pos += step_x
        positions_y = []
        pos = start_y
        while pos <= end_y:
            positions_y.append(pos)
            pos += step_y

        total = len(positions_x) * len(positions_y)
        completed = 0
        per_capture = self.time_estimates["XY Pattern"]
        # Keep Z at the current (base) level.
        for j, y_target in enumerate(positions_y):
            xs = positions_x if j % 2 == 0 else list(reversed(positions_x))
            for x_target in xs:
                print(f"Moving XY to X:{x_target:.3f} Y:{y_target:.3f}")
                self.hardware.move_to_absolute(x=x_target, y=y_target)
                time.sleep(0.5)
                filename = f"xy_X{x_target:.3f}_Y{y_target:.3f}_Z{z:.3f}.png"
                self.hardware.capture_image(filename=filename)
                completed += 1
                remaining = (total - completed) * per_capture
                self.progress_update.emit(completed, total, remaining)

    def _run_combined(self, x, y, z):
        fov_x = self.hardware.fov_x
        fov_y = self.hardware.fov_y
        step_x = fov_x * (1 - self.overlap/100.0)
        step_y = fov_y * (1 - self.overlap/100.0)
        half_range_x = self.pattern_size_x / 2.0
        half_range_y = self.pattern_size_y / 2.0
        start_x = x - half_range_x
        end_x = x + half_range_x
        start_y = y - half_range_y
        end_y = y + half_range_y
        positions_x = []
        pos = start_x
        while pos <= end_x:
            positions_x.append(pos)
            pos += step_x
        positions_y = []
        pos = start_y
        while pos <= end_y:
            positions_y.append(pos)
            pos += step_y

        if self.num_stack <= 1:
            z_positions = [z]
        else:
            z_positions = [z - self.z_stack_height/2 + i*(self.z_stack_height/(self.num_stack-1))
                           for i in range(self.num_stack)]
        total = len(positions_x) * len(positions_y) * len(z_positions)
        completed = 0
        per_capture = self.time_estimates["Combined"]
        for j, y_target in enumerate(positions_y):
            xs = positions_x if j % 2 == 0 else list(reversed(positions_x))
            for x_target in xs:
                print(f"Moving XY to X:{x_target:.3f} Y:{y_target:.3f} for combined mode")
                self.hardware.move_to_absolute(x=x_target, y=y_target)
                time.sleep(0.5)
                for z_target in z_positions:
                    self.hardware.move_to_absolute(z=z_target)
                    time.sleep(0.5)
                    filename = f"combined_X{x_target:.3f}_Y{y_target:.3f}_Z{z_target:.3f}.png"
                    self.hardware.capture_image(filename=filename)
                    completed += 1
                    remaining = (total - completed) * per_capture
                    self.progress_update.emit(completed, total, remaining)
                self.hardware.move_to_absolute(z=z)
                time.sleep(0.5)


###############################################################################
# CameraWorker: Continuously grabs frames from the camera in a separate thread.
###############################################################################
class CameraWorker(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, camera, hardware, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.hardware = hardware  # for auto_capture_in_progress flag
        self.running = True

    def run(self):
        while self.running:
            if self.hardware.auto_capture_in_progress:
                time.sleep(0.1)
                continue
            frame_list = self.camera.grab(1)
            if frame_list:
                frame = np.array(frame_list[0])
                min_val = frame.min()
                max_val = frame.max()
                scaled = (frame - min_val) / (max_val - min_val + 1e-6) * 255
                display_frame = scaled.astype(np.uint8)
                self.new_frame.emit(display_frame)

    def stop(self):
        self.running = False
        self.wait()


###############################################################################
# MicroscopeHardware: Combines LED, Camera, and Motion; includes auto-capture helpers.
###############################################################################
class MicroscopeHardware:
    def __init__(self):
        self.upHandle = None
        self.led_lib = None
        self.camera = None
        self.last_frame = None
        self.motion_controller = MotionController()
        self.move_workers = []
        self.fov_x = 12.0  # mm
        self.fov_y = 10.0  # mm
        self.max_slide_x = 50.0  # mm
        self.max_slide_y = 50.0  # mm
        self.auto_capture_in_progress = False
        self.auto_capture_folder = None
        self._init_led_hardware()
        self._init_camera_hardware()

    def _init_camera_hardware(self):
        pll.par["devices/dlls/thorlabs_tlcam"] = "path/to/dlls"
        cameras = Thorlabs.list_cameras_tlcam()
        if not cameras:
            print("No Thorlabs cameras found.")
            return
        self.camera = Thorlabs.ThorlabsTLCamera(serial=cameras[0])
        print(f"Connected to camera: {cameras[0]}")
        default_exposure = 0.1  # ms
        self.camera.set_exposure(default_exposure)
        print(f"Camera exposure set to {default_exposure} ms")

    def _init_led_hardware(self):
        dll_path = r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLUP_64.dll"
        try:
            lib = cdll.LoadLibrary(dll_path)
        except OSError:
            print(f"Failed to load LED DLL at {dll_path}. Check path.")
            return
        deviceCount = c_uint32()
        lib.TLUP_findRsrc(0, byref(deviceCount))
        if deviceCount.value == 0:
            print("No upSeries devices found.")
            return
        print("Number of upSeries devices found:", deviceCount.value)
        modelName = create_string_buffer(256)
        serialNumber = create_string_buffer(256)
        lib.TLUP_getRsrcInfo(0, 0, modelName, serialNumber, 0, 0)
        print("Connecting to LED device:")
        print("Model name:", modelName.value.decode(),
              ", Serial number:", serialNumber.value.decode())
        upName = create_string_buffer(256)
        lib.TLUP_getRsrcName(0, 0, upName)
        upHandle = c_int(0)
        res = lib.TLUP_init(upName.value, 0, 0, byref(upHandle))
        if res != 0:
            print("Failed to initialize LED device.")
            return
        self.led_lib = lib
        self.upHandle = upHandle
        self.led_lib.TLUP_setLedUseNonThorlabsLed(self.upHandle, 1)
        self.led_lib.TLUP_switchLedOutput(self.upHandle, 0)
        LEDName = create_string_buffer(256)
        LEDSerialNumber = create_string_buffer(256)
        LEDCurrentLimit = c_double()
        LEDForwardVoltage = c_double()
        LEDWavelength = c_double(0)
        self.led_lib.TLUP_getLedInfo(self.upHandle, LEDName, LEDSerialNumber,
                                     byref(LEDCurrentLimit),
                                     byref(LEDForwardVoltage),
                                     byref(LEDWavelength))
        self.led_max_current = LEDCurrentLimit.value
        if LEDWavelength.value != 0:
            print("EEPROM on LED detected.")
            print("Connected LED:", LEDName.value.decode(),
                  ", serial:", LEDSerialNumber.value.decode(),
                  ", wavelength [nm]:", LEDWavelength.value)
            print("Max LED current [A]:", LEDCurrentLimit.value,
                  ", max forward voltage [V]:", LEDForwardVoltage.value)
            currentSetpoint = c_double(0.1 * LEDCurrentLimit.value)
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, currentSetpoint)
            time.sleep(1)
            print("LED current set to 10% of limit.")
        else:
            print("No EEPROM on LED detected. Defaulting to 1mA current.")
            currentSetpoint = c_double(0.001)
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, currentSetpoint)
        print("LED init complete.\n")

    def start_camera(self):
        if self.camera:
            self.camera.start_acquisition()
            print("Camera acquisition started.")
    def stop_camera(self):
        if self.camera:
            self.camera.stop_acquisition()
            print("Camera acquisition stopped.")
    def close_hardware(self):
        if self.camera:
            self.camera.close()
            print("Camera closed.")
            self.camera = None
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_close(self.upHandle)
            self.upHandle = None
            print("LED device closed.")
        if self.motion_controller:
            self.motion_controller.disconnect_all()
    def set_exposure(self, exposure_ms):
        if self.camera:
            self.camera.set_exposure(float(exposure_ms))
            print(f"Camera exposure set to {exposure_ms} ms")
    def move_stage(self, x=None, y=None, z=None):
        if x is not None:
            worker = StageMoveWorker(self, "X", x)
            worker.finished_move.connect(lambda w=worker: self._cleanup_worker(w))
            self.move_workers.append(worker)
            worker.start()
        if y is not None:
            worker = StageMoveWorker(self, "Y", y)
            worker.finished_move.connect(lambda w=worker: self._cleanup_worker(w))
            self.move_workers.append(worker)
            worker.start()
        if z is not None:
            worker = StageMoveWorker(self, "Z", z)
            worker.finished_move.connect(lambda w=worker: self._cleanup_worker(w))
            self.move_workers.append(worker)
            worker.start()
    def _cleanup_worker(self, worker):
        if worker in self.move_workers:
            self.move_workers.remove(worker)
    def _move_axis_absolute(self, axis, step_mm):
        if axis not in self.motion_controller.devices:
            print(f"[{axis}] Axis not initialized.")
            return
        device = self.motion_controller.devices[axis]
        try:
            current_pos = float(str(device.Position))
        except Exception as e:
            print(f"[{axis}] Error reading current position: {e}")
            return
        new_pos = current_pos + step_mm
        print(f"[{axis}] Absolute move from {current_pos:.4f} mm to {new_pos:.4f} mm (requested {step_mm} mm)")
        try:
            device.MoveTo(Decimal(new_pos), Int32(20000))
        except Exception as e:
            print(f"[{axis}] MoveTo command failed: {e}")
        time.sleep(0.1)
        print(f"[{axis}] Move done. Current pos: {device.Position}")
    def move_to_absolute(self, x=None, y=None, z=None):
        if x is not None:
            device = self.motion_controller.devices.get("X")
            if device is not None:
                current_x = float(str(device.Position))
                delta = x - current_x
                self._move_axis_absolute("X", delta)
        if y is not None:
            device = self.motion_controller.devices.get("Y")
            if device is not None:
                current_y = float(str(device.Position))
                delta = y - current_y
                self._move_axis_absolute("Y", delta)
        if z is not None:
            device = self.motion_controller.devices.get("Z")
            if device is not None:
                current_z = float(str(device.Position))
                delta = z - current_z
                self._move_axis_absolute("Z", delta)
    def turn_on_led(self):
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 1)
            print("[LED] Turned on.")
    def turn_off_led(self):
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 0)
            print("[LED] Turned off.")
    def set_uv_led_intensity(self, intensity_percent):
        print("[UV LED] (Not implemented)")
    def set_white_led_intensity(self, intensity_percent):
        if self.led_lib and self.upHandle:
            current = (intensity_percent / 100.0) * self.led_max_current
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, c_double(current))
            print(f"[White LED] Intensity set to {intensity_percent}% (current = {current} A)")
    def capture_image(self, filename="captured_image.png"):
        if not self.camera:
            print("No camera available for auto capture!")
            return
        self.turn_on_led()
        time.sleep(0.5)
        frame_list = self.camera.grab(1)
        if frame_list:
            frame = np.array(frame_list[0])
            self._save_frame_to_downloads(frame, filename)
        else:
            print("[CAPTURE] No frame received from camera!")
        self.turn_off_led()
    def capture_and_save_image(self):
        self.capture_image(filename="captured_image.png")
    def save_last_frame(self):
        if self.last_frame is None:
            print("[SAVE] No current frame available to save!")
            return
        self._save_frame_to_downloads(self.last_frame, "current_frame.png")
    def _save_frame_to_downloads(self, frame, filename):
        # Save in full 16-bit depth.
        if frame.dtype != np.uint16:
            # Scale to full 16-bit (0-65535)
            if frame.max() > 0:
                frame_uint16 = (frame.astype(np.float32) / frame.max() * 65535).astype(np.uint16)
            else:
                frame_uint16 = frame.astype(np.uint16)
        else:
            frame_uint16 = frame
        # Use auto_capture_folder if defined.
        folder = self.auto_capture_folder if self.auto_capture_folder is not None else os.path.join(os.path.expanduser("~"), "Downloads")
        image_path = os.path.join(folder, filename)
        cv2.imwrite(image_path, frame_uint16)
        print(f"[SAVE] Saved image to {image_path}")


###############################################################################
# CameraDisplayWidget
###############################################################################
class CameraDisplayWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
    def update_frame(self, frame):
        if frame is None:
            return
        if len(frame.shape) == 2:
            frame = np.stack((frame,)*3, axis=-1)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


###############################################################################
# ControlPanelWidget
###############################################################################
class ControlPanelWidget(QWidget):
    def __init__(self, hardware: MicroscopeHardware, parent=None):
        super().__init__(parent)
        self.hardware = hardware
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Stage Controls Group
        stage_group = QGroupBox("Stage Controls")
        stage_layout = QGridLayout()
        stage_group.setLayout(stage_layout)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setDecimals(3)
        self.step_spin.setRange(0.001, 50.0)
        self.step_spin.setSingleStep(0.001)
        self.step_spin.setValue(0.010)
        self.step_spin.setSuffix(" mm")
        self.x_pos_button = QPushButton("X +")
        self.x_neg_button = QPushButton("X -")
        self.y_pos_button = QPushButton("Y +")
        self.y_neg_button = QPushButton("Y -")
        self.z_pos_button = QPushButton("Z +")
        self.z_neg_button = QPushButton("Z -")
        self.x_pos_button.clicked.connect(lambda: self.hardware.move_stage(x=self.step_spin.value()))
        self.x_neg_button.clicked.connect(lambda: self.hardware.move_stage(x=-self.step_spin.value()))
        self.y_pos_button.clicked.connect(lambda: self.hardware.move_stage(y=self.step_spin.value()))
        self.y_neg_button.clicked.connect(lambda: self.hardware.move_stage(y=-self.step_spin.value()))
        self.z_pos_button.clicked.connect(lambda: self.hardware.move_stage(z=self.step_spin.value()))
        self.z_neg_button.clicked.connect(lambda: self.hardware.move_stage(z=-self.step_spin.value()))
        stage_layout.addWidget(QLabel("Step size:"), 0, 0)
        stage_layout.addWidget(self.step_spin, 0, 1)
        stage_layout.addWidget(self.x_pos_button, 1, 0)
        stage_layout.addWidget(self.x_neg_button, 1, 1)
        stage_layout.addWidget(self.y_pos_button, 2, 0)
        stage_layout.addWidget(self.y_neg_button, 2, 1)
        stage_layout.addWidget(self.z_pos_button, 3, 0)
        stage_layout.addWidget(self.z_neg_button, 3, 1)
        layout.addWidget(stage_group)

        # Stage Coordinates Display
        self.coordinates_label = QLabel("Stage Coordinates: X: 0.000 mm, Y: 0.000 mm, Z: 0.000 mm")
        layout.addWidget(self.coordinates_label)
        self.coordinates_timer = QTimer(self)
        self.coordinates_timer.timeout.connect(self.update_coordinates)
        self.coordinates_timer.start(500)

        # LED Controls Group
        led_group = QGroupBox("LED Controls")
        led_layout = QGridLayout()
        led_group.setLayout(led_layout)
        led_layout.addWidget(QLabel("UV LED:"), 0, 0)
        self.uv_led_checkbox = QCheckBox("Enable")
        led_layout.addWidget(self.uv_led_checkbox, 0, 1)
        led_layout.addWidget(QLabel("UV Intensity (%):"), 1, 0)
        self.uv_intensity_slider = QSlider(Qt.Horizontal)
        self.uv_intensity_slider.setRange(0, 100)
        self.uv_intensity_slider.setValue(10)
        self.uv_intensity_slider.valueChanged.connect(self.update_uv_intensity_from_slider)
        led_layout.addWidget(self.uv_intensity_slider, 1, 1)
        self.uv_intensity_edit = QLineEdit("10")
        self.uv_intensity_edit.setFixedWidth(60)
        self.uv_intensity_edit.editingFinished.connect(self.update_uv_slider_from_intensity)
        led_layout.addWidget(self.uv_intensity_edit, 1, 2)
        led_layout.addWidget(QLabel("White LED:"), 2, 0)
        self.white_led_checkbox = QCheckBox("Enable")
        self.white_led_checkbox.stateChanged.connect(self.white_led_toggled)
        led_layout.addWidget(self.white_led_checkbox, 2, 1)
        led_layout.addWidget(QLabel("White Intensity (%):"), 3, 0)
        self.white_intensity_slider = QSlider(Qt.Horizontal)
        self.white_intensity_slider.setRange(0, 100)
        self.white_intensity_slider.setValue(10)
        self.white_intensity_slider.valueChanged.connect(self.update_white_intensity_from_slider)
        led_layout.addWidget(self.white_intensity_slider, 3, 1)
        self.white_intensity_edit = QLineEdit("10")
        self.white_intensity_edit.setFixedWidth(60)
        self.white_intensity_edit.editingFinished.connect(self.update_white_slider_from_intensity)
        led_layout.addWidget(self.white_intensity_edit, 3, 2)
        layout.addWidget(led_group)

        # Camera Controls Group
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QHBoxLayout()
        camera_group.setLayout(camera_layout)
        camera_layout.addWidget(QLabel("Exposure (ms):"))
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 100)
        self.exposure_slider.setValue(10)
        self.exposure_slider.valueChanged.connect(self.update_exposure_from_slider)
        camera_layout.addWidget(self.exposure_slider)
        self.exposure_edit = QLineEdit("0.10")
        self.exposure_edit.setFixedWidth(60)
        self.exposure_edit.editingFinished.connect(self.update_slider_from_exposure)
        camera_layout.addWidget(self.exposure_edit)
        layout.addWidget(camera_group)

        # Automatic Capture Options Group
        auto_group = QGroupBox("Automatic Capture Options")
        auto_layout = QGridLayout()
        auto_group.setLayout(auto_layout)
        auto_layout.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Focus Stack", "XY Pattern", "Combined"])
        auto_layout.addWidget(self.mode_combo, 0, 1)
        auto_layout.addWidget(QLabel("Overlap (%):"), 1, 0)
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0, 100)
        self.overlap_spin.setDecimals(1)
        self.overlap_spin.setValue(20.0)
        auto_layout.addWidget(self.overlap_spin, 1, 1)
        auto_layout.addWidget(QLabel("# Images (Stack):"), 2, 0)
        self.stack_count_spin = QSpinBox()
        self.stack_count_spin.setRange(1, 100)
        self.stack_count_spin.setValue(5)
        auto_layout.addWidget(self.stack_count_spin, 2, 1)
        auto_layout.addWidget(QLabel("Pattern Size X (mm):"), 3, 0)
        self.pattern_x_spin = QDoubleSpinBox()
        self.pattern_x_spin.setRange(0.1, self.hardware.max_slide_x)
        self.pattern_x_spin.setValue(10.0)
        auto_layout.addWidget(self.pattern_x_spin, 3, 1)
        auto_layout.addWidget(QLabel("Pattern Size Y (mm):"), 4, 0)
        self.pattern_y_spin = QDoubleSpinBox()
        self.pattern_y_spin.setRange(0.1, self.hardware.max_slide_y)
        self.pattern_y_spin.setValue(10.0)
        auto_layout.addWidget(self.pattern_y_spin, 4, 1)
        auto_layout.addWidget(QLabel("Z Stack Height (mm):"), 5, 0)
        self.z_stack_height_spin = QDoubleSpinBox()
        self.z_stack_height_spin.setRange(0.0, 20.0)
        self.z_stack_height_spin.setDecimals(3)
        self.z_stack_height_spin.setValue(1.0)
        auto_layout.addWidget(self.z_stack_height_spin, 5, 1)
        layout.addWidget(auto_group)

        # Progress Bar and Estimated Time Display
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("")
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

        # Capture/Save Buttons
        self.capture_button = QPushButton("Auto Capture")
        self.capture_button.clicked.connect(self.auto_capture_clicked)
        layout.addWidget(self.capture_button)
        self.save_button = QPushButton("Save Current Photo")
        self.save_button.clicked.connect(self.save_current_photo_clicked)
        layout.addWidget(self.save_button)
        layout.addStretch()

    def update_coordinates(self):
        mc = self.hardware.motion_controller
        coords = {}
        for axis in ['X', 'Y', 'Z']:
            device = mc.devices.get(axis, None)
            if device is not None:
                try:
                    coords[axis] = float(str(device.Position))
                except Exception as e:
                    coords[axis] = 0.0
            else:
                coords[axis] = 0.0
        self.coordinates_label.setText(
            f"Stage Coordinates: X: {coords['X']:.3f} mm, Y: {coords['Y']:.3f} mm, Z: {coords['Z']:.3f} mm"
        )

    @pyqtSlot(int)
    def white_led_toggled(self, state):
        if state == Qt.Checked:
            try:
                intensity = int(self.white_intensity_edit.text())
            except ValueError:
                intensity = self.white_intensity_slider.value()
                self.white_intensity_edit.setText(str(intensity))
            self.hardware.set_white_led_intensity(intensity)
            self.hardware.turn_on_led()
        else:
            self.hardware.turn_off_led()

    def update_uv_intensity_from_slider(self, slider_value):
        self.uv_intensity_edit.setText(str(slider_value))
    def update_uv_slider_from_intensity(self):
        try:
            val = int(self.uv_intensity_edit.text())
        except ValueError:
            val = self.uv_intensity_slider.value()
            self.uv_intensity_edit.setText(str(val))
            return
        val = max(0, min(val, 100))
        self.uv_intensity_slider.setValue(val)
    def update_white_intensity_from_slider(self, slider_value):
        self.white_intensity_edit.setText(str(slider_value))
        if self.white_led_checkbox.isChecked():
            self.hardware.set_white_led_intensity(slider_value)
    def update_white_slider_from_intensity(self):
        try:
            val = int(self.white_intensity_edit.text())
        except ValueError:
            val = self.white_intensity_slider.value()
            self.white_intensity_edit.setText(str(val))
            return
        val = max(0, min(val, 100))
        self.white_intensity_slider.setValue(val)
        if self.white_led_checkbox.isChecked():
            self.hardware.set_white_led_intensity(val)
    def update_exposure_from_slider(self, slider_value):
        exposure_ms = slider_value / 100.0
        self.exposure_edit.setText(f"{exposure_ms:.2f}")
        self.hardware.set_exposure(exposure_ms)
    def update_slider_from_exposure(self):
        try:
            entered_value = float(self.exposure_edit.text())
        except ValueError:
            slider_val = self.exposure_slider.value()
            self.exposure_edit.setText(f"{slider_val/100:.2f}")
            return
        entered_value = max(0.01, min(entered_value, 1.0))
        slider_val = int(round(entered_value * 100))
        self.exposure_slider.setValue(slider_val)
        self.hardware.set_exposure(entered_value)

    def update_progress(self, completed, total, remaining):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(completed)
        percent = (completed / total) * 100
        self.progress_label.setText(f"Progress: {percent:.1f}% | Estimated time remaining: {remaining:.0f} sec")

    def auto_capture_clicked(self):
        mode = self.mode_combo.currentText()
        overlap = self.overlap_spin.value()
        num_stack = self.stack_count_spin.value()
        pattern_x = self.pattern_x_spin.value()
        pattern_y = self.pattern_y_spin.value()
        z_stack_height = self.z_stack_height_spin.value()
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: 0% | Estimated time remaining: --")
        self.auto_capture_worker = AutoCaptureWorker(self.hardware, mode, overlap, num_stack,
                                                     pattern_x, pattern_y, z_stack_height)
        self.auto_capture_worker.progress_update.connect(self.update_progress)
        self.auto_capture_worker.finished_capture.connect(self.auto_capture_finished)
        self.auto_capture_worker.start()

    def auto_capture_finished(self):
        print("Auto Capture completed.")
        self.progress_bar.hide()
        self.progress_label.hide()

    def save_current_photo_clicked(self):
        self.hardware.save_last_frame()


###############################################################################
# MainWindow
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Control UI")
        self.hardware = MicroscopeHardware()
        self.hardware.start_camera()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.camera_display = CameraDisplayWidget()
        main_layout.addWidget(self.camera_display, stretch=2)
        self.control_panel = ControlPanelWidget(self.hardware)
        main_layout.addWidget(self.control_panel, stretch=1)
        if self.hardware.camera:
            self.camera_thread = CameraWorker(self.hardware.camera, self.hardware)
            self.camera_thread.new_frame.connect(self.on_new_frame)
            self.camera_thread.start()
        else:
            self.camera_thread = None

    @pyqtSlot(np.ndarray)
    def on_new_frame(self, frame):
        self.hardware.last_frame = frame
        self.camera_display.update_frame(frame)

    def closeEvent(self, event):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.hardware.stop_camera()
        self.hardware.close_hardware()
        super().closeEvent(event)


###############################################################################
# Entry Point
###############################################################################
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
