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
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot, QMutex, QMutexLocker
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox, QGridLayout, QCheckBox, QLineEdit, QProgressBar, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon

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
            device.StartPolling(250)
            device.EnableDevice()
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
        time.sleep(0.05)
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
        self.hardware = hardware
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
    progress_update = pyqtSignal(int, int, float)
    new_capture = pyqtSignal(np.ndarray)

    def __init__(self, hardware, mode, overlap, num_stack, pattern_size_x, pattern_size_y, z_stack_height, parent=None):
        super().__init__(parent)
        self.hardware = hardware
        self.mode = mode
        self.overlap = overlap
        self.num_stack = num_stack
        self.pattern_size_x = pattern_size_x
        self.pattern_size_y = pattern_size_y
        self.z_stack_height = z_stack_height

        self.time_estimates = {
            "Focus Stack": 3.0,
            "XY Pattern": 7.0,
            "Combined": 5.1
        }

    def run(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        folder_name = os.path.join(downloads, f"AutoCapture_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)
        self.hardware.auto_capture_folder = folder_name

        self.hardware.auto_capture_in_progress = True

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
            self.hardware.capture_image(filename=f"focusstack_X{x:.3f}_Y{y:.3f}_Z{z_target:.3f}.png")
            if self.hardware.last_frame is not None:
                self.new_capture.emit(self.hardware.last_frame)
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
        for j, y_target in enumerate(positions_y):
            xs = positions_x if j % 2 == 0 else list(reversed(positions_x))
            for x_target in xs:
                print(f"Moving XY to X:{x_target:.3f} Y:{y_target:.3f}")
                self.hardware.move_to_absolute(x=x_target, y=y_target)
                time.sleep(0.5)
                self.hardware.capture_image(filename=f"xy_X{x_target:.3f}_Y{y_target:.3f}_Z{z:.3f}.png")
                if self.hardware.last_frame is not None:
                    self.new_capture.emit(self.hardware.last_frame)
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
                    self.hardware.capture_image(filename=f"combined_X{x_target:.3f}_Y{y_target:.3f}_Z{z_target:.3f}.png")
                    if self.hardware.last_frame is not None:
                        self.new_capture.emit(self.hardware.last_frame)
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
        self.hardware = hardware
        self.running = True

    def run(self):
        while self.running:
            with QMutexLocker(self.hardware.camera_lock):
                frame_list = self.camera.grab(1)
            if frame_list:
                frame = np.array(frame_list[0])
                if frame.min() == frame.max():
                    display_frame = np.full(frame.shape, 255, dtype=np.uint8)
                else:
                    min_val = frame.min()
                    max_val = frame.max()
                    scaled = (frame - min_val) / (max_val - min_val + 1e-6) * 255
                    display_frame = scaled.astype(np.uint8)
                self.new_frame.emit(display_frame)
            else:
                if self.hardware.last_frame is not None:
                    self.new_frame.emit(self.hardware.last_frame)

    def stop(self):
        self.running = False
        self.wait()


###############################################################################
# MicroscopeHardware: Combines LED, Camera, and Motion; includes auto-capture helpers.
###############################################################################
class MicroscopeHardware:
    def __init__(self):
        self.led_handle = None
        self.led_max_current = None
        self.current_led_type = None
        self.led_lib = None
        self.camera = None
        self.last_frame = None
        self.motion_controller = MotionController()
        self.move_workers = []
        self.fov_x = 12.0  # mm (default; may change based on optic)
        self.fov_y = 10.0  # mm
        self.max_slide_x = 50.0  # mm
        self.max_slide_y = 50.0  # mm
        self.auto_capture_in_progress = False
        self.auto_capture_folder = None
        self.connect_led("White LED")
        self._init_camera_hardware()
        self.camera_lock = QMutex()

    def _init_camera_hardware(self):
        pll.par["devices/dlls/thorlabs_tlcam"] = "path/to/dlls"
        cameras = Thorlabs.list_cameras_tlcam()
        if not cameras:
            print("No Thorlabs cameras found.")
            return
        self.camera = Thorlabs.ThorlabsTLCamera(serial=cameras[0])
        print(f"Connected to camera: {cameras[0]}")
        default_exposure = 0.1
        self.camera.set_exposure(default_exposure)
        print(f"Camera exposure set to {default_exposure} s")

    def disconnect_led(self):
        if self.led_handle is not None and self.led_lib is not None:
            print(f"Disconnecting {self.current_led_type}...")
            self.led_lib.TLUP_switchLedOutput(self.led_handle, 0)
            self.led_lib.TLUP_close(self.led_handle)
            print(f"{self.current_led_type} disconnected.")
            self.led_handle = None
            self.current_led_type = None
            time.sleep(1)

    def connect_led(self, led_type):
        print(f"Attempting to connect to {led_type}...")
        self.disconnect_led()
        time.sleep(1)
        dll_path = r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLUP_64.dll"
        try:
            lib = cdll.LoadLibrary(dll_path)
        except OSError:
            print(f"Failed to load LED DLL at {dll_path}.")
            return False
        deviceCount = c_uint32()
        lib.TLUP_findRsrc(0, byref(deviceCount))
        print(f"Found {deviceCount.value} LED device(s).")
        if deviceCount.value == 0:
            print("No upSeries devices found.")
            return False

        for i in range(deviceCount.value):
            modelName = create_string_buffer(256)
            serialNumber = create_string_buffer(256)
            lib.TLUP_getRsrcInfo(i, 0, modelName, serialNumber, 0, 0)
            print(f"Device index {i}: Model: {modelName.value.decode().strip()}, Serial: {serialNumber.value.decode().strip()}")

        connected = False
        for i in range(deviceCount.value):
            modelName = create_string_buffer(256)
            serialNumber = create_string_buffer(256)
            lib.TLUP_getRsrcInfo(i, 0, modelName, serialNumber, 0, 0)
            upName = create_string_buffer(256)
            lib.TLUP_getRsrcName(i, 0, upName)
            resource_str = upName.value.decode().strip()
            print(f"Device index {i}: Resource: {resource_str}")
            led_handle = c_int(0)
            res = lib.TLUP_init(upName.value, 0, 0, byref(led_handle))
            print(f"TLUP_init returned {res} for device index {i}.")
            if res != 0:
                print(f"Failed to initialize LED device at index {i}.")
                continue
            lib.TLUP_setLedUseNonThorlabsLed(led_handle, 1)
            lib.TLUP_switchLedOutput(led_handle, 0)
            LEDName = create_string_buffer(256)
            LEDSerialNumber = create_string_buffer(256)
            LEDCurrentLimit = c_double()
            LEDForwardVoltage = c_double()
            LEDWavelength = c_double(0)
            lib.TLUP_getLedInfo(led_handle, LEDName, LEDSerialNumber,
                                byref(LEDCurrentLimit),
                                byref(LEDForwardVoltage),
                                byref(LEDWavelength))
            print(f"Device index {i} info: Model: {LEDName.value.decode().strip()}, Serial: {LEDSerialNumber.value.decode().strip()}, Wavelength: {LEDWavelength.value}")
            if led_type == "White LED" and LEDWavelength.value == -4000.0:
                print("White LED selected.")
                self.led_handle = led_handle
                self.led_max_current = LEDCurrentLimit.value
                self.current_led_type = "White LED"
                if LEDWavelength.value != 0:
                    print("EEPROM on white LED detected. Max current:", LEDCurrentLimit.value, "Max voltage:", LEDForwardVoltage.value)
                    currentSetpoint = c_double(0.1 * LEDCurrentLimit.value)
                    lib.TLUP_setLedCurrentSetpoint(led_handle, currentSetpoint)
                    time.sleep(1)
                    print("White LED current set to 10% of limit.")
                else:
                    print("No EEPROM on white LED detected. Defaulting to 1mA current.")
                    currentSetpoint = c_double(0.001)
                    lib.TLUP_setLedCurrentSetpoint(led_handle, currentSetpoint)
                connected = True
                break
            elif led_type == "UV LED" and LEDWavelength.value == 270.0:
                print("UV LED selected.")
                self.led_handle = led_handle
                self.led_max_current = LEDCurrentLimit.value
                self.current_led_type = "UV LED"
                if LEDWavelength.value != 0:
                    print("EEPROM on UV LED detected. Max current:", LEDCurrentLimit.value, "Max voltage:", LEDForwardVoltage.value)
                    currentSetpoint = c_double(0.1 * LEDCurrentLimit.value)
                    lib.TLUP_setLedCurrentSetpoint(led_handle, currentSetpoint)
                    time.sleep(1)
                    print("UV LED current set to 10% of limit.")
                else:
                    print("No EEPROM on UV LED detected. Defaulting to 1mA current.")
                    currentSetpoint = c_double(0.001)
                    lib.TLUP_setLedCurrentSetpoint(led_handle, currentSetpoint)
                connected = True
                break
            else:
                print(f"Device index {i} does not match requested type {led_type}.")
        if not connected:
            print(f"Could not connect to {led_type}.")
            return False
        self.led_lib = lib
        print(f"{self.current_led_type} connected.")
        return True

    def turn_on_led(self):
        if self.led_handle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.led_handle, 1)
            print(f"[{self.current_led_type}] Turned on.")
        else:
            print("LED not initialized.")

    def turn_off_led(self):
        if self.led_handle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.led_handle, 0)
            print(f"[{self.current_led_type}] Turned off.")
        else:
            print("LED not initialized.")

    def set_led_intensity(self, intensity_percent):
        if self.led_lib and self.led_handle and self.led_max_current is not None:
            current = (intensity_percent / 100.0) * self.led_max_current
            self.led_lib.TLUP_setLedCurrentSetpoint(self.led_handle, c_double(current))
            print(f"[{self.current_led_type}] Intensity set to {intensity_percent}% (current = {current} A)")
        else:
            print("LED not initialized.")

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
        self.disconnect_led()
        if self.motion_controller:
            self.motion_controller.disconnect_all()

    def set_exposure(self, exposure_sec):
        if self.camera:
            self.camera.set_exposure(float(exposure_sec))
            print(f"Camera exposure set to {exposure_sec} s")

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

    def capture_image(self, filename="captured_image.png"):
        if not self.camera:
            print("No camera available for auto capture!")
            return
        self.turn_on_led()
        time.sleep(0.5)
        with QMutexLocker(self.camera_lock):
            frame_list = self.camera.grab(1)
        if frame_list:
            frame = np.array(frame_list[0])
            self.last_frame = frame
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
        if frame.dtype != np.uint16:
            if frame.max() > 0:
                frame_uint16 = (frame.astype(np.float32) / frame.max() * 65535).astype(np.uint16)
            else:
                frame_uint16 = frame.astype(np.uint16)
        else:
            frame_uint16 = frame
        folder = self.auto_capture_folder if self.auto_capture_folder is not None else os.path.join(os.path.expanduser("~"), "Downloads")
        image_path = os.path.join(folder, filename)
        cv2.imwrite(image_path, frame_uint16)
        print(f"[SAVE] Saved image to {image_path}")


###############################################################################
# CameraDisplayWidget with click-to-center feature
###############################################################################
class CameraDisplayWidget(QLabel):
    def __init__(self, hardware, parent=None):
        super().__init__(parent)
        self.hardware = hardware
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

    def mousePressEvent(self, event):
        if self.pixmap() is None:
            return
        clicked_point = event.pos()
        pixmap = self.pixmap()
        disp_width = pixmap.width()
        disp_height = pixmap.height()
        widget_width = self.width()
        widget_height = self.height()
        # Calculate the offset if the pixmap is centered
        offset_x = (widget_width - disp_width) / 2
        offset_y = (widget_height - disp_height) / 2
        image_x = clicked_point.x() - offset_x
        image_y = clicked_point.y() - offset_y
        if image_x < 0 or image_y < 0 or image_x > disp_width or image_y > disp_height:
            return  # Click outside the image area.
        center_x = disp_width / 2.0
        center_y = disp_height / 2.0
        delta_pixels_x = image_x - center_x
        delta_pixels_y = image_y - center_y
        # Convert pixel difference to millimeters based on the current field of view.
        mm_per_pixel_x = self.hardware.fov_x / disp_width
        mm_per_pixel_y = self.hardware.fov_y / disp_height
        move_x = delta_pixels_x * mm_per_pixel_x
        move_y = -delta_pixels_y * mm_per_pixel_y
        print(f"Click at image coords: ({image_x:.2f}, {image_y:.2f}); Pixel delta: ({delta_pixels_x:.2f}, {delta_pixels_y:.2f}); Moving stage by: ({move_x:.3f} mm, {move_y:.3f} mm)")
        self.hardware.move_stage(x=move_x, y=move_y)


###############################################################################
# ControlPanelWidget with revised Stage Controls (arrow buttons) and updated exposure slider
###############################################################################
class ControlPanelWidget(QWidget):
    def __init__(self, hardware: MicroscopeHardware, parent=None):
        super().__init__(parent)
        self.hardware = hardware
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Revised Stage Controls Group using arrow buttons for X/Y and separate Z controls
        stage_group = QGroupBox("Stage Controls")
        stage_layout = QVBoxLayout()
        stage_group.setLayout(stage_layout)

        arrow_panel = QWidget()
        grid = QGridLayout()
        arrow_panel.setLayout(grid)
        font = QFont()
        font.setPointSize(16)
        self.btn_up = QPushButton("↑")
        self.btn_up.setFont(font)
        self.btn_up.clicked.connect(lambda: self.hardware.move_stage(y=self.step_value()))
        grid.addWidget(self.btn_up, 0, 1)
        self.btn_left = QPushButton("←")
        self.btn_left.setFont(font)
        self.btn_left.clicked.connect(lambda: self.hardware.move_stage(x=-self.step_value()))
        grid.addWidget(self.btn_left, 1, 0)
        center_label = QLabel("")
        grid.addWidget(center_label, 1, 1)
        self.btn_right = QPushButton("→")
        self.btn_right.setFont(font)
        self.btn_right.clicked.connect(lambda: self.hardware.move_stage(x=self.step_value()))
        grid.addWidget(self.btn_right, 1, 2)
        self.btn_down = QPushButton("↓")
        self.btn_down.setFont(font)
        self.btn_down.clicked.connect(lambda: self.hardware.move_stage(y=-self.step_value()))
        grid.addWidget(self.btn_down, 2, 1)
        stage_layout.addWidget(QLabel("X/Y Controls:"))
        stage_layout.addWidget(arrow_panel)

        z_panel = QWidget()
        z_layout = QHBoxLayout()
        z_panel.setLayout(z_layout)
        self.btn_z_up = QPushButton("Z ↑")
        self.btn_z_down = QPushButton("Z ↓")
        self.btn_z_up.clicked.connect(lambda: self.hardware.move_stage(z=self.step_value()))
        self.btn_z_down.clicked.connect(lambda: self.hardware.move_stage(z=-self.step_value()))
        z_layout.addWidget(self.btn_z_down)
        z_layout.addWidget(self.btn_z_up)
        stage_layout.addWidget(QLabel("Z Controls:"))
        stage_layout.addWidget(z_panel)

        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step size:"))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setDecimals(3)
        self.step_spin.setRange(0.001, 50.0)
        self.step_spin.setSingleStep(0.001)
        self.step_spin.setValue(0.010)
        self.step_spin.setSuffix(" mm")
        step_layout.addWidget(self.step_spin)
        stage_layout.addLayout(step_layout)

        layout.addWidget(stage_group)

        # Other control groups remain unchanged...
        optic_group = QGroupBox("Optic Controls")
        optic_layout = QHBoxLayout()
        optic_group.setLayout(optic_layout)
        optic_layout.addWidget(QLabel("Optic:"))
        self.optic_combo = QComboBox()
        self.optic_combo.addItems(["4X", "10X"])
        self.optic_combo.currentIndexChanged.connect(self.update_optic)
        optic_layout.addWidget(self.optic_combo)
        self.zero_button = QPushButton("Zero")
        self.zero_button.clicked.connect(self.zero_stage)
        optic_layout.addWidget(self.zero_button)
        layout.addWidget(optic_group)

        self.coordinates_label = QLabel("Stage Coordinates: X: 0.000 mm, Y: 0.000 mm, Z: 0.000 mm")
        layout.addWidget(self.coordinates_label)
        self.coordinates_timer = QTimer(self)
        self.coordinates_timer.timeout.connect(self.update_coordinates)
        self.coordinates_timer.start(500)

        led_group = QGroupBox("LED Controls")
        led_layout = QGridLayout()
        led_group.setLayout(led_layout)
        led_layout.addWidget(QLabel("Select LED:"), 0, 0)
        self.led_combo = QComboBox()
        self.led_combo.addItems(["White LED", "UV LED"])
        self.led_combo.currentIndexChanged.connect(self.led_selection_changed)
        led_layout.addWidget(self.led_combo, 0, 1, 1, 2)
        led_layout.addWidget(QLabel("LED:"), 1, 0)
        self.led_enable_checkbox = QCheckBox("Enable")
        self.led_enable_checkbox.stateChanged.connect(self.led_enable_toggled)
        led_layout.addWidget(self.led_enable_checkbox, 1, 1)
        led_layout.addWidget(QLabel("Intensity (%):"), 2, 0)
        self.led_intensity_slider = QSlider(Qt.Horizontal)
        self.led_intensity_slider.setRange(0, 100)
        self.led_intensity_slider.setValue(10)
        self.led_intensity_slider.valueChanged.connect(self.update_led_intensity_from_slider)
        led_layout.addWidget(self.led_intensity_slider, 2, 1)
        self.led_intensity_edit = QLineEdit("10")
        self.led_intensity_edit.setFixedWidth(60)
        self.led_intensity_edit.editingFinished.connect(self.update_led_slider_from_intensity)
        led_layout.addWidget(self.led_intensity_edit, 2, 2)
        layout.addWidget(led_group)

        camera_group = QGroupBox("Camera Controls")
        camera_layout = QHBoxLayout()
        camera_group.setLayout(camera_layout)
        camera_layout.addWidget(QLabel("Exposure (s):"))
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 1000)
        self.exposure_slider.setValue(100)
        self.exposure_slider.valueChanged.connect(self.update_exposure_from_slider)
        camera_layout.addWidget(self.exposure_slider)
        self.exposure_edit = QLineEdit("0.100")
        self.exposure_edit.setFixedWidth(60)
        self.exposure_edit.editingFinished.connect(self.update_slider_from_exposure)
        camera_layout.addWidget(self.exposure_edit)
        layout.addWidget(camera_group)

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

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("")
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

        self.capture_button = QPushButton("Auto Capture")
        self.capture_button.clicked.connect(self.auto_capture_clicked)
        layout.addWidget(self.capture_button)
        self.save_button = QPushButton("Save Current Photo")
        self.save_button.clicked.connect(self.save_current_photo_clicked)
        layout.addWidget(self.save_button)
        layout.addStretch()

    def step_value(self):
        return self.step_spin.value()

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

    def update_optic(self):
        optic = self.optic_combo.currentText()
        if optic == "4X":
            self.hardware.fov_x = 10.0
            self.hardware.fov_y = 12.0
        elif optic == "10X":
            self.hardware.fov_x = 4.0
            self.hardware.fov_y = 5.0
        print(f"Optic selected: {optic}. FOV set to {self.hardware.fov_x}x{self.hardware.fov_y} mm.")

    def zero_stage(self):
        optic = self.optic_combo.currentText()
        x_zero = 16.8
        y_zero = 19.5
        z_zero = 8.3 if optic == "10X" else 46.5
        print(f"Zeroing stage to X:{x_zero} mm, Y:{y_zero} mm, Z:{z_zero} mm based on optic {optic}.")
        self.hardware.move_to_absolute(x=x_zero, y=y_zero, z=z_zero)

    def led_selection_changed(self):
        selected_led = self.led_combo.currentText()
        print(f"LED selection changed to {selected_led}.")
        self.hardware.disconnect_led()
        if not self.hardware.connect_led(selected_led):
            print("LED not initialized.")
        self.led_enable_checkbox.setChecked(False)

    @pyqtSlot(int)
    def led_enable_toggled(self, state):
        if state == Qt.Checked:
            self.hardware.turn_on_led()
        else:
            self.hardware.turn_off_led()

    def update_led_intensity_from_slider(self, slider_value):
        self.led_intensity_edit.setText(str(slider_value))
        self.hardware.set_led_intensity(slider_value)

    def update_led_slider_from_intensity(self):
        try:
            val = int(self.led_intensity_edit.text())
        except ValueError:
            val = self.led_intensity_slider.value()
            self.led_intensity_edit.setText(str(val))
            return
        val = max(0, min(val, 100))
        self.led_intensity_slider.setValue(val)
        self.hardware.set_led_intensity(val)

    def update_exposure_from_slider(self, slider_value):
        exposure_sec = slider_value / 1000.0
        self.exposure_edit.setText(f"{exposure_sec:.3f}")
        self.hardware.set_exposure(exposure_sec)

    def update_slider_from_exposure(self):
        try:
            entered_value = float(self.exposure_edit.text())
        except ValueError:
            slider_val = self.exposure_slider.value()
            self.exposure_edit.setText(f"{slider_val/1000:.3f}")
            return
        entered_value = max(0.001, min(entered_value, 1.0))
        slider_val = int(round(entered_value * 1000))
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
        self.led_enable_checkbox.setChecked(False)

    def save_current_photo_clicked(self):
        self.hardware.save_last_frame()


###############################################################################
# MainWindow
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Control UI")
        self.setWindowIcon(QIcon("icon.ico"))
        self.hardware = MicroscopeHardware()
        self.hardware.start_camera()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        # Pass hardware to CameraDisplayWidget so it can handle clicks.
        self.camera_display = CameraDisplayWidget(self.hardware)
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
