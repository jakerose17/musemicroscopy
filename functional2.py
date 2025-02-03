import sys
import os
import time
import cv2
import numpy as np

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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QDoubleSpinBox, QGroupBox, QGridLayout, QCheckBox, QLineEdit
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
    """
    Controls 3 Thorlabs KCube DC Servos via the Kinesis .NET API.
    Axis → Serial Number:
      X: 27601188
      Y: 27601181
      Z: 27265400

    All moves are performed as absolute moves using MoveTo.
    The step size is given in millimeters.
    """
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
            print(f"Homing axis {axis} (serial {serial_no})...")
            device.Home(60000)
            print(f"[{axis}] Homing complete. Current pos:", device.Position)
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
# Background Thread for Camera Acquisition
###############################################################################
class CameraWorker(QThread):
    """
    Continuously grabs frames from the Thorlabs camera in a separate thread.
    Emits new_frame whenever a frame arrives.
    """
    new_frame = pyqtSignal(np.ndarray)
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.running = True
    def run(self):
        while self.running:
            frame_list = self.camera.grab(1)
            if frame_list:
                frame = np.array(frame_list[0])
                # Convert the 16-bit image to 8-bit by dividing by 256.
                display_frame = (frame / 256).clip(0, 255).astype(np.uint8)
                self.new_frame.emit(display_frame)
    def stop(self):
        self.running = False
        self.wait()


###############################################################################
# MicroscopeHardware: Combines LED, Camera, and Motion
###############################################################################
class MicroscopeHardware:
    """
    Combines:
      - Thorlabs camera (via pylablib)
      - Thorlabs upSeries LED driver (via TLUP_64.dll)
      - A MotionController instance for X, Y, Z stage
    """
    def __init__(self):
        self.upHandle = None
        self.led_lib = None
        self.camera = None
        self.last_frame = None
        self.motion_controller = MotionController()
        self._init_led_hardware()
        self._init_camera_hardware()

    def _init_camera_hardware(self):
        pll.par["devices/dlls/thorlabs_tlcam"] = "path/to/dlls"  # Update as needed
        cameras = Thorlabs.list_cameras_tlcam()
        if not cameras:
            print("No Thorlabs cameras found.")
            return
        self.camera = Thorlabs.ThorlabsTLCamera(serial=cameras[0])
        print(f"Connected to camera: {cameras[0]}")
        default_exposure = 0.1  # ms (adjust as needed)
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
            # Set the current setpoint to 10% of the maximum current.
            currentSetpoint = c_double(0.1 * LEDCurrentLimit.value)
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, currentSetpoint)
            time.sleep(1)
            print("LED current set to 10% of limit.")
        else:
            print("No EEPROM on LED detected. Defaulting to 1mA current.")
            currentSetpoint = c_double(0.001)
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, currentSetpoint)
        print("LED init complete.\n")

    # -------------------------
    # Camera methods
    # -------------------------
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
    # -------------------------
    # Motion methods (absolute moves for all axes)
    # -------------------------
    def move_stage(self, x=None, y=None, z=None):
        if x is not None:
            self._move_axis_absolute("X", x)
        if y is not None:
            self._move_axis_absolute("Y", y)
        if z is not None:
            self._move_axis_absolute("Z", z)

    def _move_axis_absolute(self, axis, step_mm):
        """Reads the current position, adds the step (in mm), and calls MoveTo."""
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

    # -------------------------
    # LED methods
    # -------------------------
    def turn_on_led(self):
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 1)
            print("[LED] Turned on.")
    def turn_off_led(self):
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 0)
            print("[LED] Turned off.")
    def set_uv_led_intensity(self, intensity_percent):
        # Placeholder for UV LED control; to be implemented later.
        print("[UV LED] (Not implemented)")
    def set_white_led_intensity(self, intensity_percent):
        if self.led_lib and self.upHandle:
            # Use the generic LED current setpoint function (as in the working code)
            current = (intensity_percent / 100.0) * self.led_max_current
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, c_double(current))
            print(f"[White LED] Intensity set to {intensity_percent}% (current = {current} A)")
    # -------------------------
    # Capture methods
    # -------------------------
    def capture_and_save_image(self):
        if not self.camera:
            print("No camera available for auto capture!")
            return
        self.turn_on_led()
        time.sleep(0.5)  # let LED stabilize
        frame_list = self.camera.grab(1)
        if frame_list:
            frame = np.array(frame_list[0])
            self._save_frame_to_downloads(frame, "captured_image.png")
        else:
            print("[CAPTURE] No frame received from camera!")
        self.turn_off_led()
    def save_last_frame(self):
        if self.last_frame is None:
            print("[SAVE] No current frame available to save!")
            return
        self._save_frame_to_downloads(self.last_frame, "current_frame.png")
    def _save_frame_to_downloads(self, frame, filename):
        if frame.max() > 0:
            frame_uint8 = (frame / frame.max() * 255).astype('uint8')
        else:
            frame_uint8 = frame.astype('uint8')
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        image_path = os.path.join(downloads, filename)
        cv2.imwrite(image_path, frame_uint8)
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

        # 1) Stage Controls Group (using QDoubleSpinBox for mm input)
        stage_group = QGroupBox("Stage Controls")
        stage_layout = QGridLayout()
        stage_group.setLayout(stage_layout)

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setDecimals(3)
        self.step_spin.setRange(0.001, 50.0)  # Range in mm
        self.step_spin.setSingleStep(0.001)
        self.step_spin.setValue(0.010)  # default 0.01 mm
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

        # 2) LED Controls Group with brightness sliders for UV and White LEDs
        led_group = QGroupBox("LED Controls")
        led_layout = QGridLayout()
        led_group.setLayout(led_layout)

        # UV LED Controls (placeholder)
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

        # White LED Controls
        led_layout.addWidget(QLabel("White LED:"), 2, 0)
        self.white_led_checkbox = QCheckBox("Enable")
        # Modify the white LED toggle so that it immediately sets brightness and turns the LED on/off.
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

        # 3) Camera Controls Group
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QHBoxLayout()
        camera_group.setLayout(camera_layout)
        camera_layout.addWidget(QLabel("Exposure (ms):"))
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 100)  # Represents 0.01 to 1.00 ms
        self.exposure_slider.setValue(10)      # default = 0.10 ms
        self.exposure_slider.valueChanged.connect(self.update_exposure_from_slider)
        camera_layout.addWidget(self.exposure_slider)
        self.exposure_edit = QLineEdit("0.10")
        self.exposure_edit.setFixedWidth(60)
        self.exposure_edit.editingFinished.connect(self.update_slider_from_exposure)
        camera_layout.addWidget(self.exposure_edit)
        layout.addWidget(camera_group)

        # 4) Capture/Save Buttons
        self.capture_button = QPushButton("Auto Capture")
        self.capture_button.clicked.connect(self.auto_capture_clicked)
        layout.addWidget(self.capture_button)
        self.save_button = QPushButton("Save Current Photo")
        self.save_button.clicked.connect(self.save_current_photo_clicked)
        layout.addWidget(self.save_button)

        layout.addStretch()

    # -----------------------------
    # White LED toggle event handler
    # -----------------------------
    @pyqtSlot(int)
    def white_led_toggled(self, state):
        if state == Qt.Checked:
            # When enabling, set the white LED intensity first then turn on the LED.
            try:
                intensity = int(self.white_intensity_edit.text())
            except ValueError:
                intensity = self.white_intensity_slider.value()
                self.white_intensity_edit.setText(str(intensity))
            self.hardware.set_white_led_intensity(intensity)
            self.hardware.turn_on_led()
        else:
            self.hardware.turn_off_led()

    # -----------------------------
    # LED intensity update methods for UV LED (placeholders)
    # -----------------------------
    def update_uv_intensity_from_slider(self, slider_value):
        self.uv_intensity_edit.setText(str(slider_value))
        # Placeholder for UV LED control; not implemented yet.
    def update_uv_slider_from_intensity(self):
        try:
            val = int(self.uv_intensity_edit.text())
        except ValueError:
            val = self.uv_intensity_slider.value()
            self.uv_intensity_edit.setText(str(val))
            return
        val = max(0, min(val, 100))
        self.uv_intensity_slider.setValue(val)
        # Placeholder for UV LED control; not implemented yet.

    # -----------------------------
    # LED intensity update methods for White LED
    # -----------------------------
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

    # -----------------------------
    # Camera exposure methods
    # -----------------------------
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

    # -----------------------------
    # Capture/Save event handlers
    # -----------------------------
    def auto_capture_clicked(self):
        self.hardware.capture_and_save_image()
    def save_current_photo_clicked(self):
        self.hardware.save_last_frame()


###############################################################################
# Main Window
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
            self.camera_thread = CameraWorker(self.hardware.camera)
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
