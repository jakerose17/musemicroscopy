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
# Add references to the required Kinesis .NET assemblies
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.DCServoCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI, DeviceConfiguration, SimulationManager
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
from System import Decimal

# ------------------------------
# PyQt imports
# ------------------------------
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QSpinBox, QGroupBox, QGridLayout, QCheckBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage

# ------------------------------
# pylablib / camera imports
# ------------------------------
import pylablib as pll
from pylablib.devices import Thorlabs
from ctypes import *

###############################################################################
# MotionController: Manages 3 Axes with Kinesis
###############################################################################
class MotionController:
    """
    Controls 3 Thorlabs KCube DC Servos via the Kinesis .NET API.
    Axis -> Serial Number:
      X: 27601188
      Y: 27601181
      Z: 27265400

    Connects to each device, homes it, and provides move methods.
    """

    def __init__(self):
        """
        Initialize the Kinesis environment and connect to each axis.
        """
        self.axis_info = {
            "X": "27601188",
            "Y": "27601181",
            "Z": "27265400",
        }
        # Dictionary of {axis_letter: KCubeDCServo instance}
        self.devices = {}

        self._init_devices()

    def _init_devices(self):
        """
        Build device list, connect to each axis, enable, and home.
        """
        # SimulationManager.Instance.InitializeSimulations()
        # (Uncomment if you need simulation mode)

        DeviceManagerCLI.BuildDeviceList()
        print("Kinesis devices found:", list(DeviceManagerCLI.GetDeviceList()))

        # Connect and home each axis
        for axis, serial_no in self.axis_info.items():
            device = KCubeDCServo.CreateKCubeDCServo(serial_no)
            device.Connect(serial_no)
            time.sleep(0.25)

            device.StartPolling(250)
            time.sleep(0.25)

            device.EnableDevice()
            time.sleep(0.25)

            # Wait for settings
            if not device.IsSettingsInitialized():
                device.WaitForSettingsInitialized(10000)  # 10 s timeout
            if not device.IsSettingsInitialized():
                print(f"[{axis}] Settings failed to initialize!")
                continue

            # Load config
            m_config = device.LoadMotorConfiguration(serial_no,
                DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)
            # e.g., set device settings name if needed
            # m_config.DeviceSettingsName = "PRMTZ8"  # example from your snippet
            m_config.UpdateCurrentConfiguration()
            device.SetSettings(device.MotorDeviceSettings, True, False)

            print(f"Homing axis {axis} (serial {serial_no})...")
            device.Home(60000)  # 60 second timeout for homing
            print(f"[{axis}] Homing complete. Current pos:", device.Position)

            self.devices[axis] = device

    def move_axis_relative(self, axis, step_um):
        """
        Move the specified axis by 'step_um' micrometers relative to current.
        """
        if axis not in self.devices:
            print(f"[{axis}] Axis not initialized.")
            return
        device = self.devices[axis]
        current_pos = float(device.Position)  # read current position
        new_pos = current_pos + step_um
        print(f"[{axis}] Moving from {current_pos:.2f} to {new_pos:.2f} (µm)")
        # Convert to Decimal
        d = Decimal(new_pos)
        device.MoveTo(d, 20000)  # 20 s timeout
        time.sleep(0.1)
        print(f"[{axis}] Move done. Current pos: {device.Position}")

    def disconnect_all(self):
        """
        Disconnect from all devices (call before program exit).
        """
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
    Emits `new_frame` whenever a frame arrives.
    """
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.running = True

    def run(self):
        while self.running:
            frame_list = self.camera.grab(1)  # blocks until a frame is ready
            if frame_list:
                frame = np.array(frame_list[0])

                # Basic scaling for display
                min_val = frame.min()
                max_val = frame.max()
                scaled = (frame - min_val) / (max_val - min_val + 1e-6) * 255
                display_frame = scaled.astype(np.uint8)

                self.new_frame.emit(display_frame)
            # optional: time.sleep(0.01)

    def stop(self):
        self.running = False
        self.wait()


###############################################################################
# MicroscopeHardware: Combines LED, Camera, and Motion
###############################################################################
class MicroscopeHardware:
    """
    - Thorlabs camera (via pylablib)
    - Thorlabs upSeries LED driver (TLUP_64.dll)
    - A MotionController instance for X,Y,Z stage
    """
    def __init__(self):
        self.upHandle = None
        self.led_lib = None
        self.camera = None
        self.last_frame = None  # store the latest frame from the worker

        # 1) Initialize motion
        self.motion_controller = MotionController()

        # 2) Initialize LED
        self._init_led_hardware()

        # 3) Initialize camera
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

        # Turn off LED initially
        self.led_lib.TLUP_setLedUseNonThorlabsLed(self.upHandle, 1)
        self.led_lib.TLUP_switchLedOutput(self.upHandle, 0)

        # Attempt to read LED info
        LEDName = create_string_buffer(256)
        LEDSerialNumber = create_string_buffer(256)
        LEDCurrentLimit = c_double()
        LEDForwardVoltage = c_double()
        LEDWavelength = c_double(0)

        self.led_lib.TLUP_getLedInfo(
            self.upHandle, LEDName, LEDSerialNumber,
            byref(LEDCurrentLimit),
            byref(LEDForwardVoltage),
            byref(LEDWavelength)
        )

        if LEDWavelength.value != 0:
            print("EEPROM on LED detected.")
            currentSetpoint = c_double(0.1 * LEDCurrentLimit.value)
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, currentSetpoint)
            time.sleep(1)
            print("LED current set to 10% of limit.")
        else:
            print("No EEPROM on LED detected. Defaulting to 1mA current.")
            currentSetpoint = c_double(0.001)
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, currentSetpoint)

        print("LED init complete.\n")

    # -------------------------------------------------------------------------
    # Camera methods
    # -------------------------------------------------------------------------
    def start_camera(self):
        if self.camera:
            self.camera.start_acquisition()
            print("Camera acquisition started.")

    def stop_camera(self):
        if self.camera:
            self.camera.stop_acquisition()
            print("Camera acquisition stopped.")

    def close_hardware(self):
        # Close camera
        if self.camera:
            self.camera.close()
            print("Camera closed.")
            self.camera = None

        # Close LED
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_close(self.upHandle)
            self.upHandle = None
            print("LED device closed.")

        # Disconnect motion
        if self.motion_controller:
            self.motion_controller.disconnect_all()

    def set_exposure(self, exposure_ms):
        if self.camera:
            self.camera.set_exposure(float(exposure_ms))
            print(f"Camera exposure set to {exposure_ms} ms")

    # -------------------------------------------------------------------------
    # Motion methods (delegate to MotionController)
    # -------------------------------------------------------------------------
    def move_stage(self, x=None, y=None, z=None):
        """
        For demonstration, if user sets x=+10 => move X by +10 µm, etc.
        Only moves axes that are not None.
        """
        if x is not None:
            self.motion_controller.move_axis_relative("X", x)
        if y is not None:
            self.motion_controller.move_axis_relative("Y", y)
        if z is not None:
            self.motion_controller.move_axis_relative("Z", z)

    # -------------------------------------------------------------------------
    # LED methods
    # -------------------------------------------------------------------------
    def turn_on_led(self):
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 1)
            print("[LED] Turned on.")

    def turn_off_led(self):
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 0)
            print("[LED] Turned off.")

    # -------------------------------------------------------------------------
    # Capture methods
    # -------------------------------------------------------------------------
    def capture_and_save_image(self):
        """
        Auto Capture:
          1) Turn LED on
          2) Wait
          3) Grab single frame
          4) Save
          5) Turn LED off
        """
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
            # Expand grayscale => 3 channels
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

        # 1) Stage controls group
        stage_group = QGroupBox("Stage Controls")
        stage_layout = QGridLayout()
        stage_group.setLayout(stage_layout)

        self.step_spin = QSpinBox()
        self.step_spin.setValue(10)
        self.step_spin.setRange(1, 1000)
        self.step_spin.setSuffix(" µm")

        # Buttons for each axis + and -
        self.x_pos_button = QPushButton("X +")
        self.x_neg_button = QPushButton("X -")
        self.y_pos_button = QPushButton("Y +")
        self.y_neg_button = QPushButton("Y -")
        self.z_pos_button = QPushButton("Z +")
        self.z_neg_button = QPushButton("Z -")

        self.x_pos_button.clicked.connect(self.x_pos_clicked)
        self.x_neg_button.clicked.connect(self.x_neg_clicked)
        self.y_pos_button.clicked.connect(self.y_pos_clicked)
        self.y_neg_button.clicked.connect(self.y_neg_clicked)
        self.z_pos_button.clicked.connect(self.z_pos_clicked)
        self.z_neg_button.clicked.connect(self.z_neg_clicked)

        # Layout arrangement
        # Row 0: step label + spin
        stage_layout.addWidget(QLabel("Step size:"), 0, 0)
        stage_layout.addWidget(self.step_spin,     0, 1)

        # Row 1: X+ and X-
        stage_layout.addWidget(self.x_pos_button, 1, 0)
        stage_layout.addWidget(self.x_neg_button, 1, 1)

        # Row 2: Y+ and Y-
        stage_layout.addWidget(self.y_pos_button, 2, 0)
        stage_layout.addWidget(self.y_neg_button, 2, 1)

        # Row 3: Z+ and Z-
        stage_layout.addWidget(self.z_pos_button, 3, 0)
        stage_layout.addWidget(self.z_neg_button, 3, 1)

        layout.addWidget(stage_group)

        # 2) LED Controls (same as before)
        led_group = QGroupBox("LED Controls")
        led_layout = QGridLayout()
        led_group.setLayout(led_layout)

        self.uv_led_checkbox = QCheckBox("UV LED")
        self.uv_intensity_edit = QLineEdit("255")
        self.uv_intensity_edit.setFixedWidth(50)
        self.uv_led_checkbox.stateChanged.connect(self.uv_led_toggled)

        self.white_led_checkbox = QCheckBox("White LED")
        self.white_intensity_edit = QLineEdit("255")
        self.white_intensity_edit.setFixedWidth(50)
        self.white_led_checkbox.stateChanged.connect(self.white_led_toggled)

        led_layout.addWidget(self.uv_led_checkbox, 0, 0)
        led_layout.addWidget(QLabel("Intensity:"), 0, 1)
        led_layout.addWidget(self.uv_intensity_edit, 0, 2)

        led_layout.addWidget(self.white_led_checkbox, 1, 0)
        led_layout.addWidget(QLabel("Intensity:"), 1, 1)
        led_layout.addWidget(self.white_intensity_edit, 1, 2)  # Or self.white_intensity_edit

        layout.addWidget(led_group)

        # 3) Camera Controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QHBoxLayout()
        camera_group.setLayout(camera_layout)

        camera_layout.addWidget(QLabel("Exposure (ms):"))

        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 100)  # 0.01..1.0 ms
        self.exposure_slider.setValue(10)      # default 0.10 ms
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
    # Stage event handlers
    # -----------------------------
    def x_pos_clicked(self):
        step = self.step_spin.value()
        self.hardware.move_stage(x=step)

    def x_neg_clicked(self):
        step = self.step_spin.value()
        self.hardware.move_stage(x=-step)

    def y_pos_clicked(self):
        step = self.step_spin.value()
        self.hardware.move_stage(y=step)

    def y_neg_clicked(self):
        step = self.step_spin.value()
        self.hardware.move_stage(y=-step)

    def z_pos_clicked(self):
        step = self.step_spin.value()
        self.hardware.move_stage(z=step)

    def z_neg_clicked(self):
        step = self.step_spin.value()
        self.hardware.move_stage(z=-step)

    # -----------------------------
    # LED event handlers
    # -----------------------------
    @pyqtSlot(int)
    def uv_led_toggled(self, state):
        # Example placeholder
        if state == Qt.Checked:
            self.hardware.turn_on_led()
        else:
            self.hardware.turn_off_led()

    @pyqtSlot(int)
    def white_led_toggled(self, state):
        # Example placeholder
        if state == Qt.Checked:
            self.hardware.turn_on_led()
        else:
            self.hardware.turn_off_led()

    # -----------------------------
    # Camera exposure
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

        if entered_value < 0.01:
            entered_value = 0.01
        elif entered_value > 1.0:
            entered_value = 1.0

        slider_val = int(round(entered_value * 100))
        self.exposure_slider.setValue(slider_val)
        self.hardware.set_exposure(entered_value)

    # -----------------------------
    # Capture/Save
    # -----------------------------
    @pyqtSlot()
    def auto_capture_clicked(self):
        self.hardware.capture_and_save_image()

    @pyqtSlot()
    def save_current_photo_clicked(self):
        self.hardware.save_last_frame()


###############################################################################
# Main Window
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Control UI")

        # 1) Initialize hardware
        self.hardware = MicroscopeHardware()
        self.hardware.start_camera()

        # 2) Build UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Camera display
        self.camera_display = CameraDisplayWidget()
        main_layout.addWidget(self.camera_display, stretch=2)

        # Control panel
        self.control_panel = ControlPanelWidget(self.hardware)
        main_layout.addWidget(self.control_panel, stretch=1)

        # 3) Worker thread for camera
        if self.hardware.camera:
            self.camera_thread = CameraWorker(self.hardware.camera)
            self.camera_thread.new_frame.connect(self.on_new_frame)
            self.camera_thread.start()
        else:
            self.camera_thread = None

    @pyqtSlot(np.ndarray)
    def on_new_frame(self, frame):
        """
        Called whenever the worker has a new frame.
        Update the display, store in hardware.last_frame.
        """
        self.hardware.last_frame = frame
        self.camera_display.update_frame(frame)

    def closeEvent(self, event):
        """
        Clean up the worker thread and hardware on exit.
        """
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
