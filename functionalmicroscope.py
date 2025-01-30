import sys
import os
import time
import cv2
import numpy as np

from ctypes import cdll, c_uint32, c_int, c_double, byref, create_string_buffer

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QSpinBox, QGroupBox, QGridLayout, QCheckBox, QLineEdit
)

import pylablib as pll
from pylablib.devices import Thorlabs

###############################################################################
# MicroscopeHardware: integrates camera + LED code
###############################################################################
class MicroscopeHardware:
    """
    Integrates:
      - The Thorlabs camera (via pylablib)
      - The Thorlabs upSeries LED driver (via TLUP_64.dll)
    and provides methods to control exposure, turn LED on/off, and capture images.
    """

    def __init__(self):
        """Initialize hardware connections here."""
        # 1) Initialize LED hardware
        self.upHandle = None
        self._init_led_hardware()

        # 2) Initialize camera hardware
        self.camera = None
        self._init_camera_hardware()

    def _init_camera_hardware(self):
        """
        Initialize the Thorlabs camera using pylablib.
        Replace 'path/to/dlls' if needed, or remove if your environment is set.
        """
        # Point pylablib to the Thorlabs camera DLLs if needed
        pll.par["devices/dlls/thorlabs_tlcam"] = "path/to/dlls"

        # List available cameras
        cameras = Thorlabs.list_cameras_tlcam()
        if not cameras:
            print("No Thorlabs cameras found.")
            return

        # Connect to the first camera found
        self.camera = Thorlabs.ThorlabsTLCamera(serial=cameras[0])
        print(f"Connected to camera: {cameras[0]}")

        # Start the camera with a default exposure (or 50 ms, etc.)
        default_exposure = 50.0  # ms
        self.camera.set_exposure(default_exposure)
        print("Camera exposure set to", default_exposure, "ms")

    def _init_led_hardware(self):
        """
        Load the LED driver DLL, connect to the first device, and configure it.
        """
        dll_path = r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLUP_64.dll"
        try:
            lib = cdll.LoadLibrary(dll_path)
        except OSError:
            print(f"Failed to load LED DLL at {dll_path}. Check path.")
            return

        # Find devices
        deviceCount = c_uint32()
        lib.TLUP_findRsrc(0, byref(deviceCount))
        if deviceCount.value == 0:
            print("No upSeries devices found.")
            return
        print("Number of upSeries devices found:", deviceCount.value)

        # Read model & serial
        modelName = create_string_buffer(256)
        serialNumber = create_string_buffer(256)
        lib.TLUP_getRsrcInfo(0, 0, modelName, serialNumber, 0, 0)
        print("Connecting to this device:")
        print("Model name:", modelName.value.decode(), 
              ", Serial number:", serialNumber.value.decode())

        # Initialize device
        upName = create_string_buffer(256)
        lib.TLUP_getRsrcName(0, 0, upName)
        upHandle = c_int(0)
        res = lib.TLUP_init(upName.value, 0, 0, byref(upHandle))
        if res != 0:
            print("Failed to initialize LED device.")
            return

        # Store references for later use
        self.led_lib = lib
        self.upHandle = upHandle

        # Enable the use of non-Thorlabs or unmounted LEDs
        self.led_lib.TLUP_setLedUseNonThorlabsLed(self.upHandle, 1)

        # Make sure the LED is off
        self.led_lib.TLUP_switchLedOutput(self.upHandle, 0)

        # Attempt to read LED info (works only if it has EEPROM)
        LEDName = create_string_buffer(256)
        LEDSerialNumber = create_string_buffer(256)
        LEDCurrentLimit = c_double()
        LEDForwardVoltage = c_double()
        LEDWavelength = c_double(0)

        self.led_lib.TLUP_getLedInfo(self.upHandle, LEDName, LEDSerialNumber,
                                     byref(LEDCurrentLimit),
                                     byref(LEDForwardVoltage),
                                     byref(LEDWavelength))

        # If the LED has EEPROM data (Thorlabs mounted LED)
        if LEDWavelength.value != 0:
            print("EEPROM on LED detected.")
            print("Connected LED:", LEDName.value.decode(), 
                  ", serial:", LEDSerialNumber.value.decode(),
                  ", wavelength [nm]:", LEDWavelength.value)
            print("Max LED current [A]:", LEDCurrentLimit.value, 
                  ", max forward voltage [V]:", LEDForwardVoltage.value)

            # Example: set current to 10% of limit
            currentSetpoint = c_double(0.1 * LEDCurrentLimit.value)
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, currentSetpoint)
            time.sleep(1)
            print("LED current set to 10% of limit.")
        else:
            print("No EEPROM on LED detected.")
            print("Defaulting to 1mA current.")
            currentSetpoint = c_double(0.001)  # 1 mA
            self.led_lib.TLUP_setLedCurrentSetpoint(self.upHandle, currentSetpoint)

        print("LED init complete.\n")

    def start_camera(self):
        """Begin continuous acquisition if supported by the driver."""
        if self.camera:
            self.camera.start_acquisition()
            print("Camera acquisition started.")

    def stop_camera(self):
        """Stop the camera acquisition."""
        if self.camera:
            self.camera.stop_acquisition()
            print("Camera acquisition stopped.")

    def get_frame(self):
        """
        Retrieve the latest frame from the camera as a numpy array (H, W, 3).
        If the camera is monochrome, you might get (H, W, 1) or just (H, W).
        """
        if not self.camera:
            return None
        frame_list = self.camera.grab(1)
        if not frame_list:
            return None

        image = frame_list[0]
        image_array = np.array(image)

        # Convert to RGB if needed or at least 3-channel for display
        if len(image_array.shape) == 2:
            # Monochrome: Expand to 3 channels for QImage.Format_RGB888
            image_array = np.stack((image_array,)*3, axis=-1)

        return image_array

    def set_exposure(self, exposure_ms):
        """Set camera exposure in milliseconds via the GUI slider."""
        if self.camera:
            self.camera.set_exposure(float(exposure_ms))
            print(f"Camera exposure set to {exposure_ms} ms")

    def move_stage(self, x=None, y=None, z=None):
        """
        Placeholder for stage movement. Replace with actual stage commands.
        """
        # If you have a real stage: call move_relative(x, y, z) or move_absolute().
        print(f"[STAGE] Moving: X={x}, Y={y}, Z={z} (placeholder)")

    def turn_on_led(self):
        """Turn on the white LED using the upSeries driver."""
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 1)
            print("[LED] Turned on.")

    def turn_off_led(self):
        """Turn off the white LED."""
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 0)
            print("[LED] Turned off.")

    def capture_and_save_image(self):
        """
        Example method to:
          1) Turn LED on
          2) Capture a single frame
          3) Save to the Downloads folder
          4) Turn LED off
        """
        # Turn on LED
        self.turn_on_led()

        # Short delay (if needed) to ensure LED stable
        time.sleep(0.5)

        # Grab one frame
        frame = self.get_frame()
        if frame is not None:
            # Convert to uint8 for saving
            if frame.max() > 0:
                frame_uint8 = (frame / frame.max() * 255).astype('uint8')
            else:
                frame_uint8 = frame.astype('uint8')

            # Path to downloads folder
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            image_path = os.path.join(downloads, "captured_image.png")
            cv2.imwrite(image_path, frame_uint8)
            print(f"[CAPTURE] Saved image to {image_path}")
        else:
            print("[CAPTURE] No frame received from camera!")

        # Turn off LED
        self.turn_off_led()

    def close_hardware(self):
        """Close camera and LED gracefully."""
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
        height, width, channel = frame.shape
        bytes_per_line = channel * width

        # Create QImage
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # Convert to QPixmap and scale
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

        # Stage Controls
        stage_group = QGroupBox("Stage Controls")
        stage_layout = QGridLayout()
        stage_group.setLayout(stage_layout)

        self.x_step_spin = QSpinBox()
        self.x_step_spin.setValue(1)
        self.x_step_spin.setRange(-1000, 1000)
        self.x_step_spin.setSuffix(" µm")

        self.y_step_spin = QSpinBox()
        self.y_step_spin.setValue(1)
        self.y_step_spin.setRange(-1000, 1000)
        self.y_step_spin.setSuffix(" µm")

        self.z_step_spin = QSpinBox()
        self.z_step_spin.setValue(1)
        self.z_step_spin.setRange(-1000, 1000)
        self.z_step_spin.setSuffix(" µm")

        move_x_button = QPushButton("Move X")
        move_x_button.clicked.connect(self.move_x_clicked)
        move_y_button = QPushButton("Move Y")
        move_y_button.clicked.connect(self.move_y_clicked)
        move_z_button = QPushButton("Move Z")
        move_z_button.clicked.connect(self.move_z_clicked)

        stage_layout.addWidget(QLabel("X step:"), 0, 0)
        stage_layout.addWidget(self.x_step_spin, 0, 1)
        stage_layout.addWidget(move_x_button, 0, 2)

        stage_layout.addWidget(QLabel("Y step:"), 1, 0)
        stage_layout.addWidget(self.y_step_spin, 1, 1)
        stage_layout.addWidget(move_y_button, 1, 2)

        stage_layout.addWidget(QLabel("Z step:"), 2, 0)
        stage_layout.addWidget(self.z_step_spin, 2, 1)
        stage_layout.addWidget(move_z_button, 2, 2)

        layout.addWidget(stage_group)

        # LED Controls (Simple On/Off for White LED)
        led_group = QGroupBox("White LED Controls")
        led_layout = QHBoxLayout()
        led_group.setLayout(led_layout)
        self.led_checkbox = QCheckBox("LED On/Off")
        self.led_checkbox.stateChanged.connect(self.white_led_toggled)
        led_layout.addWidget(self.led_checkbox)
        layout.addWidget(led_group)

        # Camera Controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QHBoxLayout()
        camera_group.setLayout(camera_layout)

        camera_layout.addWidget(QLabel("Exposure (ms):"))
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 1000)
        self.exposure_slider.setValue(50)
        self.exposure_slider.valueChanged.connect(self.exposure_changed)
        camera_layout.addWidget(self.exposure_slider)

        layout.addWidget(camera_group)

        # Capture button (turn on LED, snap one photo, save)
        self.capture_button = QPushButton("Capture & Save")
        self.capture_button.clicked.connect(self.capture_button_clicked)
        layout.addWidget(self.capture_button)

        layout.addStretch()

    @pyqtSlot()
    def move_x_clicked(self):
        step = self.x_step_spin.value()
        self.hardware.move_stage(x=step, y=None, z=None)

    @pyqtSlot()
    def move_y_clicked(self):
        step = self.y_step_spin.value()
        self.hardware.move_stage(x=None, y=step, z=None)

    @pyqtSlot()
    def move_z_clicked(self):
        step = self.z_step_spin.value()
        self.hardware.move_stage(x=None, y=None, z=step)

    @pyqtSlot(int)
    def exposure_changed(self, value):
        self.hardware.set_exposure(value)

    @pyqtSlot(int)
    def white_led_toggled(self, state):
        if state == Qt.Checked:
            self.hardware.turn_on_led()
        else:
            self.hardware.turn_off_led()

    @pyqtSlot()
    def capture_button_clicked(self):
        """
        Turns on LED, captures one frame, saves image, turns off LED.
        """
        self.hardware.capture_and_save_image()


###############################################################################
# Main Window
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Control UI")

        # Initialize hardware
        self.hardware = MicroscopeHardware()

        # If camera/LED were successfully initialized, start camera
        self.hardware.start_camera()

        # Central widget layout
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

        # Timer for live camera updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(100)  # 10 fps

    def closeEvent(self, event):
        """Cleanup hardware on exit."""
        self.hardware.stop_camera()
        self.hardware.close_hardware()
        super().closeEvent(event)

    def update_camera(self):
        """Grab current frame and display it."""
        frame = self.hardware.get_frame()
        self.camera_display.update_frame(frame)

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