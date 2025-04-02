import sys
import os
import time
import cv2
import numpy as np

from ctypes import cdll, c_uint32, c_int, c_double, byref, create_string_buffer

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QSpinBox, QGroupBox, QGridLayout, QCheckBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage

import pylablib as pll
from pylablib.devices import Thorlabs


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
        # Continuous loop to grab frames
        while self.running:
            frame_list = self.camera.grab(1)  # blocks until a frame is ready
            if frame_list:
                frame = np.array(frame_list[0])
                min_val = frame.min()
                max_val = frame.max()
                scaled = (frame - min_val) / (max_val - min_val + 1e-6) * 255
                display_frame = scaled.astype(np.uint8)
                self.new_frame.emit(display_frame)
            # Optional small sleep to reduce CPU usage if frame rate is high:
            # time.sleep(0.01)



    def stop(self):
        self.running = False
        self.wait()  # wait for thread to finish cleanly


###############################################################################
# MicroscopeHardware: Holds camera/LED logic and references last_frame
###############################################################################
class MicroscopeHardware:
    """
    Integrates:
      - The Thorlabs camera (via pylablib)
      - The Thorlabs upSeries LED driver (via TLUP_64.dll)
    Provides methods to control exposure, turn LED on/off, etc.
    Also stores the most recently received frame in `self.last_frame`.
    """

    def __init__(self):
        """Initialize hardware connections here."""
        self.upHandle = None
        self.camera = None
        self.last_frame = None  # store the latest frame from the worker

        # 1) Initialize LED hardware
        self._init_led_hardware()
        # 2) Initialize camera hardware
        self._init_camera_hardware()

    def _init_camera_hardware(self):
        """Initialize the Thorlabs camera using pylablib."""
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
        """Load the LED driver DLL, connect to the first device, and configure it."""
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
        print("Connecting to this device:")
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

        # Enable use of non-Thorlabs/unmounted LEDs
        self.led_lib.TLUP_setLedUseNonThorlabsLed(self.upHandle, 1)
        # Turn off LED initially
        self.led_lib.TLUP_switchLedOutput(self.upHandle, 0)

        # Try to read LED info (only works if there's an EEPROM)
        LEDName = create_string_buffer(256)
        LEDSerialNumber = create_string_buffer(256)
        LEDCurrentLimit = c_double()
        LEDForwardVoltage = c_double()
        LEDWavelength = c_double(0)
        self.led_lib.TLUP_getLedInfo(self.upHandle, LEDName, LEDSerialNumber,
                                     byref(LEDCurrentLimit),
                                     byref(LEDForwardVoltage),
                                     byref(LEDWavelength))
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
        """Start continuous acquisition (called before starting the worker)."""
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

    def set_exposure(self, exposure_ms):
        if self.camera:
            self.camera.set_exposure(float(exposure_ms))
            print(f"Camera exposure set to {exposure_ms} ms")

    def move_stage(self, x=None, y=None, z=None):
        """Placeholder for motorized stage movement."""
        print(f"[STAGE] Moving: X={x}, Y={y}, Z={z} (placeholder)")

    def turn_on_led(self):
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 1)
            print("[LED] Turned on.")

    def turn_off_led(self):
        if self.upHandle and self.led_lib:
            self.led_lib.TLUP_switchLedOutput(self.upHandle, 0)
            print("[LED] Turned off.")

    def capture_and_save_image(self):
        """
        Auto Capture:
          1) Turn LED on
          2) Wait briefly
          3) Grab a single frame (blocking)
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
        """
        Save the most recently received (live) frame from the worker
        without toggling the LED or waiting again.
        """
        if self.last_frame is None:
            print("[SAVE] No current frame available to save!")
            return
        self._save_frame_to_downloads(self.last_frame, "current_frame.png")

    def _save_frame_to_downloads(self, frame, filename):
        """
        Common utility method to scale, convert, and save a frame to the Downloads folder.
        """
        if frame.max() > 0:
            frame_uint8 = (frame / frame.max() * 255).astype('uint8')
        else:
            frame_uint8 = frame.astype('uint8')

        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        image_path = os.path.join(downloads, filename)
        cv2.imwrite(image_path, frame_uint8)
        print(f"[SAVE] Saved image to {image_path}")


###############################################################################
# Camera Display Widget
###############################################################################
class CameraDisplayWidget(QLabel):
    """
    A QLabel that displays the camera frame. Updated via signal from the worker.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)

    def update_frame(self, frame):
        if frame is None:
            return
        # Convert to 3 channels if needed
        if len(frame.shape) == 2:
            frame = np.stack((frame,)*3, axis=-1)

        height, width, channel = frame.shape
        bytes_per_line = channel * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


###############################################################################
# Control Panel Widget
###############################################################################
class ControlPanelWidget(QWidget):
    """
    Manual controls for stage, LEDs, camera, plus capture/save buttons.
    """
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

        # LED Controls
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
        led_layout.addWidget(self.white_intensity_edit, 1, 2)

        layout.addWidget(led_group)

        # Camera Controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QHBoxLayout()
        camera_group.setLayout(camera_layout)

        camera_layout.addWidget(QLabel("Exposure (ms):"))

        # Slider: range 0.01..1.0 ms (scaled by 100)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 100)  # 0.01..1.0
        self.exposure_slider.setValue(10)      # default=0.10 ms
        self.exposure_slider.valueChanged.connect(self.update_exposure_from_slider)
        camera_layout.addWidget(self.exposure_slider)

        # QLineEdit for direct exposure input
        self.exposure_edit = QLineEdit("0.10")
        self.exposure_edit.setFixedWidth(60)
        self.exposure_edit.editingFinished.connect(self.update_slider_from_exposure)
        camera_layout.addWidget(self.exposure_edit)

        layout.addWidget(camera_group)

        # Buttons: Auto Capture, Save Current
        self.capture_button = QPushButton("Auto Capture")
        self.capture_button.clicked.connect(self.auto_capture_clicked)
        layout.addWidget(self.capture_button)

        self.save_button = QPushButton("Save Current Photo")
        self.save_button.clicked.connect(self.save_current_photo_clicked)
        layout.addWidget(self.save_button)

        layout.addStretch()

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
    def uv_led_toggled(self, state):
        if state == Qt.Checked:
            # In real code, set UV intensity, etc.
            self.hardware.turn_on_led()
        else:
            self.hardware.turn_off_led()

    @pyqtSlot(int)
    def white_led_toggled(self, state):
        if state == Qt.Checked:
            # In real code, set White intensity, etc.
            self.hardware.turn_on_led()
        else:
            self.hardware.turn_off_led()

    @pyqtSlot()
    def auto_capture_clicked(self):
        """
        Calls hardware.capture_and_save_image().
        This toggles the LED, grabs one frame, saves it, turns off LED.
        """
        self.hardware.capture_and_save_image()

    @pyqtSlot()
    def save_current_photo_clicked(self):
        """
        Save the frame that is currently displayed (from the live view).
        """
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

        # 2) Set up central UI
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

        # 3) Create a separate worker thread to grab frames continuously
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
        """Clean up camera thread and hardware on exit."""
        # Stop background thread
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
