import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QSpinBox, QGroupBox, QGridLayout, QCheckBox, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

###############################################################################
# Mock/Placeholder Hardware Controller
###############################################################################
class MicroscopeHardware:
    """
    A placeholder class for controlling:
      - Motion stage (X, Y, Z)
      - Camera (exposure, frames)
      - LEDs (on/off, intensity)
    Replace these methods with your real hardware interactions.
    """

    def __init__(self):
        # Initialize any hardware connections here
        # e.g., self.camera = ...
        #       self.stage = ...
        #       self.uv_led = ...
        #       self.white_led = ...
        pass

    def start_camera(self):
        """Start continuous camera capture (if needed)."""
        # For example: self.camera.start_capture()
        pass

    def stop_camera(self):
        """Stop camera capture."""
        # For example: self.camera.stop_capture()
        pass

    def get_frame(self):
        """
        Retrieve the latest frame from the camera.
        For demonstration, returns a random noise image.
        In your code, return a numpy array from your ThorCam or other camera driver.
        """
        # For a real camera, you might do:
        #   frame = self.camera.get_frame()
        #   return frame
        # Here, we just simulate with random noise:
        random_image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        return random_image

    def set_exposure(self, exposure_ms):
        """Set camera exposure in milliseconds."""
        # e.g., self.camera.set_exposure(exposure_ms)
        print(f"Setting camera exposure to: {exposure_ms} ms")

    def move_stage(self, x=None, y=None, z=None):
        """
        Move stage to an absolute position or incrementally.
        If you have separate methods for relative vs absolute moves,
        adapt accordingly.
        """
        # e.g., self.stage.move_absolute(x, y, z) or move_relative(...)
        print(f"Moving stage to: X={x}, Y={y}, Z={z}")

    def turn_on_led(self, led_type, intensity=255):
        """Turn on the specified LED (UV or White) with given intensity."""
        # e.g., if led_type == 'UV': self.uv_led.set_intensity(intensity)
        #       if led_type == 'White': self.white_led.set_intensity(intensity)
        print(f"Turning on {led_type} LED with intensity {intensity}")

    def turn_off_led(self, led_type):
        """Turn off the specified LED."""
        # e.g., if led_type == 'UV': self.uv_led.off()
        #       if led_type == 'White': self.white_led.off()
        print(f"Turning off {led_type} LED")

    def capture_image_sequence(self):
        """
        Example of an 'automatic capture' that might:
         1. Move stage to certain positions
         2. Adjust camera settings
         3. Turn on LED
         4. Capture images
         5. Turn off LED
         6. Return results or save images
        """
        print("Starting automatic capture sequence...")
        # Put your logic here
        # ...
        print("Capture complete.")


###############################################################################
# Camera Display Widget
###############################################################################
class CameraDisplayWidget(QLabel):
    """
    A QLabel that displays the current camera frame.
    We'll update this regularly via a timer in the main window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        # Optionally set a fixed minimum size or use a layout that resizes nicely
        self.setMinimumSize(640, 480)

    def update_frame(self, frame):
        """
        Update the displayed image.
        :param frame: A numpy array with shape (H, W, 3).
        """
        if frame is None:
            return

        # Convert the numpy array to QImage, then to QPixmap for display
        height, width, channel = frame.shape
        bytes_per_line = channel * width

        # Create QImage from the numpy data (BGR or RGB depends on your camera data)
        # For demonstration, assume it's RGB as in our random_image generator:
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(qimg)

        # Scale pixmap to fit our label size if desired
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


###############################################################################
# Control Panel Widget
###############################################################################
class ControlPanelWidget(QWidget):
    """
    A panel with manual controls for the stage, LEDs, and camera settings.
    Also includes an 'Auto Capture' button.
    We'll emit signals or call methods directly on the hardware.
    """

    def __init__(self, hardware: MicroscopeHardware, parent=None):
        super().__init__(parent)
        self.hardware = hardware

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # GroupBox for stage controls
        stage_group = QGroupBox("Stage Controls")
        stage_layout = QGridLayout()
        stage_group.setLayout(stage_layout)

        # Example X/Y step boxes & move buttons
        self.x_step_spin = QSpinBox()
        self.x_step_spin.setValue(1)
        self.x_step_spin.setRange(-1000, 1000)
        self.x_step_spin.setSuffix(" µm")

        self.y_step_spin = QSpinBox()
        self.y_step_spin.setValue(1)
        self.y_step_spin.setRange(-1000, 1000)
        self.y_step_spin.setSuffix(" µm")

        self.z_step_spin = QSpinBox()
        self.z_step_spin.setValue(1)          # default step size (e.g., 1 µm)
        self.z_step_spin.setRange(-1000, 1000)  # allowed range (adjust as needed)
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

        # GroupBox for LED controls
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

        # GroupBox for Camera controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QHBoxLayout()
        camera_group.setLayout(camera_layout)

        camera_layout.addWidget(QLabel("Exposure (ms):"))
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 1000)  # example range 1-1000 ms
        self.exposure_slider.setValue(50)
        self.exposure_slider.valueChanged.connect(self.exposure_changed)
        camera_layout.addWidget(self.exposure_slider)

        layout.addWidget(camera_group)

        # Auto Capture button
        self.capture_button = QPushButton("Auto Capture")
        self.capture_button.clicked.connect(self.auto_capture_clicked)
        layout.addWidget(self.capture_button)

        # Stretch to consume remaining vertical space
        layout.addStretch()

    @pyqtSlot()
    def move_x_clicked(self):
        step = self.x_step_spin.value()
        # Example: move stage relative in X
        self.hardware.move_stage(x=step, y=None, z=None)

    @pyqtSlot()
    def move_y_clicked(self):
        step = self.y_step_spin.value()
        # Example: move stage relative in Y
        self.hardware.move_stage(x=None, y=step, z=None)

    @pyqtSlot()
    def move_z_clicked(self):
        step = self.z_step_spin.value()
        # Example: move stage relative in Z
        self.hardware.move_stage(x=None, y=None, z=step)

    @pyqtSlot(int)
    def exposure_changed(self, value):
        self.hardware.set_exposure(value)

    @pyqtSlot(int)
    def uv_led_toggled(self, state):
        if state == Qt.Checked:
            intensity = int(self.uv_intensity_edit.text())
            self.hardware.turn_on_led('UV', intensity)
        else:
            self.hardware.turn_off_led('UV')

    @pyqtSlot(int)
    def white_led_toggled(self, state):
        if state == Qt.Checked:
            intensity = int(self.white_intensity_edit.text())
            self.hardware.turn_on_led('White', intensity)
        else:
            self.hardware.turn_off_led('White')

    @pyqtSlot()
    def auto_capture_clicked(self):
        self.hardware.capture_image_sequence()


###############################################################################
# Main Window
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Control UI")

        # Initialize hardware controller
        self.hardware = MicroscopeHardware()
        self.hardware.start_camera()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout: horizontal split (left: camera view, right: controls)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Camera display
        self.camera_display = CameraDisplayWidget()
        main_layout.addWidget(self.camera_display, stretch=2)

        # Control panel
        self.control_panel = ControlPanelWidget(self.hardware)
        main_layout.addWidget(self.control_panel, stretch=1)

        # Set up a timer for live view updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(100)  # update ~10 frames/sec (100 ms interval)

    def closeEvent(self, event):
        """Make sure we stop camera and clean up hardware on exit."""
        self.hardware.stop_camera()
        super().closeEvent(event)

    def update_camera(self):
        """Grab the latest frame from hardware and display it."""
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
