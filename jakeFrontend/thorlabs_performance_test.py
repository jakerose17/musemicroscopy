"""
Camera Performance Test - PyQt5

This example shows how to create a live image viewer using PyQt5 and measure camera performance.
It uses the ThorLabs Python SDK for camera control and numpy for image processing.
Supports both monochrome and color cameras.

"""

import sys
import time
import numpy as np
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from datetime import datetime

# Configure DLL path
def configure_path():
    is_64bits = sys.maxsize > 2**32
    relative_path_to_dlls = os.path.join('Python Toolkit', 'dlls')
    
    if is_64bits:
        relative_path_to_dlls = os.path.join(relative_path_to_dlls, '64_lib')
    else:
        relative_path_to_dlls = os.path.join(relative_path_to_dlls, '32_lib')
    
    absolute_path_to_dlls = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path_to_dlls))
    
    if os.path.exists(absolute_path_to_dlls):
        os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ['PATH']
        try:
            # Python 3.8 introduces a new method to specify dll directory
            os.add_dll_directory(absolute_path_to_dlls)
        except AttributeError:
            pass
        print(f"Added DLL path: {absolute_path_to_dlls}")
    else:
        print(f"ERROR: DLL path does not exist: {absolute_path_to_dlls}")
        sys.exit(1)

# Configure path before importing SDK
configure_path()

try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
    from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
    from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK
    print("Successfully imported ThorLabs SDK")
except ImportError as e:
    print(f"Error importing ThorLabs SDK: {e}")
    print("Please make sure the ThorLabs Python SDK is installed and DLLs are in the correct location")
    sys.exit(1)

class CameraThread(QThread):
    """Thread for continuous image acquisition"""
    new_frame = pyqtSignal(np.ndarray)
    fps_update = pyqtSignal(float)
    raw_frame = pyqtSignal(np.ndarray)  # New signal for raw frame data

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = False
        self.frame_count = 0
        self.start_time = 0
        
        # Setup color processing if necessary
        if self.camera.camera_sensor_type != SENSOR_TYPE.BAYER:
            # Sensor type is not compatible with the color processing library
            self._is_color = False
            print("Camera is monochrome")
        else:
            self._mono_to_color_sdk = MonoToColorProcessorSDK()
            self._image_width = self.camera.image_width_pixels
            self._image_height = self.camera.image_height_pixels
            self._mono_to_color_processor = self._mono_to_color_sdk.create_mono_to_color_processor(
                SENSOR_TYPE.BAYER,
                self.camera.color_filter_array_phase,
                self.camera.get_color_correction_matrix(),
                self.camera.get_default_white_balance_matrix(),
                self.camera.bit_depth
            )
            self._is_color = True
            print("Camera is color (Bayer pattern)")

    def _get_color_image(self, frame):
        """Convert Bayer pattern to RGB image"""
        # verify the image size
        width = frame.image_buffer.shape[1]
        height = frame.image_buffer.shape[0]
        if (width != self._image_width) or (height != self._image_height):
            self._image_width = width
            self._image_height = height
            print("Image dimension change detected, updating color processor")
            
        # Get the raw frame data first
        raw_data = frame.image_buffer.copy()
        
        # color the image for display. transform_to_24 will scale to 8 bits per channel
        color_image_data = self._mono_to_color_processor.transform_to_24(
            frame.image_buffer,
            self._image_width,
            self._image_height
        )
        color_image_data = color_image_data.reshape(self._image_height, self._image_width, 3)
        
        # Emit both the display image and the raw data
        self.raw_frame.emit(raw_data)
        return color_image_data

    def _get_mono_image(self, frame):
        """Convert monochrome image to 8-bit for display"""
        # Get the raw frame data first
        raw_data = frame.image_buffer.copy()
        
        # Scale down to 8-bit for display
        scaled_image = frame.image_buffer >> (self.camera.bit_depth - 8)
        
        # Emit both the display image and the raw data
        self.raw_frame.emit(raw_data)
        return scaled_image

    def run(self):
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()

        while self.running:
            try:
                frame = self.camera.get_pending_frame_or_null()
                if frame is not None:
                    # Convert frame to numpy array
                    if self._is_color:
                        array = self._get_color_image(frame)
                    else:
                        array = self._get_mono_image(frame)
                    
                    # Emit frame and update FPS
                    self.new_frame.emit(array)
                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    if elapsed >= 1.0:  # Update FPS every second
                        fps = self.frame_count / elapsed
                        self.fps_update.emit(fps)
                        self.frame_count = 0
                        self.start_time = time.time()
            except Exception as e:
                print(f"Error in camera thread: {e}")
                break

    def stop(self):
        self.running = False
        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thorlabs Camera Performance Test")
        self.resize(1200, 800)

        # Initialize variables
        self.camera_thread = None
        self.sdk = None
        self.camera = None
        self.current_frame = None  # Store the current frame
        self.current_raw_frame = None  # Store the raw frame data
        self.camera_bit_depth = None  # Store camera's native bit depth

        # Create UI elements
        self.init_ui()

        # Start camera
        self.start_camera()

    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Display widget
        self.display = QLabel("No Camera Connected")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setStyleSheet("background-color: black; color: white;")
        main_layout.addWidget(self.display)

        # Control panel
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        # Start/Stop button
        self.toggle_button = QPushButton("Stop Stream")
        self.toggle_button.clicked.connect(self.toggle_streaming)
        control_layout.addWidget(self.toggle_button)

        # Save button
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_current_frame)
        control_layout.addWidget(self.save_button)

        # FPS display
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(QFont("Arial", 14))
        control_layout.addWidget(self.fps_label)

    def start_camera(self):
        try:
            # Initialize SDK and camera
            self.sdk = TLCameraSDK()
            camera_list = self.sdk.discover_available_cameras()
            if not camera_list:
                self.display.setText("No cameras detected")
                return

            self.camera = self.sdk.open_camera(camera_list[0])
            print(f"Opened camera: {self.camera.name}")
            
            # Get and store camera bit depth
            self.camera_bit_depth = self.camera.bit_depth
            print(f"Camera bit depth: {self.camera_bit_depth} bits")
            
            # Print camera sensor type
            sensor_type = self.camera.camera_sensor_type
            print(f"Camera sensor type: {sensor_type}")

            # Configure camera for performance
            self.camera.frames_per_trigger_zero_for_unlimited = 0  # Continuous mode
            self.camera.exposure_time_us = 1000  # 1ms exposure
            self.camera.arm(2)
            self.camera.issue_software_trigger()

            # Start camera thread
            self.camera_thread = CameraThread(self.camera)
            self.camera_thread.new_frame.connect(self.update_frame)
            self.camera_thread.raw_frame.connect(self.update_raw_frame)
            self.camera_thread.fps_update.connect(self.update_fps)
            self.camera_thread.start()
            self.toggle_button.setText("Stop Stream")

        except Exception as e:
            print(f"Error starting camera: {e}")
            self.display.setText("Failed to initialize camera")

    def update_frame(self, frame):
        # Store the current frame
        self.current_frame = frame.copy()
        
        # Convert frame to QImage for display
        if len(frame.shape) == 2:  # Monochrome
            h, w = frame.shape
            bytes_per_line = w
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color
            h, w, channels = frame.shape
            bytes_per_line = channels * w
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to pixmap and display
        pixmap = QPixmap.fromImage(img)
        self.display.setPixmap(pixmap.scaled(self.display.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def update_raw_frame(self, frame):
        """Store the raw frame data with full bit depth"""
        self.current_raw_frame = frame.copy()

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def toggle_streaming(self):
        if self.camera_thread and self.camera_thread.running:
            self.camera_thread.stop()
            self.toggle_button.setText("Start Stream")
        else:
            if self.camera_thread:
                self.camera_thread.start()
                self.toggle_button.setText("Stop Stream")
            else:
                self.start_camera()

    def save_current_frame(self):
        if self.current_frame is None:
            return

        # Get user's downloads folder
        downloads_folder = os.path.expanduser("~/Downloads")
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_capture_{timestamp}.png"  # Changed to PNG for better 8-bit support
        filepath = os.path.join(downloads_folder, filename)

        try:
            # Save the 8-bit display image
            if len(self.current_frame.shape) == 2:  # Monochrome
                cv2.imwrite(filepath, self.current_frame)
            else:  # Color
                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, bgr_image)
            
            print(f"Image saved to: {filepath}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if self.camera:
            self.camera.dispose()
        if self.sdk:
            self.sdk.dispose()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
