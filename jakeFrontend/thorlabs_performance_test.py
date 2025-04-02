import sys
import time
import numpy as np
import clr
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

# Try to import direct .NET references for Thorlabs
try:
    from System.Reflection import Assembly
    # Explicitly load the uc480DotNet assembly using its full file path
    thorlabs_dll = r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam\uc480DotNet.dll"
    Assembly.LoadFile(thorlabs_dll)


    # Add references to Thorlabs .NET libraries
    thorlabs_path = r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam"
    sys.path.append(thorlabs_path)
    clr.AddReference("uc480DotNet")

    # Import .NET namespaces
    from uc480 import uc480NET
    from uc480.Info import Camera
    from uc480.Defines import *
    from System import Array, Int32
    from System.Runtime.InteropServices import GCHandle, GCHandleType

    NET_AVAILABLE = True
    print("Thorlabs .NET API loaded successfully")
except Exception as e:
    NET_AVAILABLE = False
    print(f"Couldn't load Thorlabs .NET API: {e}")

# Try to import pylablib for comparison
try:
    import pylablib as pll
    from pylablib.devices import Thorlabs
    PYLABLIB_AVAILABLE = True
    print("Pylablib loaded successfully")
except ImportError:
    PYLABLIB_AVAILABLE = False
    print("Pylablib not available")

class ThorlabsDirectCamera:
    """Direct implementation using Thorlabs .NET API"""
    def __init__(self):
        self.cam = None
        self.mem_id = 0
        self.width = 0
        self.height = 0
        self.bit_depth = 8
        self.running = False

    def initialize(self):
        """Initialize the camera connection"""
        # Create camera instance
        self.cam = uc480NET.Camera()

        # Initialize camera
        ret = self.cam.Init()
        if ret != IS_SUCCESS:
            raise Exception(f"Camera initialization failed with error {ret}")

        # Get camera info
        info = Camera()
        ret = self.cam.Information.GetCameraInfo(info)
        if ret != IS_SUCCESS:
            raise Exception(f"Failed to get camera info: {ret}")

        print(f"Camera: {info.Model}, Serial: {info.SerialNumber}")

        # Get sensor info
        sensor_info = self.cam.Information.GetSensorInfo()
        self.width = sensor_info.MaxWidth
        self.height = sensor_info.MaxHeight
        print(f"Sensor size: {self.width}x{self.height}")

        # Allocate memory
        self._allocate_memory()

        # Set fastest possible settings
        self._optimize_settings()

        return True

    def _allocate_memory(self):
        """Allocate memory for frame capture"""
        # Set color mode
        ret = self.cam.PixelFormat.Set(IS_CM_MONO8)  # Use 8-bit mono for speed
        if ret != IS_SUCCESS:
            raise Exception(f"Failed to set pixel format: {ret}")

        # Allocate memory
        mem_id = Int32()
        ret = self.cam.Memory.Allocate(mem_id, self.width, self.height, self.bit_depth)
        if ret != IS_SUCCESS:
            raise Exception(f"Failed to allocate memory: {ret}")

        self.mem_id = mem_id

    def _optimize_settings(self):
        """Set optimal settings for maximum frame rate"""
        # Set maximum pixel clock (camera dependent)
        pixel_clocks = Array[Int32]([0])
        self.cam.PixelClock.GetList(pixel_clocks)
        if pixel_clocks.Length > 0:
            max_clock = max(pixel_clocks)
            self.cam.PixelClock.Set(max_clock)
            print(f"Set pixel clock to {max_clock} MHz")

        # Set minimum exposure time
        ret = self.cam.Timing.Exposure.SetToMinimum()
        if ret == IS_SUCCESS:
            exposure = self.cam.Timing.Exposure.Get()
            print(f"Set exposure to minimum: {exposure} ms")

        # Set ROI to center 1024x1024 for faster transfer
        try:
            ret = self.cam.Size.AOI.Set(
                (self.width - 1024) // 2,  # X position
                (self.height - 1024) // 2,  # Y position
                1024,  # Width
                1024   # Height
            )
            if ret == IS_SUCCESS:
                print("Set ROI to center 1024x1024")
        except:
            print("Could not set ROI")

    def start_capture(self):
        """Start live capture"""
        ret = self.cam.Acquisition.Capture()
        if ret != IS_SUCCESS:
            raise Exception(f"Failed to start capture: {ret}")
        self.running = True

    def stop_capture(self):
        """Stop live capture"""
        if self.running:
            ret = self.cam.Acquisition.Stop()
            if ret != IS_SUCCESS:
                print(f"Warning: Failed to stop capture: {ret}")
            self.running = False

    def grab_frame(self):
        """Grab a single frame and convert to numpy array"""
        if not self.running:
            self.start_capture()

        # Freeze frame
        ret = self.cam.Acquisition.Freeze()
        if ret != IS_SUCCESS:
            return None

        # Get the image data
        array = self._get_image_data()
        return array

    def _get_image_data(self):
        """Convert image data to numpy array"""
        # Get current AOI size
        rect = self.cam.Size.AOI.Get()
        width = rect.Width
        height = rect.Height

        # Create buffer for image data
        buffer_size = width * height
        buffer = Array.CreateInstance(System.Byte, buffer_size)

        # Copy image data to buffer
        ret = self.cam.Memory.CopyToArray(self.mem_id, buffer)
        if ret != IS_SUCCESS:
            return None

        # Convert to numpy array
        handle = GCHandle.Alloc(buffer, GCHandleType.Pinned)
        try:
            ptr = handle.AddrOfPinnedObject().ToInt64()
            array = np.frombuffer(buffer, dtype=np.uint8, count=buffer_size)
            array = array.reshape((height, width))
            return array
        finally:
            if handle.IsAllocated:
                handle.Free()

    def close(self):
        """Close the camera connection"""
        if self.cam is not None:
            self.stop_capture()
            self.cam.Exit()
            self.cam = None

class PylablibThorlabsCamera:
    """Implementation using Pylablib library for comparison"""
    def __init__(self):
        self.camera = None
        self.width = 0
        self.height = 0
        self.running = False

    def initialize(self):
        # Update SDK path
        pll.par["devices/dlls/thorlabs_tlcam"] = r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam\SDK"

        # List cameras
        cameras = Thorlabs.list_cameras_tlcam()
        if not cameras:
            print("No Thorlabs cameras found via pylablib")
            return False

        # Connect to first camera
        try:
            self.camera = Thorlabs.ThorlabsTLCamera(serial=cameras[0])
            print(f"Connected to camera: {cameras[0]} via pylablib")

            # Get camera info
            self.width = self.camera.get_detector_size()[0]
            self.height = self.camera.get_detector_size()[1]
            print(f"Sensor size: {self.width}x{self.height}")

            # Optimize settings
            self._optimize_settings()

            return True
        except Exception as e:
            print(f"Error connecting to camera via pylablib: {e}")
            return False

    def _optimize_settings(self):
        """Set optimal settings for maximum frame rate"""
        try:
            # Set minimum exposure
            self.camera.set_exposure(0.001)  # 1ms

            # Set ROI to center 1024x1024
            try:
                center_x = self.width // 2
                center_y = self.height // 2
                self.camera.set_roi(
                    center_x - 512,  # x start
                    center_y - 512,  # y start
                    1024,  # width
                    1024   # height
                )
                print("Set ROI to center 1024x1024 with pylablib")
            except:
                print("Could not set ROI with pylablib")

            # Other optimizations
            if hasattr(self.camera, 'set_frame_queue_size'):
                self.camera.set_frame_queue_size(1)
        except Exception as e:
            print(f"Error optimizing camera settings: {e}")

    def start_capture(self):
        if self.camera:
            self.camera.start_acquisition()
            self.running = True

    def stop_capture(self):
        if self.camera and self.running:
            self.camera.stop_acquisition()
            self.running = False

    def grab_frame(self):
        if not self.running:
            self.start_capture()

        try:
            frame_list = self.camera.grab(1)
            if frame_list:
                frame = np.array(frame_list[0])

                # If 16-bit, normalize to 8-bit for display
                if frame.dtype == np.uint16:
                    frame = (frame / 256).astype(np.uint8)

                return frame
        except Exception as e:
            print(f"Error grabbing frame: {e}")

        return None

    def close(self):
        if self.camera:
            self.stop_capture()
            self.camera.close()
            self.camera = None

class CameraThread(QThread):
    new_frame = pyqtSignal(np.ndarray)
    fps_update = pyqtSignal(float)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = False
        self.frame_count = 0
        self.last_fps_time = time.time()

    def run(self):
        self.running = True
        self.camera.start_capture()

        while self.running:
            frame = self.camera.grab_frame()

            if frame is not None:
                # Emit the frame
                self.new_frame.emit(frame)

                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_fps_time

                if elapsed >= 1.0:
                    fps = self.frame_count / elapsed
                    self.fps_update.emit(fps)
                    self.frame_count = 0
                    self.last_fps_time = current_time

    def stop(self):
        self.running = False
        self.camera.stop_capture()
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thorlabs Camera Performance Test")
        self.resize(1200, 800)

        # Initialize variables
        self.camera_thread = None
        self.current_camera = None
        self.camera_mode = None

        # Create UI elements
        self.init_ui()

        # Start camera
        self.switch_camera_mode()

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

        # Camera mode selection
        self.mode_combo = QComboBox()
        if NET_AVAILABLE:
            self.mode_combo.addItem(".NET Direct API")
        if PYLABLIB_AVAILABLE:
            self.mode_combo.addItem("Pylablib API")
        self.mode_combo.currentIndexChanged.connect(self.switch_camera_mode)
        control_layout.addWidget(self.mode_combo)

        # Start/Stop button
        self.toggle_button = QPushButton("Stop Stream")
        self.toggle_button.clicked.connect(self.toggle_streaming)
        control_layout.addWidget(self.toggle_button)

        # FPS display
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(QFont("Arial", 14))
        control_layout.addWidget(self.fps_label)

    def switch_camera_mode(self):
        # Stop current camera if running
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        if self.current_camera:
            self.current_camera.close()
            self.current_camera = None

        # Get selected mode
        mode = self.mode_combo.currentText()
        self.camera_mode = mode

        # Initialize new camera
        if mode == ".NET Direct API" and NET_AVAILABLE:
            self.current_camera = ThorlabsDirectCamera()
            success = self.current_camera.initialize()
        elif mode == "Pylablib API" and PYLABLIB_AVAILABLE:
            self.current_camera = PylablibThorlabsCamera()
            success = self.current_camera.initialize()
        else:
            self.display.setText("Selected camera mode is not available")
            return

        if success:
            # Start camera thread
            self.camera_thread = CameraThread(self.current_camera)
            self.camera_thread.new_frame.connect(self.update_frame)
            self.camera_thread.fps_update.connect(self.update_fps)
            self.camera_thread.start()
            self.toggle_button.setText("Stop Stream")
        else:
            self.display.setText("Failed to initialize camera")

    def update_frame(self, frame):
        # Convert frame to QImage for display
        if len(frame.shape) == 2:
            # Grayscale
            h, w = frame.shape
            bytes_per_line = w
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # RGB/BGR
            h, w, channels = frame.shape
            bytes_per_line = channels * w
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Convert to pixmap and display
        pixmap = QPixmap.fromImage(img)
        self.display.setPixmap(pixmap.scaled(self.display.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f} ({self.camera_mode})")

    def toggle_streaming(self):
        if self.camera_thread and self.camera_thread.running:
            self.camera_thread.stop()
            self.toggle_button.setText("Start Stream")
        else:
            if self.camera_thread:
                self.camera_thread.start()
                self.toggle_button.setText("Stop Stream")
            else:
                self.switch_camera_mode()

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if self.current_camera:
            self.current_camera.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
