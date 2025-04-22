import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
import traceback
import stitch_backend

from ctypes import cdll, c_uint32, c_int, c_double, c_char_p, byref, create_string_buffer
from enum import Enum

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QMutex, QMutexLocker, QTimer, QPoint, QRect, QRectF
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout, QCheckBox, QLineEdit,
    QFileDialog, QMessageBox, QComboBox, QProgressBar, QTabWidget
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QPainter, QColor, QIntValidator, QDoubleValidator

import pylablib as pll
from pylablib.devices import Thorlabs

# ------------------------------
# Constants
# ------------------------------
HISTOGRAM_WIDTH = 256
HISTOGRAM_HEIGHT = 100

# Configure ThorLabs SDK paths
def configure_thorlabs_paths():
    """Configure paths for ThorLabs SDK DLLs"""
    is_64bits = sys.maxsize > 2**32
    
    # Handle PyInstaller environment
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        application_path = sys._MEIPASS
        executable_dir = os.path.dirname(sys.executable)
    else:
        # Running in normal Python environment
        application_path = os.path.dirname(os.path.abspath(__file__))
        executable_dir = application_path
    
    # Add multiple possible DLL paths to search, prioritizing the executable directory
    dll_paths = [
        executable_dir,  # First try the executable directory
        os.path.join(application_path, 'Python Toolkit', 'dlls', '64_lib' if is_64bits else '32_lib'),
        os.path.join(os.getcwd(), 'Python Toolkit', 'dlls', '64_lib' if is_64bits else '32_lib'),
        os.path.join(os.path.dirname(sys.executable), 'Python Toolkit', 'dlls', '64_lib' if is_64bits else '32_lib')
    ]
    
    dll_found = False
    for path in dll_paths:
        if os.path.exists(path):
            os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
            try:
                os.add_dll_directory(path)
            except AttributeError:
                pass
            print(f"Added ThorLabs DLL path: {path}")
            dll_found = True
    
    if not dll_found:
        print(f"WARNING: Could not find ThorLabs DLLs in any of these paths:")
        for path in dll_paths:
            print(f"  - {path}")

# Configure SDK paths before importing
configure_thorlabs_paths()

# Try to import ThorLabs SDK components
try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
    from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
    from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK
    print("Successfully imported ThorLabs SDK for color processing")
except ImportError as e:
    print(f"Warning: ThorLabs SDK import error: {e}")
    print("Color processing functionality will be limited")

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
        self.homed_axes = set()  # Keep track of which axes are homed
        self._init_devices()

    def _init_devices(self):
        DeviceManagerCLI.BuildDeviceList()
        print("Kinesis devices found:", list(DeviceManagerCLI.GetDeviceList()))
        for axis, serial_no in self.axis_info.items():
            try:
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
                
                # Set high velocity through motor configuration
                try:
                    # Set a high velocity value (in steps/s)
                    # For KCube DCServo motors, a value of 100000 steps/s is typically safe
                    velocity = Decimal(100000)
                    m_config.Velocity = velocity
                    m_config.UpdateCurrentConfiguration()
                    device.SetSettings(device.MotorDeviceSettings, True, False)
                    print(f"[{axis}] Velocity set to {velocity} steps/s")
                except Exception as e:
                    print(f"[{axis}] Error setting velocity: {e}")
                
                # Check if the device needs to be homed
                try:
                    # Check if the device is homed based on position and status
                    if device.Status.IsHomed:
                        print(f"[{axis}] Already homed. Current pos: {device.Position}")
                        self.homed_axes.add(axis)
                    else:
                        print(f"Homing axis {axis} (serial {serial_no})...")
                        try:
                            device.Home(120000)
                            print(f"[{axis}] Homing complete.")
                            self.homed_axes.add(axis)
                        except Exception as e:
                            print(f"[{axis}] Error during homing: {e}")
                            print(f"[{axis}] Will need to be homed manually.")
                except Exception as e:
                    print(f"[{axis}] Error checking homing status: {e}")
                
                # Store the device even if homing failed - the user can home it manually later
                self.devices[axis] = device
                
            except Exception as e:
                print(f"[{axis}] Error initializing: {e}")
    
    def ensure_all_axes_homed(self):
        """Check if all axes are homed, and try to home any that aren't"""
        for axis, device in self.devices.items():
            if axis not in self.homed_axes:
                try:
                    print(f"[{axis}] Not homed. Attempting to home now...")
                    device.Home(120000)  # 120 second timeout
                    print(f"[{axis}] Homing complete.")
                    self.homed_axes.add(axis)
                except Exception as e:
                    print(f"[{axis}] Failed to home: {e}")
                    # We'll continue even if homing fails
        
        # Return True if all axes are homed, False otherwise
        return len(self.homed_axes) == len(self.devices)

    def disconnect_all(self):
        for axis, device in self.devices.items():
            if device is not None:
                print(f"Disconnecting axis {axis}...")
                device.StopPolling()
                device.DisableDevice()
                device.Disconnect()
        SimulationManager.Instance.UninitializeSimulations()

    def _get_coordinate(self, axis):
        """Get the current position of the specified axis in mm"""
        if axis not in self.devices:
            raise ValueError(f"Axis {axis} not found")
        
        try:
            device = self.devices[axis]
            position = float(str(device.Position))
            return position
        except Exception as e:
            print(f"Error getting {axis} position: {e}")
            return 0.0


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
        self.hardware._move_axis_relative(self.axis, self.step_mm)
        self.finished_move.emit()

class ZeroStageWorker(QThread):
    finished_move = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, hardware, parent=None):
        super().__init__(parent)
        self.hardware = hardware

    def run(self):
        try:
            if self.hardware.zero_stage_at_optic_center():
                self.finished_move.emit()
            else:
                self.error_occurred.emit("Failed to move stage to focus position")
        except Exception as e:
            self.error_occurred.emit(f"Error during zeroing: {str(e)}")

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
        self.pattern_size_x = pattern_size_x  # Now in mm
        self.pattern_size_y = pattern_size_y  # Now in mm
        self.z_stack_height = z_stack_height  # In mm

        self.time_estimates = {
            "Focus Stack": 3.0,
            "XY Pattern": 7.0,
            "Combined": 5.1,
        }
        
    def run(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        folder_name = os.path.join(downloads, f"AutoCapture_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)
        self.hardware.auto_capture_folder = folder_name
        self.hardware.current_capture_folder = folder_name

        # Set auto capture flag
        self.hardware.auto_capture_in_progress = True
        
        # Pause briefly to ensure we have a stable connection
        time.sleep(0.2)

        # Get current position
        current_x = self._get_current_coordinate('X')
        current_y = self._get_current_coordinate('Y')
        current_z = self._get_current_coordinate('Z')

        if self.mode == "z":
            self._run_focus_stack(current_x, current_y, current_z, folder_name)
        elif self.mode == "xy":
            self._run_xy_pattern(current_x, current_y, current_z, folder_name)
        elif self.mode == "xyz":
            self._run_combined(current_x, current_y, current_z, folder_name)

        # Ensure LED is turned off after capture
        self.hardware.turn_off_led()
            
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

    def _run_focus_stack(self, x, y, z, folder_name):
        if self.num_stack <= 1:
            z_positions = [z]
        else:
            z_positions = [z - self.z_stack_height/2 + i*(self.z_stack_height/(self.num_stack-1))
                           for i in range(self.num_stack)]
        total = len(z_positions)
        completed = 0
        per_capture = self.time_estimates["Focus Stack"]
        for z_target in z_positions:
            self.hardware.move_to_absolute(z=z_target, wait_for_completion=True)
            time.sleep(0.1)
            self.hardware.save_last_frame(filename=f"focusstack_X{x:.3f}_Y{y:.3f}_Z{z_target:.3f}.tiff",foldername=folder_name)
            if self.hardware.last_frame is not None:
                self.new_capture.emit(self.hardware.last_frame)
            completed += 1
            remaining = (total - completed) * per_capture
            self.progress_update.emit(completed, total, remaining)

    def _return_pattern_coordinates(self, x, y, fov_x=None, fov_y=None):
        """Calculate grid positions for pattern capture.
        
        Args:
            x (float): Center X coordinate
            y (float): Center Y coordinate
            
        Returns:
            tuple: (list of X positions, list of Y positions)
        """
        # Get field of view and calculate step sizes
        if fov_x is None:
            fov_x = self.hardware.fov_x
        if fov_y is None:
            fov_y = self.hardware.fov_y
        step_x = fov_x * (1 - self.overlap/100.0)
        step_y = fov_y * (1 - self.overlap/100.0)
        
        # Calculate number of positions needed
        count_x = int(np.ceil(self.pattern_size_x / step_x))
        count_y = int(np.ceil(self.pattern_size_y / step_y))
        
        # Calculate total pattern size
        total_width = step_x * (count_x - 1)
        total_height = step_y * (count_y - 1)
        
        # Calculate starting positions (centered on x,y)
        start_x = x - total_width/2
        start_y = y - total_height/2
        
        # Generate position lists
        positions_x = [start_x + i * step_x for i in range(count_x)]
        positions_y = [start_y + i * step_y for i in range(count_y)]
        
        return positions_x, positions_y

    def _run_xy_pattern(self, x, y, z, folder_name):
        positions_x, positions_y = self._return_pattern_coordinates(x, y)

        total = len(positions_x) * len(positions_y)
        completed = 0
        per_capture = self.time_estimates["XY Pattern"]
        for j, y_target in enumerate(positions_y):
            xs = positions_x if j % 2 == 0 else list(reversed(positions_x))
            for x_target in xs:
                self.hardware.move_to_absolute(x=x_target, y=y_target, wait_for_completion=True)
                time.sleep(0.1)
                self.hardware.save_last_frame(filename=f"xy_X{x_target:.3f}_Y{y_target:.3f}_Z{z:.3f}.tiff",foldername=folder_name)
                if self.hardware.last_frame is not None:
                    self.new_capture.emit(self.hardware.last_frame)
                completed += 1
                remaining = (total - completed) * per_capture
                self.progress_update.emit(completed, total, remaining)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stitch_backend.stitch_original_images(folder_name, file_name=f"stitched_xy_Z{z:.3f}_{timestamp}.png", input_overlap_percent=self.overlap)

    def _run_combined(self, x, y, z, folder_name):
        positions_x, positions_y = self._return_pattern_coordinates(x, y)

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
                self.hardware.move_to_absolute(x=x_target, y=y_target, wait_for_completion=True)
                time.sleep(0.1)
                for z_target in z_positions:
                    self.hardware.move_to_absolute(z=z_target, wait_for_completion=True)
                    time.sleep(0.1)
                    self.hardware.save_last_frame(filename=f"combined_X{x_target:.3f}_Y{y_target:.3f}_Z{z_target:.3f}.tiff",foldername=folder_name)
                    if self.hardware.last_frame is not None:
                        self.new_capture.emit(self.hardware.last_frame)
                    completed += 1
                    remaining = (total - completed) * per_capture
                    self.progress_update.emit(completed, total, remaining)
                self.hardware.move_to_absolute(z=z, wait_for_completion=True)
                time.sleep(0.1)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stitch_backend.stitch_original_images(folder_name, file_name=f"stitched_xyz_Z{z:.3f}_{timestamp}.png", input_overlap_percent=self.overlap)

###############################################################################
# PreviewFieldWorker: Runs a preview scan of the entire field and creates a mosaic
###############################################################################
class PreviewFieldWorker(QThread):
    progress_update = pyqtSignal(int, int, float)
    new_preview_image = pyqtSignal(np.ndarray, int, int)  # image, grid_x, grid_y
    finished_preview = pyqtSignal(np.ndarray)  # final stitched mosaic
    
    def __init__(self, hardware, overlap_percent=0, parent=None):
        super().__init__(parent)
        self.hardware = hardware
        self.running = True
        self.overlap_percent = overlap_percent
        
        # Create a timestamped folder in Downloads for this preview
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.preview_folder = os.path.join(os.path.expanduser("~"), "Downloads", f"preview_field_{timestamp}")
        os.makedirs(self.preview_folder, exist_ok=True)
        print(f"Created preview folder: {self.preview_folder}")
    
    def run(self):
        """Run the preview field scan"""
        try:
            # First move to zero height for the current optic
            z_zero = self.hardware.z_zero_positions.get(self.hardware.current_optic, 0)
            print(f"Moving to Z = {z_zero} mm (zero height for {self.hardware.current_optic})")
            self.hardware.move_to_absolute(z=z_zero, wait_for_completion=True)
            
            fov_x = self.hardware.fov_x
            fov_y = self.hardware.fov_y
            step_x = fov_x * (1 - self.overlap_percent/100.0)
            step_y = fov_y * (1 - self.overlap_percent/100.0)
            
            # Calculate number of positions needed
            count_x = int(np.ceil((self.hardware.max_slide_x-fov_x)/fov_x))
            count_y = int(np.ceil((self.hardware.max_slide_y-fov_y)/fov_y))
            
            # Calculate total pattern size
            total_width = step_x * (count_x - 1)
            total_height = step_y * (count_y - 1)
            
            # Calculate starting positions (centered on x,y)
            start_x = self.hardware.motion_stage_center_x - total_width/2 if self.hardware.motion_stage_center_x - total_width/2 > self.hardware.motion_stage_min_x else self.hardware.motion_stage_min_x
            start_y = self.hardware.motion_stage_center_y - total_height/2 if self.hardware.motion_stage_center_y - total_height/2 > self.hardware.motion_stage_min_y else self.hardware.motion_stage_min_y
            
            # Generate position lists
            positions_x = [start_x + i * step_x for i in range(count_x)]
            positions_y = [start_y + i * step_y for i in range(count_y)]

            print(positions_x)
            print(positions_y)

            
            # Use hardware's actual FOV values
            print(f"Using FOV from hardware: {fov_x}mm x {fov_y}mm for {self.hardware.current_optic}")

                
            
            # Define grid dimensions
            grid_width = len(positions_x)
            grid_height = len(positions_y)
            total_positions = grid_width * grid_height
            
            # Initialize image grid (use None for empty spaces)
            grid_images = [[None for _ in range(grid_width)] for _ in range(grid_height)]
            
            # Report grid dimensions
            print(f"Preview field: {grid_width}x{grid_height} grid ({total_positions} positions)")
            
            # Progress tracking
            completed = 0
            
            # Use the positions in the original order for movement, but map to flipped display indices
            for j, y_pos in enumerate(reversed(positions_y)):
                # Get the Y display index (flipped vertically)
                display_y = grid_height - j - 1
                
                # Get positions for this row - same movement pattern
                x_positions = positions_x if j % 2 == 0 else reversed(positions_x)
                
                for i, x_pos in enumerate(x_positions):
                    # Check if stop was requested
                    if not self.running:
                        return
                    
                    # Convert i to proper x_index based on direction
                    display_x = i if j % 2 == 0 else grid_width - i - 1
                    
                    # Move to this position
                    self.hardware.move_to_absolute(x=x_pos, y=y_pos, wait_for_completion=True)
                    
                    # Wait for move to complete and position to stabilize
                    time.sleep(0.1)
                    
                    # Capture image
                    try:
                        # Small delay to stabilize
                        time.sleep(0.1)
                        # Take photo - save to our preview folder
                        filename = f"preview_{display_x}_{display_y}.jpg"
                        filepath = os.path.join(self.preview_folder, filename)
                        
                        with QMutexLocker(self.hardware.camera_lock):
                            if not self.hardware.camera:
                                print("[SAVE] No camera available to save!")
                                return None
                            
                            # Trigger and get frame
                            self.hardware.camera.issue_software_trigger()
                            timeout_ms = 5000
                            start_time = time.time()
                            
                            frame = None
                            while time.time() - start_time < timeout_ms/1000:
                                camera_frame = self.hardware.camera.get_pending_frame_or_null()
                                if camera_frame is not None:
                                    frame = camera_frame.image_buffer
                                    break
                                time.sleep(0.01)
                            
                            if frame is None:
                                print("[SAVE] Failed to capture frame for saving, trying again at current location")
                                return None

                        # Save the frame
                        x = self.hardware.motion_controller._get_coordinate('X')
                        y = self.hardware.motion_controller._get_coordinate('Y')
                        z = self.hardware.motion_controller._get_coordinate('Z')
                        self.hardware.save_last_frame(
                            filename=f"preview_X{x:.3f}_Y{y:.3f}_Z{z:.3f}.tiff",
                            foldername=self.preview_folder
                        )
                        
                        # Load the saved image
                        saved_image_path = os.path.join(self.preview_folder, f"preview_X{x:.3f}_Y{y:.3f}_Z{z:.3f}.tiff")
                        if os.path.exists(saved_image_path):
                            # Load the saved image
                            image = cv2.imread(saved_image_path)
                            if image is not None:
                                # Convert from BGR to RGB
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                
                                # Store in grid with flipped Y index to get correct vertical orientation
                                flipped_y = grid_height - display_y - 1  # Flip Y for display
                                grid_images[flipped_y][display_x] = image
                                
                                # Emit new image for display (with flipped Y)
                                self.new_preview_image.emit(image, display_x, flipped_y)
                    except Exception as e:
                        print(f"Error capturing image at ({x_pos}, {y_pos}): {e}")
                    
                    # Update progress
                    completed += 1
                    remaining_time = (total_positions - completed) * 1.0  # Rough estimate
                    self.progress_update.emit(completed, total_positions, remaining_time)
            
            # Create a mosaic from all images
            if self.running:
                mosaic = self._create_mosaic(grid_images, grid_width, grid_height)
                self.finished_preview.emit(mosaic)
        
        except Exception as e:
            print(f"Error during preview field scan: {e}")
            traceback.print_exc()
    
    def _create_mosaic(self, grid_images, grid_width, grid_height):
        """Create a stitched mosaic from the grid of images"""
        if not grid_images or not grid_images[0]:
            # Return empty mosaic if no images
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Find first non-None image to get dimensions
        sample_img = None
        for row in grid_images:
            for img in row:
                if img is not None:
                    sample_img = img
                    break
            if sample_img is not None:
                break
        
        if sample_img is None:
            # No valid images found
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        img_height, img_width = sample_img.shape[:2]
        
        # Create the mosaic canvas
        mosaic_width = grid_width * img_width
        mosaic_height = grid_height * img_height
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
        
        # Place each image in the mosaic
        for y in range(grid_height):
            for x in range(grid_width):
                if grid_images[y][x] is not None:
                    y_start = y * img_height
                    y_end = y_start + img_height
                    x_start = x * img_width
                    x_end = x_start + img_width
                    img = grid_images[y][x]
                    # Make sure image dimensions match expected shape
                    if img.shape[:2] == (img_height, img_width):
                        mosaic[y_start:y_end, x_start:x_end] = img
        
        return mosaic
    
    def stop(self):
        """Request the worker to stop"""
        self.running = False
        self.wait()  # Wait for thread to finish



###############################################################################
# CameraWorker: Continuously grabs frames from the camera in a separate thread.
###############################################################################
class CameraWorker(QThread):
    new_frame = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera, hardware, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.hardware = hardware
        self.running = True
        self.last_display_time = time.time()
        self.frame_rate_limit = 30  # Limit FPS to avoid excessive CPU usage
        self.min_frame_interval = 1.0 / self.frame_rate_limit
        self.last_autoexposure_time = 0
        self.autoexposure_interval = 0.5  # Only adjust exposure every 0.5 seconds
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        # Check if we're using ThorLabs SDK directly or pylablib
        self.using_sdk_directly = hasattr(self.hardware, '_camera_sdk') and self.hardware._camera_sdk is not None
    
    def run(self):
        while self.running:
            try:                
                current_time = time.time()
                time_since_last_frame = current_time - self.last_display_time
                
                # Rate limiting to prevent UI freezing
                if time_since_last_frame < self.min_frame_interval:
                    time.sleep(self.min_frame_interval - time_since_last_frame)
                    continue
                
                # Acquire frame based on camera type
                if self.using_sdk_directly:
                    # Using ThorLabs SDK directly
                    try:
                        frame = self.camera.get_pending_frame_or_null()
                        if frame is None:
                            time.sleep(0.001)  # Avoid CPU spinning
                            continue
                            
                        # Get raw frame data
                        np_frame = frame.image_buffer
                        
                    except Exception as e:
                        self.consecutive_errors += 1
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            self.error_occurred.emit(f"Multiple camera errors: {str(e)}")
                            # Reset counter but continue trying
                            self.consecutive_errors = 0
                        continue
                
                else:
                    # Using pylablib camera
                    with QMutexLocker(self.hardware.camera_lock):
                        try:
                            frame_list = self.camera.grab(1)
                        except Exception as e:
                            self.consecutive_errors += 1
                            if self.consecutive_errors >= self.max_consecutive_errors:
                                self.error_occurred.emit(f"Multiple camera errors: {str(e)}")
                                # Reset counter but continue trying
                                self.consecutive_errors = 0
                            continue
                    
                    if not frame_list:
                        time.sleep(0.01)  # Short sleep to prevent CPU spinning
                        continue
                    
                    # Get raw frame data
                    np_frame = np.array(frame_list[0])
                
                # Reset error counter on successful frame
                self.consecutive_errors = 0
                
                # Process the frame through the hardware
                processed_frame = self.hardware.process_frame(np_frame)
                
                # Update the last frame time
                self.hardware.last_frame_time = time.time()
                
                # Emit the processed frame
                self.new_frame.emit(processed_frame)
                
                # Update display timing
                self.last_display_time = time.time()
                
            except Exception as e:
                # Handle any unexpected errors
                self.consecutive_errors += 1
                error_message = f"Camera error: {str(e)}"
                print(error_message)
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self.error_occurred.emit(f"Multiple camera errors occurred: {str(e)}. Camera thread will stop.")
                    self.running = False
                    break
                
                # Short sleep on error to prevent CPU spinning
                time.sleep(0.1)
    
    def stop(self):
        self.running = False
        self.wait()  # Wait for thread to finish


###############################################################################
# MicroscopeHardware: Combines LED, Camera, and Motion; includes auto-capture helpers.
###############################################################################
class MicroscopeHardware:
    def __init__(self):
        # Initialize camera lock first
        self.camera_lock = QMutex()
        
        self.led_handles = {}  # Dictionary to store multiple LED handles
        self.led_max_currents = {}  # Dictionary to store max currents
        self.current_led_type = None
        self.led_lib = None
        self.camera = None
        self.last_frame = None
        self.last_frame_time = 0  # Track when frames were captured
        
        # Initialize motion controller first to give it time to home axes
        self.motion_controller = MotionController()
        self.move_workers = []
        
        # Optics-specific parameters
        self.optics_params = {
            "4X": {
                "fov_x": 12.0,  # Field of view in mm
                "fov_y": 10.0,
                "z_zero": 46.3,  # Z zero position in mm (absolute)
            },
            "10X": {
                "fov_x": 2.6,  # Field of view in mm
                "fov_y": 2.15,
                "z_zero": 8.6,  # Z zero position in mm (absolute)
            }
        }

        self.motion_stage_center_x = 16.3
        self.motion_stage_center_y = 19.1

        self.motion_stage_min_x = 0.0
        self.motion_stage_min_y = 0.0
        self.motion_stage_max_x = 48.0
        self.motion_stage_max_y = 48.0
        
        # Default to 4X optic
        self.current_optic = "4X"
        self.fov_x = self.optics_params["4X"]["fov_x"]
        self.fov_y = self.optics_params["4X"]["fov_y"]
        
        # Create a dictionary for z_zero positions
        self.z_zero_positions = {
            "4X": self.optics_params["4X"]["z_zero"],
            "10X": self.optics_params["10X"]["z_zero"]
        }
        self.z_zero = self.z_zero_positions["4X"]  # For backward compatibility
        
        self.max_slide_x = 50.0  # mm
        self.max_slide_y = 51.0  # mm
        self.auto_capture_in_progress = False
        self.auto_capture_folder = None
        
        # Camera-specific properties
        self.camera_model = "Unknown"
        self.bit_depth = 12  # Default to 12-bit
        self.max_pixel_value = 4095  # Default for 12-bit
        self.sensor_width = 2448  # Default for CS505CU1
        self.sensor_height = 2048  # Default for CS505CU1
        
        # Image processing settings
        self.normalize_enabled = True
        self.last_frame_time = 0
        self._processing_cache = {}
        
        # LED controller configuration
        self.led_controllers = {}  # Dictionary to store LED controller info
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "led_config.json")
        
        # Initialize hardware in a specific order
        self._init_led_lib()
        self._detect_led_controllers()
        self._init_camera_hardware()
        
        # Auto connect to default LED
        self._connect_to_default_led()
        
        # Display homing status
        self.check_axes_homed()
    
    def check_axes_homed(self):
        """Check and report on the homing status of all axes"""
        if not self.motion_controller:
            return
        
        unhomed_axes = []
        for axis in ["X", "Y", "Z"]:
            if axis not in self.motion_controller.homed_axes:
                unhomed_axes.append(axis)
        
        if unhomed_axes:
            print(f"WARNING: The following axes are not homed: {', '.join(unhomed_axes)}")
            print("Stage movement may be inaccurate. Consider restarting the application.")
        else:
            print("All axes are properly homed.")
            
    def ensure_axes_homed(self):
        """Attempt to home any unhomed axes"""
        if not self.motion_controller:
            return False
            
        return self.motion_controller.ensure_all_axes_homed()

    def _connect_to_default_led(self):
        """Connect to the default LED based on availability"""
        if not self.led_controllers:
            return
            
        # Prefer White LED if available, otherwise UV
        if "White LED" in self.led_controllers:
            self.connect_led("White LED")
        elif "UV LED" in self.led_controllers:
            self.connect_led("UV LED")
        else:
            # Connect to the first available LED
            first_led = next(iter(self.led_controllers))
            self.connect_led(first_led)



    def _init_led_lib(self):
        """Load the LED library"""
        if self.led_lib is None:
            dll_path = r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLUP_64.dll"
            try:
                self.led_lib = cdll.LoadLibrary(dll_path)
                print("LED library loaded successfully.")
            except OSError as e:
                print(f"Failed to load LED DLL at {dll_path}: {e}")
    
    def _detect_led_controllers(self):
        """Auto-detect LED controllers on any available COM port"""        
        # List of possible COM ports to try
        possible_ports = [f"COM{i}" for i in range(1, 20)]  # Check COM1 through COM19
        
        # Try to initialize each port directly
        for port in possible_ports:
            try:
                resource_name = f"ASRL{port[3:]}::INSTR"  # Convert COM4 to ASRL4::INSTR
                resource_bytes = resource_name.encode()
                handle = c_int(0)
                res = self.led_lib.TLUP_init(resource_bytes, 0, 0, byref(handle))
                
                if res != 0:
                    # Not a valid LED controller port
                    continue
                    
                print(f"Found LED controller on {port}")
                
                # Try to identify the connected LED
                try:
                    # Configure controller
                    self.led_lib.TLUP_setLedUseNonThorlabsLed(handle, 1)
                    
                    # Get LED info
                    LEDName = create_string_buffer(256)
                    LEDSerialNumber = create_string_buffer(256)
                    LEDCurrentLimit = c_double()
                    LEDForwardVoltage = c_double()
                    LEDWavelength = c_double(0)
                    
                    self.led_lib.TLUP_getLedInfo(handle, LEDName, LEDSerialNumber,
                                               byref(LEDCurrentLimit),
                                               byref(LEDForwardVoltage),
                                               byref(LEDWavelength))
                    
                    led_model = LEDName.value.decode().strip()
                    led_serial = LEDSerialNumber.value.decode().strip()
                    wavelength = LEDWavelength.value
                    
                    # Identify LED type based on wavelength
                    led_type = None
                    if wavelength == -4000.0:
                        led_type = "White LED"
                    elif wavelength == 275.0:
                        led_type = "UV LED"
                    else:
                        led_type = f"Unknown LED (wavelength: {wavelength})"
                    
                    print(f"Detected {led_type} on {port}")
                    print(f"  Model: {led_model}")
                    print(f"  Serial: {led_serial}")
                    print(f"  Wavelength: {wavelength}")
                    print(f"  Max Current: {LEDCurrentLimit.value}")
                    
                    # Store controller info for this LED type
                    self.led_controllers[led_type] = {
                        "port": port,
                        "resource": resource_name,
                        "led_model": led_model,
                        "led_serial": led_serial,
                        "led_wavelength": wavelength
                    }
                    
                except Exception as e:
                    print(f"Error identifying LED on {port}: {e}")
                
                # Close this controller before trying the next one
                self.led_lib.TLUP_close(handle)
                print(f"Closed connection to {port} (will reconnect later if needed)")
                
            except Exception as e:
                print(f"Error checking {port}: {e}")
        
        # Report detected LEDs
        if not self.led_controllers:
            print("No LED controllers detected on any COM port.")
        else:
            print(f"Found {len(self.led_controllers)} LED controller(s):")
            for led_type, info in self.led_controllers.items():
                print(f"  {led_type}: {info['port']} ({info['led_model']}, {info['led_serial']})")
    
    def connect_led(self, requested_led_type=None):
        """Connect to the specified LED type using direct COM port access"""
        # First turn off any current LED
        self.turn_off_led()
        
        # Close any open handles to avoid conflicts
        if self.current_led_type in self.led_handles:
            try:
                handle = self.led_handles[self.current_led_type]
                self.led_lib.TLUP_close(handle)
                # Clear current handles
                self.led_handles.pop(self.current_led_type, None)
                self.led_max_currents.pop(self.current_led_type, None)
                self.current_led_type = None
            except Exception as e:
                print(f"Error closing LED controller: {e}")
        
        # If no specific type requested, default to White LED
        if requested_led_type is None:
            if "White LED" in self.led_controllers:
                requested_led_type = "White LED"
            elif "UV LED" in self.led_controllers:
                requested_led_type = "UV LED"
            elif self.led_controllers:
                # Just use the first available LED
                requested_led_type = next(iter(self.led_controllers))
            else:
                print("No LED controllers available.")
                return False
        
        # Check if the requested type is available
        if requested_led_type not in self.led_controllers:
            print(f"LED type {requested_led_type} not available.")
            return False
        
        # Get the controller info for this LED type
        led_info = self.led_controllers[requested_led_type]
        resource_name = led_info["resource"]
        
        # Initialize the controller
        handle = c_int(0)
        res = self.led_lib.TLUP_init(resource_name.encode(), 0, 0, byref(handle))
        
        if res != 0:
            print(f"Failed to initialize controller for {requested_led_type}: error {res}")
            return False
        
        try:
            # Configure the controller
            self.led_lib.TLUP_setLedUseNonThorlabsLed(handle, 1)
            self.led_lib.TLUP_switchLedOutput(handle, 0)  # Make sure it's off initially
            
            # Get LED info to confirm connection and get current limit
            LEDName = create_string_buffer(256)
            LEDSerialNumber = create_string_buffer(256)
            LEDCurrentLimit = c_double()
            LEDForwardVoltage = c_double()
            LEDWavelength = c_double(0)
            
            self.led_lib.TLUP_getLedInfo(handle, LEDName, LEDSerialNumber,
                                       byref(LEDCurrentLimit),
                                       byref(LEDForwardVoltage),
                                       byref(LEDWavelength))
            
            # Set current to 10% of max
            current_limit = LEDCurrentLimit.value
            currentSetpoint = c_double(0.1 * current_limit)
            self.led_lib.TLUP_setLedCurrentSetpoint(handle, currentSetpoint)
            
            # Store the handle
            self.led_handles[requested_led_type] = handle
            self.led_max_currents[requested_led_type] = current_limit
            self.current_led_type = requested_led_type
            
            print(f"Connected to {requested_led_type}")
            return True
            
        except Exception as e:
            print(f"Error configuring {requested_led_type}: {e}")
            self.led_lib.TLUP_close(handle)
            return False

    def turn_on_led(self):
        """Turn on the current LED"""
        if self.current_led_type and self.current_led_type in self.led_handles:
            handle = self.led_handles[self.current_led_type]
            self.led_lib.TLUP_switchLedOutput(handle, 1)
            print(f"[{self.current_led_type}] Turned on.")
        else:
            print("No LED selected.")

    def turn_off_led(self):
        """Turn off the current LED"""
        if self.current_led_type and self.current_led_type in self.led_handles:
            handle = self.led_handles[self.current_led_type]
            self.led_lib.TLUP_switchLedOutput(handle, 0)
            print(f"[{self.current_led_type}] Turned off.")
        else:
            print("No LED selected.")

    def set_led_intensity(self, intensity_percent):
        """Set the intensity of the current LED"""
        if (self.current_led_type and 
            self.current_led_type in self.led_handles and 
            self.current_led_type in self.led_max_currents):
            handle = self.led_handles[self.current_led_type]
            max_current = self.led_max_currents[self.current_led_type]
            current = (intensity_percent / 100.0) * max_current
            self.led_lib.TLUP_setLedCurrentSetpoint(handle, c_double(current))
            print(f"[{self.current_led_type}] Intensity set to {intensity_percent}% (current = {current} A)")
        else:
            print("LED not initialized.")

    def _init_camera_hardware(self):
        # Initialize color processing variables
        self._mono_to_color_sdk = None
        self._mono_to_color_processor = None
        self._is_color = False
        self._camera_sdk = None

        try:
            # Try to initialize using direct ThorLabs SDK first
            try:
                self._camera_sdk = TLCameraSDK()
                camera_list = self._camera_sdk.discover_available_cameras()
                
                if not camera_list:
                    print("No cameras detected with ThorLabs SDK, falling back to pylablib")
                    raise ImportError("No cameras found")
                
                # Open the first camera
                self.camera = self._camera_sdk.open_camera(camera_list[0])
                print(f"Connected to camera: {self.camera.name}")
                
                # Configure camera for performance
                self.camera.frames_per_trigger_zero_for_unlimited = 0  # Continuous mode
                self.camera.exposure_time_us = 100000  # Initial 100ms exposure
                
                # Get camera model information
                self.camera_model = self.camera.name
                self.bit_depth = self.camera.bit_depth
                self.sensor_width = self.camera.image_width_pixels
                self.sensor_height = self.camera.image_height_pixels
                
                # Set up color processing if necessary
                if self.camera.camera_sensor_type == SENSOR_TYPE.BAYER:
                    # Initialize color processing for Bayer pattern cameras
                    self._mono_to_color_sdk = MonoToColorProcessorSDK()
                    self._mono_to_color_processor = self._mono_to_color_sdk.create_mono_to_color_processor(
                        SENSOR_TYPE.BAYER,
                        self.camera.color_filter_array_phase,
                        self.camera.get_color_correction_matrix(),
                        self.camera.get_default_white_balance_matrix(),
                        self.camera.bit_depth
                    )
                    self._is_color = True
                    self.bayer_pattern = "BAYER"  # Generic label, actual pattern handled by SDK
                    print(f"Camera is color, using ThorLabs color processing")
                else:
                    # Monochrome camera
                    self._is_color = False
                    self.bayer_pattern = None
                    print(f"Camera is monochrome")
                
                # Arm the camera
                self.camera.arm(2)
                self.camera.issue_software_trigger()
                
                # Parameters for proper frame processing
                self.max_pixel_value = 2**self.bit_depth - 1
                print(f"Camera initialized: {self.camera_model}, {self.sensor_width}x{self.sensor_height}, {self.bit_depth}-bit")
                return
                
            except (ImportError, AttributeError, NameError) as e:
                print(f"Direct ThorLabs SDK initialization failed: {e}, falling back to pylablib")
        
            # Fall back to pylablib if direct SDK fails
            pll.par["devices/dlls/thorlabs_tlcam"] = r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam\SDK"
            
            cameras = Thorlabs.list_cameras_tlcam()
            if not cameras:
                print("No Thorlabs cameras found.")
                self.camera = None
                return
            
            # Connect to camera and get information
            self.camera = Thorlabs.ThorlabsTLCamera(serial=cameras[0])
            print(f"Connected to camera: {cameras[0]} (via pylablib)")
            
            # Check if camera is a CS505CU1 model (or similar)
            try:
                camera_info = self.camera.get_device_info()
                # Try to access attributes directly on TDeviceInfo object
                try:
                    self.camera_model = camera_info.model
                    print(f"Camera model: {self.camera_model}")
                except AttributeError:
                    # If model attribute is not available, print what's available
                    print("Camera info available attributes:", dir(camera_info))
                    self.camera_model = "Unknown"
                
                # Try to get bit depth - first check if it's an attribute
                try:
                    self.bit_depth = camera_info.bit_depth
                    print(f"Bit depth: {self.bit_depth}")
                except AttributeError:
                    # Fallback to hardcoded value for CS505CU1
                    if "CS505CU" in str(camera_info):
                        self.bit_depth = 12
                    else:
                        self.bit_depth = 12  # Default to 12-bit if not specified
                    print(f"Using default bit depth: {self.bit_depth}")
            except Exception as e:
                print(f"Error accessing camera info: {e}")
                self.camera_model = "Unknown"
                self.bit_depth = 12  # Default to 12-bit
            
            # Get sensor dimensions if available
            if "CS505CU" in self.camera_model:
                self.sensor_width = 2448  # CS505CU1 width
                self.sensor_height = 2048  # CS505CU1 height
                print(f"Sensor dimensions: {self.sensor_width}x{self.sensor_height}")
                
                # Configure for optimal performance with CS505CU1
                self.configure_cs505cu1()
            else:
                # Try to get dimensions from camera
                try:
                    # Try different ways to access dimensions
                    if hasattr(camera_info, 'width') and hasattr(camera_info, 'height'):
                        self.sensor_width = camera_info.width
                        self.sensor_height = camera_info.height
                    elif hasattr(self.camera, 'get_sensor_width') and hasattr(self.camera, 'get_sensor_height'):
                        self.sensor_width = self.camera.get_sensor_width()
                        self.sensor_height = self.camera.get_sensor_height()
                    else:
                        # Default values for CS505CU1
                        self.sensor_width = 2448
                        self.sensor_height = 2048
                    print(f"Sensor dimensions: {self.sensor_width}x{self.sensor_height}")
                except Exception as e:
                    print(f"Error getting sensor dimensions: {e}, using defaults")
                    self.sensor_width = 2448
                    self.sensor_height = 2048
            
            # Set fixed values for CS505CU1 camera
            if "CS505CU" in self.camera_model or "CS505" in self.camera_model:
                self.max_pixel_value = 2**self.bit_depth - 1  # 4095 for 12-bit
                # Set known Bayer pattern for CS505CU1 (typically GRBG)
                self.bayer_pattern = "GRBG"
                print(f"Using fixed Bayer pattern for {self.camera_model}: {self.bayer_pattern}")
                
                # Set color correction matrix for CS505CU1 under LED illumination
                # These values are specific to this camera/illumination combo
                self.color_correction_matrix = np.array([
                    [1.80, -0.54, -0.26],  # Red channel  
                    [-0.24, 1.28, -0.04],  # Green channel
                    [0.04, -0.55, 1.51]    # Blue channel
                ])
            else:
                # For other cameras, use more conservative settings
                self.max_pixel_value = 2**16 - 1  # Assume 16-bit max
                # Try to determine Bayer pattern
                try:
                    # Check for known pattern in camera info
                    if hasattr(camera_info, 'bayer_pattern'):
                        self.bayer_pattern = camera_info.bayer_pattern
                    elif "27727" in cameras[0]:  # Specific serial for CS505CU1
                        self.bayer_pattern = "GRBG"
                    else:
                        self.bayer_pattern = "RGGB"  # Default
                    print(f"Using Bayer pattern: {self.bayer_pattern}")
                except Exception:
                    # Default to most common pattern
                    self.bayer_pattern = "RGGB"
                    print(f"Using default Bayer pattern: {self.bayer_pattern}")
            
            # Configure initial exposure
            self.camera.set_exposure(0.1)  # Set initial exposure
            print(f"Camera exposure set to 0.1 s")
            
            # Get initial frame to check format - with error handling
            self.raw_mode = True  # Default to raw mode
            frame_grab_success = False
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    with QMutexLocker(self.camera_lock):
                        # Remove timeout parameter that causes errors
                        frame_list = self.camera.grab(1)
                        
                        if frame_list:
                            frame = np.array(frame_list[0])
                            print(f"Initial frame shape: {frame.shape}")
                            print(f"Initial frame dtype: {frame.dtype}")
                            print(f"Initial frame range: [{frame.min()}, {frame.max()}]")
                            
                            # Check if frame is already RGB
                            if len(frame.shape) == 3:
                                print("Camera is providing RGB data")
                                self.raw_mode = False
                            else:
                                print("Camera is providing raw data")
                                self.raw_mode = True
                            
                            frame_grab_success = True
                            break  # Exit retry loop if successful
                        else:
                            print(f"No frames received on attempt {attempt+1}/{max_retries}")
                except Exception as e:
                    print(f"Error grabbing initial frame (attempt {attempt+1}/{max_retries}): {e}")
                
                # Wait before retrying
                time.sleep(1)
            
            if not frame_grab_success:
                print("WARNING: Could not grab initial frame after multiple attempts")
                print("The application will continue, but camera functions may be limited")

        except Exception as e:
            print(f"ERROR initializing camera hardware: {e}")
            self.camera = None
            print("The application will continue without camera functionality")

    def configure_cs505cu1(self):
        """Apply optimal settings for CS505CU1 camera"""
        try:
            # Set basic camera settings
            if hasattr(self.camera, 'set_black_level'):
                self.camera.set_black_level(0)
                print("Set black level to 0")
            
            # Set gain to minimum for best image quality 
            if hasattr(self.camera, 'set_gain'):
                self.camera.set_gain(1.0)
                print("Set gain to minimum")
            
            # Additional parameters for CS505CU1
            # Check if 12-bit mode is available and set it
            if hasattr(self.camera, 'set_bit_depth'):
                self.camera.set_bit_depth(12)
                print("Set bit depth to 12-bit mode")
                
            print("CS505CU1 configured for optimal performance")
        except Exception as e:
            print(f"Error configuring CS505CU1 camera: {e}")

    def start_camera(self):
        """Start camera acquisition"""
        if self.camera:
            try:
                # Check if using direct SDK or pylablib
                if hasattr(self, '_camera_sdk') and self._camera_sdk is not None:
                    # Direct SDK camera is already armed in _init_camera_hardware
                    print("Camera acquisition started (ThorLabs SDK)")
                else:
                    # Pylablib camera
                    self.camera.start_acquisition()
                    print("Camera acquisition started (pylablib)")
                return True
            except Exception as e:
                print(f"ERROR starting camera acquisition: {e}")
                print("The application will continue without live camera feed.")
                return False
        else:
            print("No camera available to start.")
            return False

    def stop_camera(self):
        """Stop camera acquisition"""
        if self.camera:
            try:
                # Check if using direct SDK or pylablib
                if hasattr(self, '_camera_sdk') and self._camera_sdk is not None:
                    # Direct SDK camera
                    if hasattr(self.camera, 'disarm'):
                        self.camera.disarm()
                    print("Camera acquisition stopped (ThorLabs SDK)")
                else:
                    # Pylablib camera
                    self.camera.stop_acquisition()
                    print("Camera acquisition stopped (pylablib)")
            except Exception as e:
                print(f"ERROR stopping camera: {e}")

    def close_hardware(self):
        """Close all hardware connections"""
        if self.camera:
            try:
                # Check if using direct SDK or pylablib
                if hasattr(self, '_camera_sdk') and self._camera_sdk is not None:
                    # Direct SDK camera
                    if hasattr(self.camera, 'disarm'):
                        self.camera.disarm()
                    self.camera.dispose()
                    if self._mono_to_color_processor:
                        self._mono_to_color_processor.dispose()
                    if self._mono_to_color_sdk:
                        self._mono_to_color_sdk.dispose()
                    if self._camera_sdk:
                        self._camera_sdk.dispose()
                    print("ThorLabs SDK camera closed.")
                else:
                    # Pylablib camera
                    self.camera.close()
                    print("Pylablib camera closed.")
                self.camera = None
            except Exception as e:
                print(f"ERROR closing camera: {e}")
            
        # Close any open LED controllers
        self.turn_off_led()
        for led_type, handle in list(self.led_handles.items()):
            try:
                print(f"Closing {led_type} controller...")
                self.led_lib.TLUP_close(handle)
                self.led_handles.pop(led_type, None)
                self.led_max_currents.pop(led_type, None)
            except Exception as e:
                print(f"Error closing {led_type} controller: {e}")
        self.current_led_type = None
        
        if self.motion_controller:
            self.motion_controller.disconnect_all()

    def set_exposure(self, exposure_sec):
        """Set camera exposure time in seconds"""
        if self.camera:
            try:
                self.current_exposure = float(exposure_sec)
                
                # Check if using direct SDK or pylablib
                if hasattr(self, '_camera_sdk') and self._camera_sdk is not None:
                    # Direct SDK - convert to microseconds
                    self.camera.exposure_time_us = int(self.current_exposure * 1000000)
                else:
                    # Pylablib camera
                    self.camera.set_exposure(self.current_exposure)
                    
                print(f"Camera exposure set to {self.current_exposure} s")
            except Exception as e:
                print(f"ERROR setting exposure: {e}")
            
        
    def process_frame(self, frame):
        """Process a frame from the camera for display using ThorLabs SDK or fallback processing"""
        if frame is None or frame.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)  # Return empty frame
        # Store original frame for possible saving
        self.last_frame = frame.copy()
        self.last_frame_time = time.time()
        
        try:
            # Check if we're using the direct ThorLabs SDK with color processor
            if self._mono_to_color_processor is not None and self._is_color:
                # For color camera with ThorLabs SDK
                width = frame.shape[1]
                height = frame.shape[0]
                
                # Use ThorLabs color processor to transform to 24-bit RGB
                color_image_data = self._mono_to_color_processor.transform_to_24(
                    frame,
                    width,
                    height
                )
                # Reshape to proper dimensions
                color_image_data = color_image_data.reshape(height, width, 3)
                return color_image_data
            
            elif self._camera_sdk is not None and not self._is_color:
                # For monochrome camera with ThorLabs SDK - convert to 8-bit
                scaled_image = frame >> (self.bit_depth - 8)
                # Convert to RGB for display consistency
                rgb_image = cv2.cvtColor(scaled_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                return rgb_image
            
            else:
                # Fallback for pylablib camera - simple min-max normalization
                if frame.size == 0:
                    return np.zeros((100, 100, 3), dtype=np.uint8)
                
                # Apply simple normalization
                min_val = frame.min()
                max_val = frame.max()
                
                if max_val > min_val:
                    # Scale to 0-255 range
                    scaled = (frame - min_val) / (max_val - min_val) * 255.0
                    normalized = scaled.astype(np.uint8)
                else:
                    normalized = np.full_like(frame, 127, dtype=np.uint8)
                
                # Convert to RGB if needed
                if len(normalized.shape) == 2:
                    normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
                
                return normalized
        
        except Exception as e:
            print(f"Error in frame processing: {e}")
            # Return a gray frame in case of error
            return np.full((480, 640, 3), 128, dtype=np.uint8)
            
    def move_stage(self, x=None, y=None, z=None, wait_for_completion=True):
        """Move stage by relative amounts in mm"""
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
        
        # Wait for all movements to complete
        if wait_for_completion:
            self._wait_for_moves_complete()

    def _wait_for_moves_complete(self):
        """Wait for all move workers to complete"""
        while self.move_workers:
            for worker in self.move_workers[:]:  # Create a copy of the list to avoid modification during iteration
                if not worker.isRunning():
                    self._cleanup_worker(worker)
            time.sleep(0.1)  # Small delay to prevent CPU hogging

    def _cleanup_worker(self, worker):
        if worker in self.move_workers:
            self.move_workers.remove(worker)

    def _move_axis_relative(self, axis, step_mm):
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
        try:
            device.MoveTo(Decimal(new_pos), Int32(60000))
        except Exception as e:
            print(f"[{axis}] MoveTo command failed: {e}")

    def move_to_absolute(self, x=None, y=None, z=None, wait_for_completion=True):
        """Move axes to absolute positions in mm using relative movements"""
        # Get current positions
        current_x = self.motion_controller._get_coordinate("X") if x is not None else None
        current_y = self.motion_controller._get_coordinate("Y") if y is not None else None
        current_z = self.motion_controller._get_coordinate("Z") if z is not None else None
        
        # Calculate relative movements
        delta_x = x - current_x if x is not None else None
        delta_y = y - current_y if y is not None else None
        delta_z = z - current_z if z is not None else None
        
        # Move all axes simultaneously
        self.move_stage(x=delta_x, y=delta_y, z=delta_z, wait_for_completion=wait_for_completion)

    def set_optic(self, optic_name):
        """Set the current optic and update related parameters"""
        if optic_name in self.optics_params:
            self.current_optic = optic_name
            params = self.optics_params[optic_name]
            self.fov_x = params["fov_x"]
            self.fov_y = params["fov_y"]
            self.z_zero = self.z_zero_positions[optic_name]  # Use z_zero_positions
            return True
        return False
    
    def zero_stage_at_optic_center(self):
        """Move the Z axis to the correct zero position for the current optic"""
        if self.motion_controller:
            try:
                # Get the absolute z-position for the current optic
                target_z = self.z_zero  # This is now an absolute position
                
                # Move Z to the absolute position defined for the current optic
                self.move_to_absolute(x=self.motion_stage_center_x,y=self.motion_stage_center_y,z=target_z, wait_for_completion=False)
                
                print(f"Moved Z axis to {target_z} mm (zero position for {self.current_optic}) and center stage")
                return True
            except Exception as e:
                print(f"Error zeroing stage: {e}")
                return False
        return False
    
    def turn_off_led(self):
        """Turn off the current LED"""
        if self.current_led_type and self.current_led_type in self.led_handles:
            handle = self.led_handles[self.current_led_type]
            self.led_lib.TLUP_switchLedOutput(handle, 0)
            # Emit a signal that the LED has been turned off (to be connected to UI)
            return True
        return False

    def save_last_frame(self, filename, foldername):
        """Save the most recently captured frame to the foldername with filename"""
        try:
            # Capture a fresh frame
            with QMutexLocker(self.camera_lock):
                if not self.camera:
                    print("[SAVE] No camera available to save!")
                    return None

                # Calculate total frame time (exposure + readout)
                exposure_time_us = self.camera.exposure_time_us
                readout_time_ns = self.camera.sensor_readout_time_ns
                total_frame_time_sec = (exposure_time_us / 1000000.0) + (readout_time_ns / 1000000000.0)
                
                # Add a small safety margin
                total_frame_time_sec += 0.1  # Add 100ms margin
                
                # Try up to 3 times to capture a frame
                max_attempts = 6
                frame = None
                
                for attempt in range(max_attempts):
                    # Trigger and wait for complete frame
                    self.camera.issue_software_trigger()
                    time.sleep(total_frame_time_sec)  # Wait for frame to complete
                    
                    # Try to get the frame
                    timeout_ms = 5000
                    start_time = time.time()
                    
                    while time.time() - start_time < timeout_ms/1000:
                        camera_frame = self.camera.get_pending_frame_or_null()
                        if camera_frame is not None:
                            frame = camera_frame.image_buffer
                            break
                        time.sleep(0.01)
                    
                    if frame is not None:
                        break  # Success, exit the retry loop
                    
                    print(f"[SAVE] Attempt {attempt + 1}/{max_attempts} failed to capture frame, trying again...")
                
                if frame is None:
                    print(f"[SAVE] Failed to capture frame after {max_attempts} attempts")
                    return None
            
            # Create a timestamp for the filename
            tiff_filename = filename
            tiff_path = os.path.join(foldername, tiff_filename)
            
            # Save as 16-bit TIFF
            if self._mono_to_color_processor is not None and self._is_color:
                try:
                    width = frame.shape[1]
                    height = frame.shape[0]
                    
                    # Use ThorLabs color processor to transform to 48-bit RGB (16 bits per channel)
                    color_image_data = self._mono_to_color_processor.transform_to_48(
                        frame,
                        width,
                        height
                    )
                    # Reshape to proper dimensions
                    color_image_data = color_image_data.reshape(height, width, 3)
                    
                    # Scale to use full 16-bit range (0-65535)
                    # First normalize to 0-1 range
                    min_val = color_image_data.min()
                    max_val = color_image_data.max()
                    if max_val > min_val:
                        normalized = (color_image_data.astype(np.float32) - min_val) / (max_val - min_val)
                        # Scale to 16-bit range
                        scaled = (normalized * 65535).astype(np.uint16)
                    else:
                        scaled = color_image_data.astype(np.uint16)
                    
                    # Convert RGB to BGR for OpenCV
                    scaled = cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR)
                    
                    # Save as 16-bit TIFF
                    cv2.imwrite(tiff_path, scaled)
                    print(f"16-bit image saved to {tiff_path}")
                    return tiff_path
                    
                except Exception as e:
                    print(f"Failed to save 16-bit version: {e}")
                    return None
            else:
                # For monochrome cameras, save as 16-bit TIFF
                cv2.imwrite(tiff_path, frame.astype(np.uint16))
                print(f"16-bit monochrome image saved to {tiff_path}")
                return tiff_path
                
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None


###############################################################################
# GridDisplayWidget: Displays a grid of images for the preview field feature
###############################################################################
class GridDisplayWidget(QLabel):
    # Define signals
    move_to_position = pyqtSignal(float, float)
    
    def __init__(self, hardware, parent=None):
        super().__init__(parent)
        self.hardware = hardware
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        
        # Grid properties
        self.grid_width = 0
        self.grid_height = 0
        self.grid_images = []
        self.cell_width = 0
        self.cell_height = 0
        
        # Display options
        self.show_mosaic = True  # Always show mosaic
        self.mosaic_image = None
        
        # Current position indicator (green box)
        self.current_x = -1  # mm
        self.current_y = -1  # mm
        self.show_current_position = False
        self.show_current_roi = False
        self.current_fov_width = 12.0  # mm for 4X optic by default
        self.current_fov_height = 10.0  # mm for 4X optic by default
        
    def setup_grid(self, width, height):
        """Initialize the grid dimensions"""
        self.grid_width = width
        self.grid_height = height
        self.grid_images = [[None for _ in range(width)] for _ in range(height)]
        self.update()
        
    def clear_grid(self):
        """Clear all images from the grid"""
        if self.grid_width > 0 and self.grid_height > 0:
            self.grid_images = [[None for _ in range(self.grid_width)] for _ in range(self.grid_height)]
            self.mosaic_image = None
            self.update()
            
    def add_image(self, image, grid_x, grid_y):
        """Add an image to the grid at the specified position"""
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.grid_images[grid_y][grid_x] = image
            self.update()
            
    def set_mosaic(self, mosaic):
        """Set the full stitched mosaic image"""
        self.mosaic_image = mosaic
        self.show_mosaic = True  # Always show mosaic
        self.update()
        
  
    
    def update_current_position(self, x_mm, y_mm, optic_name=None):
        """Update the current position indicator"""
        if optic_name:
            self.current_optic = optic_name
            
        # Calculate the total field of view including camera FOV
        fov_x = self.hardware.fov_x
        fov_y = self.hardware.fov_y
        
        # Calculate total field size based on grid dimensions
        total_field_x = self.grid_max_vis_x - self.grid_min_vis_x
        total_field_y = self.grid_max_vis_y - self.grid_min_vis_y
        
        # Calculate position as percentage of total field
        x_percent = (x_mm - self.grid_min_vis_x) / total_field_x
        y_percent = 1.0 - (y_mm - self.grid_min_vis_y) / total_field_y  # Invert y-axis
        
        # Calculate box size as percentage of total field
        box_width_percent = 1/self.grid_width
        box_height_percent = 1/self.grid_height
        
        # Store the position and size
        self.current_x = x_percent
        self.current_y = y_percent
        self.current_box_width = box_width_percent
        self.current_box_height = box_height_percent
        
        # Show the current position indicator
        self.show_current_position = True
        self.show_current_roi = True
        # Update the display
        self.update()

    def _draw_current_position(self):
        """Draw the current position indicator"""
        if self.current_x is None or self.current_y is None:
            return
            
        painter = QPainter(self)
        painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        
        #recalulate the box position and size based on the new scale
        self.current_box_width = self.hardware.fov_x / (self.grid_max_vis_x-self.grid_min_vis_x)
        self.current_box_height = self.hardware.fov_y / (self.grid_max_vis_y-self.grid_min_vis_y)

        if self.show_mosaic and self.mosaic_image is not None:
            # Scale relative to mosaic
            mosaic_height, mosaic_width = self.mosaic_image.shape[:2]
            widget_width = self.width()
            widget_height = self.height()
            
            # Calculate scaling factors to maintain aspect ratio
            scale_x = widget_width / mosaic_width
            scale_y = widget_height / mosaic_height
            scale = min(scale_x, scale_y)
            
            # Calculate offsets to center the mosaic
            offset_x = (widget_width - mosaic_width * scale) / 2
            offset_y = (widget_height - mosaic_height * scale) / 2

            # Calculate box position and size in pixels
            box_x = int(offset_x + self.current_x * mosaic_width * scale)
            box_y = int(offset_y + self.current_y * mosaic_height * scale)
            box_width = int(self.current_box_width * mosaic_width * scale)
            box_height = int(self.current_box_height * mosaic_height * scale)
        else:
            # Scale relative to grid
            if self.grid_width == 0 or self.grid_height == 0:
                return
                
            # Calculate the aspect ratio of the grid
            grid_aspect = self.hardware.fov_x / self.hardware.fov_y
            
            # Calculate the maximum size that fits in the widget while maintaining aspect ratio
            widget_width = self.width()
            widget_height = self.height()
            widget_aspect = widget_width / widget_height
            
            if widget_aspect > grid_aspect:
                # Widget is wider than grid aspect ratio
                cell_height = widget_height / self.grid_height
                cell_width = cell_height * grid_aspect
            else:
                # Widget is taller than grid aspect ratio
                cell_width = widget_width / self.grid_width
                cell_height = cell_width / grid_aspect
            
            # Center the grid in the widget
            total_width = cell_width * self.grid_width
            total_height = cell_height * self.grid_height
            x_offset = (widget_width - total_width) / 2
            y_offset = (widget_height - total_height) / 2
            
            # Calculate box position and size in pixels
            box_x = int(x_offset + self.current_x * total_width)
            box_y = int(y_offset + self.current_y * total_height)
            box_width = int(self.current_box_width * total_width)
            box_height = int(self.current_box_height * total_height)
        
        # Draw the box
        painter.drawRect(int(box_x-box_width/2), int(box_y-box_height/2), box_width, box_height)
        painter.end()


    def _draw_current_ROI(self):
        """Draw a red rectangle showing the current region of interest for auto-capture"""
        if not hasattr(self, 'grid_width') or not hasattr(self, 'grid_height'):
            return

        # Get pattern size from auto-capture controls
        main_window = self.window()  # Get the main window
        if hasattr(main_window, 'control_panel'):
            pattern_width = main_window.control_panel.pattern_x_spin.value()
            pattern_height = main_window.control_panel.pattern_y_spin.value()
        else:
            # Default values if control panel not available
            print("No control panel found - using default pattern size")
            pattern_width = 15.0
            pattern_height = 15.0

        scale_width = pattern_width / self.hardware.fov_x
        scale_height = pattern_height / self.hardware.fov_y
            
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        
        if self.show_mosaic and self.mosaic_image is not None:
            # Scale relative to mosaic
            mosaic_height, mosaic_width = self.mosaic_image.shape[:2]
            widget_width = self.width()
            widget_height = self.height()
            
            # Calculate scaling factors to maintain aspect ratio
            scale_x = widget_width / mosaic_width
            scale_y = widget_height / mosaic_height
            scale = min(scale_x, scale_y)
            
            # Calculate offsets to center the mosaic
            offset_x = (widget_width - mosaic_width * scale) / 2
            offset_y = (widget_height - mosaic_height * scale) / 2
            
            # Calculate box position and size in pixels
            box_x = int(offset_x + self.current_x * mosaic_width * scale)
            box_y = int(offset_y + self.current_y * mosaic_height * scale)
            box_width = int(self.current_box_width * mosaic_width * scale * scale_width)
            box_height = int(self.current_box_height * mosaic_height * scale * scale_height)
        else:
            # Scale relative to grid
            if self.grid_width == 0 or self.grid_height == 0:
                return
                
            # Calculate the aspect ratio of the grid
            grid_aspect = self.hardware.fov_x / self.hardware.fov_y
            
            # Calculate the maximum size that fits in the widget while maintaining aspect ratio
            widget_width = self.width()
            widget_height = self.height()
            widget_aspect = widget_width / widget_height
            
            if widget_aspect > grid_aspect:
                # Widget is wider than grid aspect ratio
                cell_height = widget_height / self.grid_height
                cell_width = cell_height * grid_aspect
            else:
                # Widget is taller than grid aspect ratio
                cell_width = widget_width / self.grid_width
                cell_height = cell_width / grid_aspect
            
            # Center the grid in the widget
            total_width = cell_width * self.grid_width
            total_height = cell_height * self.grid_height
            x_offset = (widget_width - total_width) / 2
            y_offset = (widget_height - total_height) / 2
            
            # Calculate box position and size in pixels
            box_x = int(x_offset + self.current_x * total_width)
            box_y = int(y_offset + self.current_y * total_height)
            box_width = int(self.current_box_width * total_width * scale_width)
            box_height = int(self.current_box_height * total_height * scale_height)
        
        # Draw the box
        painter.drawRect(int(box_x-box_width/2), int(box_y-box_height/2), box_width, box_height)
        painter.end()
        
        
    def mousePressEvent(self, event):
        """Handle mouse clicks to move to position"""
        if not self.grid_width or not self.grid_height:
            return
            
        # Get click position as percentage of widget size
        click_x = event.x() / self.width()
        click_y = 1.0 - (event.y() / self.height())  # Invert y-axis
            
        # Calculate total field size including camera FOV
        total_field_x = self.grid_max_vis_x - self.grid_min_vis_x
        total_field_y = self.grid_max_vis_y - self.grid_min_vis_y
        
        # Convert percentage to mm coordinates
        target_x = click_x * total_field_x + self.grid_min_vis_x
        target_y = click_y * total_field_y + self.grid_min_vis_y
        
        # Emit the move signal with the target coordinates
        self.move_to_position.emit(target_x, target_y)
            
    def paintEvent(self, event):
        """Draw either the grid of images or the full mosaic"""
        super().paintEvent(event)
        
        if self.show_mosaic and self.mosaic_image is not None:
            self._draw_mosaic()
        else:
            self._draw_grid()
            
        # Draw current position indicator if showing
        if self.show_current_position:
            self._draw_current_position()
            
        # Draw current ROI if showing
        if self.show_current_roi:
            self._draw_current_ROI()
            
    def _draw_mosaic(self):
        """Draw the full stitched mosaic"""
        if self.mosaic_image is None:
            return
            
        # Convert numpy array to QImage for display
        if len(self.mosaic_image.shape) == 3:
            height, width, channels = self.mosaic_image.shape
            bytes_per_line = channels * width
            format = QImage.Format_RGB888
        else:
            height, width = self.mosaic_image.shape
            bytes_per_line = width
            format = QImage.Format_Grayscale8
            
        q_img = QImage(self.mosaic_image.data, width, height, bytes_per_line, format)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale to fit the widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Draw the mosaic centered in the widget
        painter = QPainter(self)
        point = QPoint((self.width() - scaled_pixmap.width()) // 2,
                      (self.height() - scaled_pixmap.height()) // 2)
        painter.drawPixmap(point, scaled_pixmap)
        
    def _draw_grid(self):
        """Draw the grid of images"""
        if self.grid_width == 0 or self.grid_height == 0:
            return
            
        # Calculate the aspect ratio based on FOV dimensions instead of grid dimensions
        fov_aspect = self.hardware.fov_x / self.hardware.fov_y
        
        # Calculate the maximum size that fits in the widget while maintaining aspect ratio
        widget_width = self.width()
        widget_height = self.height()
        widget_aspect = widget_width / widget_height
        
        if widget_aspect > fov_aspect:
            # Widget is wider than FOV aspect ratio
            cell_height = widget_height / self.grid_height
            cell_width = cell_height * fov_aspect
        else:
            # Widget is taller than FOV aspect ratio
            cell_width = widget_width / self.grid_width
            cell_height = cell_width / fov_aspect
        
        # Center the grid in the widget
        total_width = cell_width * self.grid_width
        total_height = cell_height * self.grid_height
        x_offset = (widget_width - total_width) / 2
        y_offset = (widget_height - total_height) / 2
        
        # Draw each cell
        painter = QPainter(self)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Calculate cell rect - the grid display is bottom-up to match physical motion
                rect = QRect(
                    int(x_offset + x * cell_width),
                    int(y_offset + y * cell_height),
                    int(cell_width),
                    int(cell_height)
                )
                
                # Draw the cell
                if self.grid_images[y][x] is not None:
                    # Cell has an image
                    img = self.grid_images[y][x]
                    if len(img.shape) == 3:  # Color image
                        height, width, channel = img.shape
                        bytes_per_line = 3 * width
                        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    else:  # Monochrome image
                        height, width = img.shape
                        # Convert to RGB by repeating the monochrome channel
                        rgb_img = np.stack((img,) * 3, axis=-1)
                        bytes_per_line = 3 * width
                        q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    
                    pixmap = QPixmap.fromImage(q_img)
                    scaled_pixmap = pixmap.scaled(
                        int(cell_width), int(cell_height), 
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    
                    # Center the image in the cell
                    x_img_offset = (cell_width - scaled_pixmap.width()) / 2
                    y_img_offset = (cell_height - scaled_pixmap.height()) / 2
                    painter.drawPixmap(
                        int(rect.x() + x_img_offset),
                        int(rect.y() + y_img_offset),
                        scaled_pixmap
                    )
                else:
                    # Empty cell - draw a border
                    painter.setPen(QPen(Qt.gray, 1, Qt.SolidLine))
                    painter.drawRect(rect)
        
        painter.end()


###############################################################################
# CameraDisplayWidget 
###############################################################################
class CameraDisplayWidget(QLabel):
    def __init__(self, hardware, parent=None):
        super().__init__(parent)
        self.hardware = hardware
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("QLabel { background-color: black; }")
        
        # FPS calculation
        self.last_update_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.fps_update_interval = 1.0  # Update FPS every second
        self.show_fps = False
        
        # Mouse position
        self.last_mouse_pos = QPoint(0, 0)
    
    def update_frame(self, frame):
        """Update the display with a new camera frame"""
        if frame is None:
            return
        display_frame = frame
            
        # Update FPS calculation
        current_time = time.time()
        self.frame_count += 1
        elapsed = current_time - self.last_update_time
        if elapsed >= self.fps_update_interval:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_update_time = current_time
            self.update()  # Trigger a repaint to update FPS display
            
        # Convert frame to QImage and display
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale the pixmap to fit the widget size while preserving aspect ratio
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
        self.hardware.move_stage(x=move_x, y=move_y, wait_for_completion=False)


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
        
        # X-axis controls
        stage_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_coord_label = QLabel("0.00 mm")
        stage_layout.addWidget(self.x_coord_label, 0, 1)
        
        x_neg_button = QPushButton("◄")
        x_neg_button.clicked.connect(lambda: self.hardware.move_stage(x=-self.step_value(), y=None, z=None, wait_for_completion=False))
        stage_layout.addWidget(x_neg_button, 0, 2)
        
        x_pos_button = QPushButton("►")
        x_pos_button.clicked.connect(lambda: self.hardware.move_stage(x=self.step_value(), y=None, z=None, wait_for_completion=False))
        stage_layout.addWidget(x_pos_button, 0, 3)
        
        # Y-axis controls
        stage_layout.addWidget(QLabel("Y:"), 1, 0)
        self.y_coord_label = QLabel("0.00 mm")
        stage_layout.addWidget(self.y_coord_label, 1, 1)
        
        y_neg_button = QPushButton("▼")
        y_neg_button.clicked.connect(lambda: self.hardware.move_stage(x=None, y=-self.step_value(), z=None, wait_for_completion=False))
        stage_layout.addWidget(y_neg_button, 1, 2)
        
        y_pos_button = QPushButton("▲")
        y_pos_button.clicked.connect(lambda: self.hardware.move_stage(x=None, y=self.step_value(), z=None, wait_for_completion=False))
        stage_layout.addWidget(y_pos_button, 1, 3)
        
        # Z-axis controls
        stage_layout.addWidget(QLabel("Z:"), 2, 0)
        self.z_coord_label = QLabel("0.00 mm")
        stage_layout.addWidget(self.z_coord_label, 2, 1)
        
        z_neg_button = QPushButton("▼")
        z_neg_button.clicked.connect(lambda: self.hardware.move_stage(x=None, y=None, z=-self.step_value(), wait_for_completion=False))
        stage_layout.addWidget(z_neg_button, 2, 2)
        
        z_pos_button = QPushButton("▲")
        z_pos_button.clicked.connect(lambda: self.hardware.move_stage(x=None, y=None, z=self.step_value(), wait_for_completion=False))
        stage_layout.addWidget(z_pos_button, 2, 3)
        
        # Step size
        stage_layout.addWidget(QLabel("Step (mm):"), 3, 0)
        self.step_spinbox = QDoubleSpinBox()
        self.step_spinbox.setRange(0.001, 10.0)
        self.step_spinbox.setValue(0.1)
        self.step_spinbox.setSingleStep(0.1)
        stage_layout.addWidget(self.step_spinbox, 3, 1)
        
        # Zero button
        zero_button = QPushButton("Zero")
        zero_button.clicked.connect(self.zero_stage)
        stage_layout.addWidget(zero_button, 3, 2, 1, 2)  # span 2 columns
        
        # Optic selector - limited to 4X and 10X only
        stage_layout.addWidget(QLabel("Optic:"), 4, 0)
        self.optic_combo = QComboBox()
        self.optic_combo.addItems(["4X", "10X"])  # Only 4X and 10X are supported
        self.optic_combo.setCurrentIndex(0)  # default to 4X
        self.optic_combo.currentIndexChanged.connect(self.update_optic)
        stage_layout.addWidget(self.optic_combo, 4, 1, 1, 3)  # span 3 columns
        
        layout.addWidget(stage_group)
        
        # LED Controls
        led_group = QGroupBox("LED Controls")
        led_layout = QGridLayout()
        led_group.setLayout(led_layout)
        
        # LED type selector
        led_layout.addWidget(QLabel("LED Type:"), 0, 0)
        self.led_combo = QComboBox()
        self.led_combo.addItem("None")
        # Add detected LEDs
        if hasattr(self.hardware, 'led_controllers'):
            for led_type in self.hardware.led_controllers.keys():
                self.led_combo.addItem(led_type)
        self.led_combo.currentIndexChanged.connect(self.led_selection_changed)
        led_layout.addWidget(self.led_combo, 0, 1, 1, 2)  # span 2 columns
        
        # LED enable checkbox
        self.led_enable = QCheckBox("Enable")
        self.led_enable.setChecked(False)
        self.led_enable.stateChanged.connect(self.led_enable_toggled)
        led_layout.addWidget(self.led_enable, 1, 0)
        
        # LED intensity slider
        led_layout.addWidget(QLabel("Intensity:"), 1, 1)
        self.led_slider = QSlider(Qt.Horizontal)
        self.led_slider.setRange(0, 100)
        self.led_slider.setValue(50)
        self.led_slider.valueChanged.connect(self.update_led_intensity_from_slider)
        led_layout.addWidget(self.led_slider, 1, 2)
        
        # LED intensity manual input
        led_layout.addWidget(QLabel("Intensity (%):"), 2, 0)
        self.led_intensity_input = QLineEdit()
        self.led_intensity_input.setText("50")
        self.led_intensity_input.setValidator(QIntValidator(0, 100))
        self.led_intensity_input.editingFinished.connect(self.update_led_intensity_from_text)
        led_layout.addWidget(self.led_intensity_input, 2, 1, 1, 2)
        
        layout.addWidget(led_group)
        
        # Camera Controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QGridLayout()
        camera_group.setLayout(camera_layout)
        
        # Exposure slider
        camera_layout.addWidget(QLabel("Exposure:"), 0, 0)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 200)  # 0.001 to 2.0 seconds (increased max)
        self.exposure_slider.setValue(10)       # default 0.1 seconds
        self.exposure_slider.valueChanged.connect(self.update_exposure_from_slider)
        camera_layout.addWidget(self.exposure_slider, 0, 1)
        
        # Exposure value label
        self.exposure_label = QLabel("0.1 s")
        camera_layout.addWidget(self.exposure_label, 0, 2)
        
        
        # Exposure manual input
        camera_layout.addWidget(QLabel("Exposure (s):"), 1, 0)
        self.exposure_input = QLineEdit()
        self.exposure_input.setText("0.1")
        self.exposure_input.setValidator(QDoubleValidator(0.001, 2.0, 3))
        self.exposure_input.editingFinished.connect(self.update_exposure_from_text)
        camera_layout.addWidget(self.exposure_input, 1, 1, 1, 2)
        
        layout.addWidget(camera_group)
        
        # Auto-Capture Controls
        capture_group = QGroupBox("Capture Controls")
        capture_layout = QGridLayout()
        capture_group.setLayout(capture_layout)
        
        # Single capture button
        self.capture_button = QPushButton("Save Current Photo")
        self.capture_button.clicked.connect(self.save_current_photo_clicked)
        capture_layout.addWidget(self.capture_button, 0, 0, 1, 2)  # span 2 columns
        
        # Preview Field button
        self.preview_field_button = QPushButton("Preview Field")
        # Don't use lambda with parent() - it's not reliable
        # Instead, we'll connect this in MainWindow after setup
        capture_layout.addWidget(self.preview_field_button, 1, 0, 1, 2)  # span 2 columns
        
        # Auto-capture button
        self.auto_capture_button = QPushButton("Auto Capture")
        self.auto_capture_button.clicked.connect(self.auto_capture_clicked)
        capture_layout.addWidget(self.auto_capture_button, 2, 0, 1, 2)  # span 2 columns
        
        # Stop button
        self.stop_button = QPushButton("Stop Capture")
        self.stop_button.clicked.connect(lambda: self.window().stop_capture_processes() if self.window() else None)
        self.stop_button.setEnabled(False)
        capture_layout.addWidget(self.stop_button, 3, 0, 1, 2)  # span 2 columns
        
        # Capture type
        capture_layout.addWidget(QLabel("Capture Type:"), 4, 0)
        self.capture_type_combo = QComboBox()
        self.capture_type_combo.addItems(["XY Grid", "Z Stack", "XYZ Combined"])
        capture_layout.addWidget(self.capture_type_combo, 4, 1)
        
        # Pattern Size X
        capture_layout.addWidget(QLabel("Grid Width (mm):"), 5, 0)
        self.pattern_x_spin = QDoubleSpinBox()
        self.pattern_x_spin.setRange(1.0, 100.0)
        self.pattern_x_spin.setValue(15.0)
        self.pattern_x_spin.setSingleStep(1.0)
        self.pattern_x_spin.setDecimals(1)
        self.pattern_x_spin.valueChanged.connect(self._update_roi_display)
        capture_layout.addWidget(self.pattern_x_spin, 5, 1)
        
        # Pattern Size Y
        capture_layout.addWidget(QLabel("Grid Height (mm):"), 6, 0)
        self.pattern_y_spin = QDoubleSpinBox()
        self.pattern_y_spin.setRange(1.0, 100.0)
        self.pattern_y_spin.setValue(15.0)
        self.pattern_y_spin.setSingleStep(1.0)
        self.pattern_y_spin.setDecimals(1)
        self.pattern_y_spin.valueChanged.connect(self._update_roi_display)
        capture_layout.addWidget(self.pattern_y_spin, 6, 1)
        
        # Overlap %
        capture_layout.addWidget(QLabel("Overlap (%):"), 7, 0)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 75)
        self.overlap_spin.setValue(25)
        capture_layout.addWidget(self.overlap_spin, 7, 1)
        
        # Z Stack Settings
        capture_layout.addWidget(QLabel("Z Stack Steps:"), 8, 0)
        self.z_stack_spin = QSpinBox()
        self.z_stack_spin.setRange(2, 100)
        self.z_stack_spin.setValue(5)
        capture_layout.addWidget(self.z_stack_spin, 8, 1)
        
        # Z Stack Height
        capture_layout.addWidget(QLabel("Z Height (mm):"), 9, 0)
        self.z_height_spin = QDoubleSpinBox()
        self.z_height_spin.setRange(0.001, 10.0)
        self.z_height_spin.setValue(0.5)
        self.z_height_spin.setSingleStep(0.1)
        self.z_height_spin.setDecimals(2)
        capture_layout.addWidget(self.z_height_spin, 9, 1)
        
        # Progress bar
        capture_layout.addWidget(QLabel("Progress:"), 10, 0)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        capture_layout.addWidget(self.progress_bar, 10, 1)
        
        # Status label
        self.status_label = QLabel("Ready")
        capture_layout.addWidget(self.status_label, 11, 0, 1, 2)  # span 2 columns
        
        layout.addWidget(capture_group)
        
        # Stretch to fill space
        layout.addStretch()
        
        # Set initial state
        self.update_coordinates()
        self.update_led_slider_from_intensity()
        
        # Set default LED type in combo if hardware has one connected
        if hasattr(self.hardware, 'current_led_type') and self.hardware.current_led_type:
            index = self.led_combo.findText(self.hardware.current_led_type)
            if index >= 0:
                self.led_combo.setCurrentIndex(index)
                self.led_enable.setEnabled(True)
        
        # Setup timer for periodic UI updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_coordinates)
        self.update_timer.start(500)  # Update every 500ms
    
    def step_value(self):
        return self.step_spinbox.value()
    
    def update_coordinates(self):
        """Update the coordinate labels based on current position"""
        # Only update if we have a motion controller
        if not self.hardware.motion_controller:
            return
            
        try:
            # Get current coordinates based on how the motion controller stores positions
            x_pos = 0
            y_pos = 0
            z_pos = 0
            
            mc = self.hardware.motion_controller
            
            # Check which attribute structure is used by the motion controller
            if hasattr(mc, 'stages') and isinstance(mc.stages, dict):
                # New structure using stages dictionary
                x_pos = mc.stages.get("X", {}).get("position", 0)
                y_pos = mc.stages.get("Y", {}).get("position", 0)
                z_pos = mc.stages.get("Z", {}).get("position", 0)
            elif hasattr(mc, 'devices') and isinstance(mc.devices, dict):
                # Old structure using devices dictionary
                for axis, device in mc.devices.items():
                    if axis == 'X' and hasattr(device, 'Position'):
                        x_pos = float(str(device.Position))
                    elif axis == 'Y' and hasattr(device, 'Position'):
                        y_pos = float(str(device.Position))
                    elif axis == 'Z' and hasattr(device, 'Position'):
                        z_pos = float(str(device.Position))
            
            # Update labels
            self.x_coord_label.setText(f"{x_pos:.2f} mm")
            self.y_coord_label.setText(f"{y_pos:.2f} mm")
            self.z_coord_label.setText(f"{z_pos:.2f} mm")
            
            # Also check if LED is actually on, and update the UI if needed
            if self.led_enable.isChecked() and self.hardware.current_led_type:
                try:
                    # Check if the LED is actually on
                    handle = self.hardware.led_handles.get(self.hardware.current_led_type, None)
                    if handle:
                        # This would require a way to check the actual LED state
                        # If there's no direct way, consider adding a flag in the hardware class
                        pass
                except Exception:
                    pass
        except Exception as e:
            print(f"Error updating coordinates: {e}")
    
    def update_optic(self):
        """Update field of view and Z zero point based on selected optic"""
        optic = self.optic_combo.currentText()
        # Set the hardware optic parameters
        if self.hardware.set_optic(optic):
            # Update relevant UI elements
            # Adjust Z height based on the optic
            if optic == "4X":
                self.z_height_spin.setValue(0.5)  # Default Z height for 4X
            elif optic == "10X":
                self.z_height_spin.setValue(0.2)  # Default Z height for 10X
    
    
    def led_selection_changed(self):
        """Handle LED selection change"""
        # Turn off current LED before switching
        if self.led_enable.isChecked():
            self.hardware.turn_off_led()
            self.led_enable.setChecked(False)  # Ensure checkbox is unchecked
        
        # Connect to new LED
        requested_led_type = self.led_combo.currentText()
        if requested_led_type == "None":
            self._update_led_ui_state()
            return
            
        if self.hardware.connect_led(requested_led_type):
            self._update_led_ui_state()
        else:
            QMessageBox.warning(self, "LED Connection Error", 
                              f"Failed to connect to {requested_led_type} LED")
    
    @pyqtSlot(int)
    def led_enable_toggled(self, state):
        """Toggle LED on/off"""
        if state == Qt.Checked:
            self.hardware.turn_on_led()
        else:
            self.hardware.turn_off_led()
    
    def update_led_intensity_from_slider(self, slider_value):
        """Update LED intensity from slider value"""
        # Set LED intensity (0-100%)
        intensity = slider_value
        self.hardware.set_led_intensity(intensity)
        
        # Update the text field
        self.led_intensity_input.setText(str(intensity))
    
    def update_led_intensity_from_text(self):
        """Update LED intensity from text input"""
        try:
            intensity = int(self.led_intensity_input.text())
            if 0 <= intensity <= 100:
                # Update slider
                self.led_slider.setValue(intensity)
                # Update hardware
                self.hardware.set_led_intensity(intensity)
        except ValueError:
            # If invalid, reset to current slider value
            self.led_intensity_input.setText(str(self.led_slider.value()))
    
    def update_led_slider_from_intensity(self):
        """Update slider based on current LED intensity"""
        # This would be called when changing LED type
        # For now, just reset to 50%
        self.led_slider.setValue(50)
        self.led_intensity_input.setText("50")
    
    def update_exposure_from_slider(self, slider_value):
        """Update exposure from slider value"""
        # Convert 1-200 to 0.001-2.0 seconds
        exposure_sec = slider_value / 100.0
        
        # Update hardware
        self.hardware.set_exposure(exposure_sec)
        
        # Update label and text field
        self.exposure_label.setText(f"{exposure_sec:.3f} s")
        self.exposure_input.setText(f"{exposure_sec:.3f}")
    
    def update_exposure_from_text(self):
        """Update exposure from text input"""
        try:
            exposure_sec = float(self.exposure_input.text())
            if 0.001 <= exposure_sec <= 2.0:
                # Update slider (value 1-200)
                slider_value = int(exposure_sec * 100)
                self.exposure_slider.setValue(slider_value)
                
                # Update hardware directly
                self.hardware.set_exposure(exposure_sec)
                
                # Update label
                self.exposure_label.setText(f"{exposure_sec:.3f} s")
        except ValueError:
            # If invalid, reset to current slider value
            exposure_sec = self.exposure_slider.value() / 100.0
            self.exposure_input.setText(f"{exposure_sec:.3f}")
    
    def update_slider_from_exposure(self):
        """Update slider based on current exposure"""
        if hasattr(self.hardware, 'current_exposure'):
            exposure_val = min(max(self.hardware.current_exposure, 0.001), 2.0)
            slider_val = int(exposure_val * 100)
            self.exposure_slider.setValue(slider_val)
            self.exposure_label.setText(f"{exposure_val:.3f} s")
            self.exposure_input.setText(f"{exposure_val:.3f}")
    
    def update_progress(self, completed, total, remaining):
        """Update auto-capture progress"""
        if total > 0:
            progress = int(100 * completed / total)
            self.progress_bar.setValue(progress)
            
            # Format remaining time
            if remaining > 60:
                mins = int(remaining / 60)
                secs = int(remaining % 60)
                self.status_label.setText(f"Capturing: {completed}/{total} ({mins}m {secs}s remaining)")
            else:
                self.status_label.setText(f"Capturing: {completed}/{total} ({int(remaining)}s remaining)")
    
    def auto_capture_clicked(self):
        """Start auto-capture sequence"""
        if self.hardware.auto_capture_in_progress:
            self.status_label.setText("Capture already in progress")
            return
            
        # Get parameters
        capture_mode = self.capture_type_combo.currentText()
        overlap_percent = self.overlap_spin.value()
        num_z_stack = self.z_stack_spin.value()
        
        # Get grid dimensions in mm
        pattern_width_mm = self.pattern_x_spin.value()
        pattern_height_mm = self.pattern_y_spin.value()
        z_height_mm = self.z_height_spin.value()
        
        # Validate parameters
        if pattern_width_mm <= 0 or pattern_height_mm <= 0:
            self.status_label.setText("Error: Grid dimensions must be greater than 0")
            return
        
        if z_height_mm <= 0:
            self.status_label.setText("Error: Z height must be greater than 0")
            return
            
        # Get save directory
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Save Directory",
            os.path.expanduser("~/Downloads"),
            QFileDialog.ShowDirsOnly
        )
        
        if not save_dir:
            self.status_label.setText("Capture cancelled")
            return
            
        # Set capture folder
        self.hardware.auto_capture_folder = save_dir
        
        # Update UI
        self.auto_capture_button.setEnabled(False)
        self.status_label.setText("Starting capture...")
        self.progress_bar.setValue(0)
        
        # Enable stop button
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(True)
        
        # Convert capture mode to worker mode
        mode_map = {
            "XY Grid": "xy",
            "Z Stack": "z",
            "XYZ Combined": "xyz"
        }
        worker_mode = mode_map.get(capture_mode, "xy")
        
        # Start capture in worker thread
        self.hardware.auto_capture_worker = AutoCaptureWorker(
            self.hardware,
            mode=worker_mode, 
            overlap=overlap_percent, 
            num_stack=num_z_stack,
            pattern_size_x=pattern_width_mm,
            pattern_size_y=pattern_height_mm,
            z_stack_height=z_height_mm
        )
            
        # Connect signals
        self.hardware.auto_capture_worker.progress_update.connect(self.update_progress)
        self.hardware.auto_capture_worker.finished_capture.connect(self.auto_capture_finished)
        
        # Start the worker
        self.hardware.auto_capture_worker.start()
        
    def auto_capture_finished(self):
        """Handle completion of auto-capture"""
        self.status_label.setText("Capture complete!")
        self.progress_bar.setValue(100)
        self.auto_capture_button.setEnabled(True)
        self.hardware.auto_capture_in_progress = False
        self.hardware.auto_capture_folder = None
        
        # Disable the stop button
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(False)
        
        # Make sure LED checkbox is unchecked since the LED is turned off in hardware
        if self.led_enable.isChecked():
            self.led_enable.setChecked(False)
    
    def save_current_photo_clicked(self):
        """Save current frame"""
        x = self.hardware.motion_controller._get_coordinate('X')
        y = self.hardware.motion_controller._get_coordinate('Y')
        z = self.hardware.motion_controller._get_coordinate('Z')
        self.hardware.save_last_frame(
            filename=f"capture_current_X{x:.3f}_Y{y:.3f}_Z{z:.3f}.tiff",
            foldername=os.path.join(os.path.expanduser("~"), "Downloads")
        )
    
    def zero_stage(self):
        """Move the Z axis to the absolute zero position for the current optic"""
        # Create and start the worker thread
        self.zero_worker = ZeroStageWorker(self.hardware, self)
        self.zero_worker.finished_move.connect(self._on_zero_complete)
        self.zero_worker.error_occurred.connect(self._on_zero_error)
        self.zero_worker.start()

    def _on_zero_complete(self):
        """Handle successful completion of zeroing"""
        print(f"Stage moved to focus position for {self.optic_combo.currentText()} optic")
        self.update_coordinates()

    def _on_zero_error(self, error_message):
        """Handle errors during zeroing"""
        print(error_message)
        QMessageBox.warning(self, "Zero Stage Error", error_message)

    def _update_led_ui_state(self):
        """Update the LED UI elements based on the current LED state"""
        if self.led_combo.currentText() == "None":
            self.led_enable.setEnabled(False)
            self.led_slider.setEnabled(False)
            self.led_intensity_input.setEnabled(False)
            self.led_enable.setChecked(False)
        else:
            self.led_enable.setEnabled(True)
            self.led_slider.setEnabled(True)
            self.led_intensity_input.setEnabled(True)
            # Update the slider based on the current intensity
            self.update_led_slider_from_intensity()

    def _update_roi_display(self):
        """Update the ROI display when pattern size values change"""
        if hasattr(self.parent(), 'grid_display'):
            self.parent().grid_display.update()


###############################################################################
# MainWindow
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Control UI")
        
        # Initialize overlap percentage for preview field
        self.overlap_percent = 0.0  # Default 20% overlap
        
        # Initialize hardware
        self.hardware = MicroscopeHardware()
        
        # Check for unhomed axes and show warning if needed
        self.check_axes_homing_status()
        
        # Start camera with error handling
        camera_started = self.hardware.start_camera()
        if not camera_started:
            self.statusBar().showMessage("Camera not available - limited functionality")
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create left panel with tabs
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Create tab widget for display
        self.tab_widget = QTabWidget()
        
        # Create camera display widget (for Live View tab)
        self.camera_display = CameraDisplayWidget(self.hardware)
        self.tab_widget.addTab(self.camera_display, "Live View")
        
        # Create grid display widget (for Preview Field tab)
        self.grid_display = GridDisplayWidget(self.hardware)
        self.tab_widget.addTab(self.grid_display, "Preview Field")
        
        # Add tab widget to left panel
        left_layout.addWidget(self.tab_widget)
        
        # Add left panel to main layout
        main_layout.addWidget(left_panel, stretch=2)
        
        # Create control panel
        self.control_panel = ControlPanelWidget(self.hardware)
        main_layout.addWidget(self.control_panel, stretch=1)
        
        # Add preview field button to control panel
        self.add_preview_field_button()
        
        # Update LED UI state to match hardware
        self._update_led_ui_state()
        
        # Start camera thread only if camera is available
        self.camera_thread = None
        if self.hardware.camera:
            try:
                self.camera_thread = CameraWorker(self.hardware.camera, self.hardware)
                self.camera_thread.new_frame.connect(self.on_new_frame)
                self.camera_thread.error_occurred.connect(self.on_camera_error)
                self.camera_thread.start()
            except Exception as e:
                self.statusBar().showMessage(f"Error starting camera thread: {e}")
                self.camera_thread = None
        else:
            self.statusBar().showMessage("No camera available")
        
        # Preview field worker (will be created when needed)
        self.preview_worker = None
    
    def check_axes_homing_status(self):
        """Check if all axes are homed and show warning if not"""
        if not hasattr(self.hardware, 'motion_controller') or not self.hardware.motion_controller:
            return
            
        unhomed_axes = []
        for axis in ["X", "Y", "Z"]:
            if axis not in self.hardware.motion_controller.homed_axes:
                unhomed_axes.append(axis)
        
        if unhomed_axes:
            warning_message = (
                f"Warning: The following axes are not homed: {', '.join(unhomed_axes)}\n\n"
                "This may cause inaccurate positioning during operation.\n\n"
                "Would you like to attempt to home these axes now?"
            )
            reply = QMessageBox.warning(
                self, "Unhomed Axes Detected", 
                warning_message,
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Try to home the axes
                QMessageBox.information(
                    self, "Homing Axes",
                    "The system will now attempt to home the unhomed axes.\n"
                    "This process may take a minute or more.\n\n"
                    "Click OK to begin."
                )
                
                self.statusBar().showMessage("Homing axes in progress...")
                result = self.hardware.ensure_axes_homed()
                
                if result:
                    QMessageBox.information(
                        self, "Homing Complete",
                        "All axes have been successfully homed."
                    )
                    self.statusBar().showMessage("All axes homed successfully")
                else:
                    remaining_unhomed = []
                    for axis in ["X", "Y", "Z"]:
                        if axis not in self.hardware.motion_controller.homed_axes:
                            remaining_unhomed.append(axis)
                    
                    if remaining_unhomed:
                        QMessageBox.warning(
                            self, "Homing Incomplete",
                            f"The following axes could not be homed: {', '.join(remaining_unhomed)}\n\n"
                            "Stage movement may be inaccurate.\n"
                            "You may need to restart the application."
                        )
                        self.statusBar().showMessage(f"Warning: {', '.join(remaining_unhomed)} not homed")
                    else:
                        self.statusBar().showMessage("All axes homed successfully")

    def add_preview_field_button(self):
        """Connect to the Preview Field and Stop buttons in the control panel"""
        # Connect the stop button in the control panel
        if hasattr(self.control_panel, 'stop_button'):
            self.stop_button = self.control_panel.stop_button
        
        # Connect the preview field button to our start_preview_field method
        if hasattr(self.control_panel, 'preview_field_button'):
            self.preview_button = self.control_panel.preview_field_button
            # Direct connection to our method
            self.preview_button.clicked.connect(self.start_preview_field)

    def start_preview_field(self):
        """Handler for Preview Field button click - start preview field scan"""
        if self.preview_worker and self.preview_worker.isRunning():
            QMessageBox.warning(self, "Preview In Progress", 
                              "A preview field scan is already running.")
            return
            
        # Confirm with user before proceeding
        reply = QMessageBox.question(
            self, "Preview Field Scan",
            f"This will scan the entire field with the current optic.\n"
            f"It will move to Z position {self.hardware.z_zero} mm "
            f"and take photos across the field.\n\n"
            f"Do you want to proceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Switch to the Preview Field tab
            self.tab_widget.setCurrentIndex(1)
            
            # Set up the grid display for preview
            self._setup_preview_grid()
            
            # Create and start the preview worker
            self.preview_worker = PreviewFieldWorker(self.hardware, overlap_percent=self.overlap_percent)
            self.preview_worker.progress_update.connect(self.update_preview_progress)
            self.preview_worker.new_preview_image.connect(self.add_preview_image)
            self.preview_worker.finished_preview.connect(self.preview_completed)
            
            # Start the preview
            self.preview_worker.start()
            
            # Enable the stop button after starting the worker
            self.stop_button.setEnabled(True)
            
            self.statusBar().showMessage("Preview field scan started...")

    def _setup_preview_grid(self):
        """Initialize the grid display for preview field"""
        fov_x = self.hardware.fov_x
        fov_y = self.hardware.fov_y
        print(f"Using FOV from hardware: {fov_x}mm x {fov_y}mm for {self.hardware.current_optic}")

        step_x = fov_x * (1 - self.overlap_percent/100.0)
        step_y = fov_y * (1 - self.overlap_percent/100.0)
        
        # Calculate number of positions needed
        count_x = int(np.ceil((self.hardware.max_slide_x-fov_x)/fov_x))
        count_y = int(np.ceil((self.hardware.max_slide_y-fov_y)/fov_y))
        
        # Calculate total pattern size
        total_width = step_x * (count_x - 1)
        total_height = step_y * (count_y - 1)
        
        # Calculate starting positions (centered on x,y)
        start_x = self.hardware.motion_stage_center_x - total_width/2 if self.hardware.motion_stage_center_x - total_width/2 > self.hardware.motion_stage_min_x else self.hardware.motion_stage_min_x
        start_y = self.hardware.motion_stage_center_y - total_height/2 if self.hardware.motion_stage_center_y - total_height/2 > self.hardware.motion_stage_min_y else self.hardware.motion_stage_min_y
        
        # Generate position lists
        positions_x = [start_x + i * step_x for i in range(count_x)]
        positions_y = [start_y + i * step_y for i in range(count_y)]
        
        # Grid dimensions should match the positions array length
        grid_width = len(positions_x)
        grid_height = len(positions_y)
        
        self.grid_display.grid_min_vis_x = positions_x[0]-fov_x/2
        self.grid_display.grid_max_vis_x = positions_x[-1]+fov_x/2
        self.grid_display.grid_min_vis_y = positions_y[0]-fov_y/2
        self.grid_display.grid_max_vis_y = positions_y[-1]+fov_y/2

        # Initialize grid with exact same dimensions as will be used for capture
        self.grid_display.setup_grid(grid_width, grid_height)
        self.grid_display.clear_grid()
        
        # Update FOV size in grid display for correct green box
        self.grid_display.current_fov_width = fov_x
        self.grid_display.current_fov_height = fov_y
        
        # Connect grid click to move stage
        if not hasattr(self, '_grid_connections_setup'):
            self.grid_display.move_to_position.connect(self.move_to_grid_position)
            self._grid_connections_setup = True
        
        print(f"Set up preview grid: {grid_width}x{grid_height} ({positions_x[-1]}mm x {positions_y[-1]}mm)")

    def move_to_grid_position(self, x_mm, y_mm):
        """Move the stage to the selected position on the grid"""
        if not hasattr(self.hardware, 'motion_controller') or not self.hardware.motion_controller:
            self.statusBar().showMessage("Motion controller not available")
            return
        
        if x_mm<self.hardware.motion_stage_min_x:
            x_mm = self.hardware.motion_stage_min_x
        elif x_mm>self.hardware.motion_stage_max_x:
            x_mm = self.hardware.motion_stage_max_x
        if y_mm<self.hardware.motion_stage_min_y:
            y_mm = self.hardware.motion_stage_min_y
        elif y_mm>self.hardware.motion_stage_max_y:
            y_mm = self.hardware.motion_stage_max_y

        try:
            # Use the current Z position
            self.hardware.move_to_absolute(x=x_mm, y=y_mm, wait_for_completion=False)
            self.statusBar().showMessage(f"Moving to X={x_mm:.1f}mm, Y={y_mm:.1f}mm")
            
            # Also switch to Live View tab to see the camera feed
            #self.tab_widget.setCurrentIndex(0)
        except Exception as e:
            self.statusBar().showMessage(f"Error moving stage: {e}")
            QMessageBox.warning(
                self, 
                "Motion Error", 
                f"Could not move to the selected position: {str(e)}"
            )

    def update_preview_progress(self, completed, total, remaining):
        """Update the progress bar during preview field scan"""
        progress = int(100 * completed / total) if total > 0 else 0
        remaining_time = remaining*7.5
        # Update status bar
        if remaining_time > 60:
            mins = int(remaining_time / 60)
            secs = int(remaining_time % 60)
            self.statusBar().showMessage(f"Preview scan: {completed}/{total} positions ({mins}m {secs}s remaining)")
        else:
            self.statusBar().showMessage(f"Preview scan: {completed}/{total} positions ({int(remaining_time)}s remaining)")
        
        # Update the progress bar in the control panel
        if hasattr(self.control_panel, 'progress_bar'):
            self.control_panel.progress_bar.setValue(progress)
    
    def add_preview_image(self, image, grid_x, grid_y):
        """Add a new image to the preview grid"""
        # Ensure coordinates are within grid bounds
        if not self.grid_display:
            return
            
        if grid_x < 0 or grid_y < 0 or grid_x >= self.grid_display.grid_width or grid_y >= self.grid_display.grid_height:
            print(f"Warning: Image coordinates ({grid_x},{grid_y}) outside grid dimensions ({self.grid_display.grid_width}x{self.grid_display.grid_height})")
            # Try to expand grid if needed
            if grid_x >= self.grid_display.grid_width or grid_y >= self.grid_display.grid_height:
                new_width = max(self.grid_display.grid_width, grid_x + 1)
                new_height = max(self.grid_display.grid_height, grid_y + 1)
                print(f"Expanding grid to {new_width}x{new_height}")
                self.grid_display.setup_grid(new_width, new_height)
        
        # Add the image to the grid (already flipped in the worker class)
        self.grid_display.add_image(image, grid_x, grid_y)
    
    def preview_completed(self, mosaic):
        """Handle completion of the preview field scan"""
        # Set the mosaic in the grid display
        self.grid_display.set_mosaic(mosaic)
        
        # Update UI
        self.preview_button.setEnabled(True)
        self.statusBar().showMessage("Preview field scan completed")

    
    def toggle_mosaic_view(self):
        """Toggle between grid view and mosaic view in the preview display"""
        self.grid_display.toggle_mosaic_view()

    def _update_led_ui_state(self):
        """Update the LED UI elements to match hardware state"""
        # Set the correct LED type in the combo box
        if hasattr(self.hardware, 'current_led_type') and self.hardware.current_led_type:
            index = self.control_panel.led_combo.findText(self.hardware.current_led_type)
            if index >= 0:
                self.control_panel.led_combo.setCurrentIndex(index)
                self.control_panel.led_enable.setEnabled(True)
        
        # Make sure target brightness is set correctly
        if hasattr(self.control_panel, 'target_slider'):
            self.control_panel.target_slider.setValue(self.hardware.autoexposure_target)
        
        if hasattr(self.control_panel, 'target_label'):
            self.control_panel.target_label.setText(f"{self.hardware.autoexposure_target}%")

    @pyqtSlot(np.ndarray)
    def on_new_frame(self, frame):
        """Update UI with new frame"""
        self.camera_display.update_frame(frame)
        
        if self.tab_widget.currentIndex() == 1:  # Preview Field tab
            # Update current position indicator on grid display
            if hasattr(self.hardware, 'motion_controller') and self.hardware.motion_controller:
                try:
                    # Get current position
                    x_mm = self.hardware.motion_controller._get_coordinate("X")
                    y_mm = self.hardware.motion_controller._get_coordinate("Y")
                    optic = self.hardware.current_optic
                    
                    # Make sure grid display has current FOV values
                    self.grid_display.current_fov_width = self.hardware.fov_x
                    self.grid_display.current_fov_height = self.hardware.fov_y
                    
                    # Update position on grid display
                    self.grid_display.update_current_position(x_mm, y_mm, optic)
                except Exception as e:
                    print(f"Error updating current position: {e}")

    def closeEvent(self, event):
        # Stop any active preview
        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker.wait()
        
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.hardware.stop_camera()
        self.hardware.close_hardware()
        super().closeEvent(event)

    # Add this method to handle camera errors
    @pyqtSlot(str)
    def on_camera_error(self, error_message):
        """Handle camera errors from the worker thread"""
        self.statusBar().showMessage(f"Camera error: {error_message}")
        
        # Show a more detailed error dialog
        QMessageBox.warning(
            self, 
            "Camera Error", 
            f"{error_message}\n\nThe application will continue running, but camera functionality may be limited."
        )

    def start_auto_capture(self):
        """Start auto-capture - just forward to control panel's auto_capture_clicked method"""
        if hasattr(self.control_panel, 'auto_capture_clicked'):
            self.control_panel.auto_capture_clicked()
        
    def stop_capture_processes(self):
        """Stop any active capture processes"""
        # Stop preview worker if running
        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.running = False
            self.preview_worker.wait(1000)  # Wait up to 1 second
            self.statusBar().showMessage("Preview scan stopped")
            if hasattr(self.control_panel, 'status_label'):
                self.control_panel.status_label.setText("Preview cancelled")
            if hasattr(self.control_panel, 'progress_bar'):
                self.control_panel.progress_bar.setValue(0)
        
        # Stop auto-capture worker if running
        if hasattr(self.hardware, 'auto_capture_worker') and self.hardware.auto_capture_worker and self.hardware.auto_capture_worker.isRunning():
            self.hardware.auto_capture_worker.terminate()
            self.hardware.auto_capture_worker.wait(1000)  # Wait up to 1 second
            self.hardware.auto_capture_in_progress = False
            self.statusBar().showMessage("Auto-capture stopped")
            if hasattr(self.control_panel, 'status_label'):
                self.control_panel.status_label.setText("Capture cancelled")
            if hasattr(self.control_panel, 'progress_bar'):
                self.control_panel.progress_bar.setValue(0)
            if hasattr(self.control_panel, 'auto_capture_button'):
                self.control_panel.auto_capture_button.setEnabled(True)
        
        # Disable stop button
        if hasattr(self.control_panel, 'stop_button'):
            self.control_panel.stop_button.setEnabled(False)
        
        # Make sure camera is running
        self.hardware.start_camera()


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

