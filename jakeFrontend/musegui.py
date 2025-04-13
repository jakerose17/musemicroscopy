#!/usr/bin/env python3
"""
MUSE Microscopy GUI
A modular microscope control application for the MUSE project.
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
import traceback
from enum import Enum
from ctypes import cdll, c_uint32, c_int, c_double, c_char_p, byref, create_string_buffer

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QMutex, QMutexLocker, QTimer, QObject
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout, 
    QCheckBox, QLineEdit, QFileDialog, QMessageBox, QComboBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator, QPainter, QFont

# Configure paths for ThorLabs SDK
def configure_thorlabs_paths():
    """Configure paths for ThorLabs SDK DLLs"""
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
        print(f"Added ThorLabs DLL path: {absolute_path_to_dlls}")
    else:
        print(f"WARNING: ThorLabs DLL path does not exist: {absolute_path_to_dlls}")

# Configure SDK paths before importing
configure_thorlabs_paths()

# Try to import ThorLabs SDK components
try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
    from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
    from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK
    print("Successfully imported ThorLabs SDK for color processing")
    THORLABS_SDK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ThorLabs SDK import error: {e}")
    print("Color processing functionality will be limited")
    THORLABS_SDK_AVAILABLE = False

# Import Kinesis (for stage control)
import clr
kinesis_path = r"C:\Program Files\Thorlabs\Kinesis"
try:
    clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll")
    clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericMotorCLI.dll")
    clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.DCServoCLI.dll")
    from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI, DeviceConfiguration, SimulationManager
    from Thorlabs.MotionControl.GenericMotorCLI import *
    from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
    from System import Decimal, Int32
    KINESIS_AVAILABLE = True
    print("Successfully imported Kinesis SDK for motion control")
except Exception as e:
    print(f"Warning: Kinesis import error: {e}")
    print("Motion control functionality will be limited")
    KINESIS_AVAILABLE = False

#######################################################################
# Hardware Classes
#######################################################################

class CameraWorker(QThread):
    """Worker thread for camera acquisition"""
    
    # Signals
    new_frame = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.running = False
        
    def run(self):
        """Main acquisition loop"""
        self.running = True
        
        while self.running:
            try:
                # Get frame from camera using ThorLabs SDK's method
                frame = None
                
                # Trigger the camera to capture a frame
                self.camera.issue_software_trigger()
                
                # Try to get the pending frame
                camera_frame = self.camera.get_pending_frame_or_null()
                if camera_frame is not None:
                    # Extract the image buffer from the frame
                    frame = camera_frame.image_buffer
                    
                    # Emit signal with new frame
                    if frame is not None:
                        self.new_frame.emit(frame)
                
            except Exception as e:
                error_message = f"Camera acquisition error: {str(e)}"
                print(error_message)
                self.error_occurred.emit(error_message)
                
                # Sleep briefly to avoid hammering the system in case of error
                time.sleep(0.1)
                
            # Slight delay to avoid hogging the CPU
            time.sleep(0.01)
            
    def stop(self):
        """Stop the acquisition loop"""
        self.running = False
        self.wait()  # Wait for thread to finish

class MicroscopeCamera:
    """Manages camera initialization, acquisition, and frame processing"""
    
    def __init__(self):
        self.camera = None
        self.camera_sdk = None
        self.mono_to_color_sdk = None
        self.mono_to_color_processor = None
        self.is_color = False
        self.camera_model = "Unknown"
        self.bit_depth = 12
        self.camera_lock = QMutex()
        self.processing_lock = QMutex()  # Add mutex for frame processing
        self.sensor_width = 2448
        self.sensor_height = 2048
        self.last_frame = None
        self.last_frame_time = 0
        self.current_exposure = 0.1  # Default 100ms
        
        # Try to initialize camera
        self._initialize_camera()
        
    def _initialize_camera(self):
        """Initialize camera with ThorLabs SDK if available"""
        if not THORLABS_SDK_AVAILABLE:
            print("ThorLabs SDK not available, camera initialization skipped")
            return
            
        try:
            # Initialize camera SDK and discover cameras
            self.camera_sdk = TLCameraSDK()
            camera_list = self.camera_sdk.discover_available_cameras()
            
            if not camera_list:
                print("No cameras detected")
                return
                
            # Open the first camera
            self.camera = self.camera_sdk.open_camera(camera_list[0])
            print(f"Connected to camera: {self.camera.name}")
            
            # Configure camera
            self.camera.frames_per_trigger_zero_for_unlimited = 0  # Continuous mode
            self.camera.exposure_time_us = 100000  # 100ms exposure
            
            # Get camera info
            self.camera_model = self.camera.name
            self.bit_depth = self.camera.bit_depth
            self.sensor_width = self.camera.image_width_pixels
            self.sensor_height = self.camera.image_height_pixels
            
            # Set up color processing if camera has Bayer pattern
            if self.camera.camera_sensor_type == SENSOR_TYPE.BAYER:
                self.mono_to_color_sdk = MonoToColorProcessorSDK()
                self.mono_to_color_processor = self.mono_to_color_sdk.create_mono_to_color_processor(
                    SENSOR_TYPE.BAYER,
                    self.camera.color_filter_array_phase,
                    self.camera.get_color_correction_matrix(),
                    self.camera.get_default_white_balance_matrix(),
                    self.camera.bit_depth
                )
                self.is_color = True
                print(f"Camera is color (Bayer pattern)")
            else:
                self.is_color = False
                print(f"Camera is monochrome")
                
            # Arm the camera to start capturing
            self.camera.arm(2)
            self.camera.issue_software_trigger()
            
            print(f"Camera initialized: {self.camera_model}, {self.sensor_width}x{self.sensor_height}, {self.bit_depth}-bit")
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.camera = None
            
    def start_acquisition(self):
        """Start camera acquisition"""
        if self.camera:
            # Camera is already armed in initialization
            print("Camera acquisition started")
            return True
        else:
            print("No camera available")
            return False
            
    def stop_acquisition(self):
        """Stop camera acquisition"""
        if self.camera:
            try:
                if hasattr(self.camera, 'disarm'):
                    self.camera.disarm()
                print("Camera acquisition stopped")
            except Exception as e:
                print(f"Error stopping camera: {e}")
                
    def close(self):
        """Clean up camera resources"""
        if self.camera:
            try:
                self.camera.disarm()
                self.camera.dispose()
                if self.mono_to_color_processor:
                    self.mono_to_color_processor.dispose()
                if self.mono_to_color_sdk:
                    self.mono_to_color_sdk.dispose()
                if self.camera_sdk:
                    self.camera_sdk.dispose()
                print("Camera closed")
            except Exception as e:
                print(f"Error closing camera: {e}")
        self.camera = None
        
    def set_exposure(self, exposure_sec):
        """Set camera exposure time in seconds"""
        with QMutexLocker(self.camera_lock):
            if self.camera:
                try:
                    self.current_exposure = float(exposure_sec)
                    # Convert to microseconds for the camera
                    self.camera.exposure_time_us = int(self.current_exposure * 1000000)
                    print(f"Camera exposure set to {self.current_exposure} s")
                except Exception as e:
                    print(f"Error setting exposure: {e}")
                
    def process_frame(self, frame):
        """Process a frame from the camera using ThorLabs SDK processing"""
        with QMutexLocker(self.processing_lock):
            if frame is None or frame.size == 0:
                return np.zeros((100, 100, 3), dtype=np.uint8)
                
            # Store original frame for saving
            self.last_frame = frame.copy()
            self.last_frame_time = time.time()
            
            try:
                # For color camera with ThorLabs SDK
                if self.mono_to_color_processor and self.is_color:
                    width = frame.shape[1]
                    height = frame.shape[0]
                    
                    # Use ThorLabs color processor
                    try:
                        color_image_data = self.mono_to_color_processor.transform_to_24(
                            frame, width, height
                        )
                        # Reshape to proper dimensions
                        color_image_data = color_image_data.reshape(height, width, 3)
                        return color_image_data
                    except Exception as e:
                        print(f"Could not transform image to 24bpp; {e}")
                        # Fallback to grayscale conversion
                        # Scale to 8-bit
                        scaled_image = frame >> (self.bit_depth - 8)
                        # Convert to RGB
                        rgb_image = cv2.cvtColor(scaled_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                        return rgb_image
                    
                # For monochrome camera with ThorLabs SDK
                elif self.camera_sdk and not self.is_color:
                    # Scale to 8-bit
                    scaled_image = frame >> (self.bit_depth - 8)
                    # Convert to RGB for display consistency
                    rgb_image = cv2.cvtColor(scaled_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    return rgb_image
                    
                # Fallback normalization for other cameras
                else:
                    # Simple min-max normalization
                    min_val = frame.min()
                    max_val = frame.max()
                    
                    if max_val > min_val:
                        scaled = (frame - min_val) / (max_val - min_val) * 255.0
                        normalized = scaled.astype(np.uint8)
                    else:
                        normalized = np.full_like(frame, 127, dtype=np.uint8)
                        
                    # Convert to RGB if grayscale
                    if len(normalized.shape) == 2:
                        normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
                        
                    return normalized
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Return gray image in case of error
                return np.full((480, 640, 3), 128, dtype=np.uint8)
                
    def capture_image(self, filename=None):
        """Capture a single image and save it"""
        with QMutexLocker(self.camera_lock):
            if not self.camera:
                print("No camera available for capture")
                return None
                
            # Try to grab a frame
            frame = None
            max_attempts = 3
            
            for attempt in range(max_attempts):
                try:
                    # Trigger and get frame
                    self.camera.issue_software_trigger()
                    timeout_ms = 1000
                    start_time = time.time()
                    
                    while time.time() - start_time < timeout_ms/1000:
                        camera_frame = self.camera.get_pending_frame_or_null()
                        if camera_frame is not None:
                            frame = camera_frame.image_buffer.copy()
                            self.last_frame = frame.copy()
                            break
                        time.sleep(0.01)
                        
                    if frame is not None:
                        break  # Success
                except Exception as e:
                    print(f"Error capturing frame (attempt {attempt+1}/{max_attempts}): {e}")
                time.sleep(0.5)  # Wait between attempts
                
            if frame is None:
                print("Failed to capture image")
                return None
                
            # Process the frame (outside the lock to avoid deadlock)
        
        # Process frame with processing lock
        with QMutexLocker(self.processing_lock):
            processed_frame = self.process_fallback(frame)
            
            # Save the image if filename provided
            if filename:
                self._save_image(frame, filename, processed_frame)
                
            return processed_frame
    
    def process_fallback(self, frame):
        """Fallback processing when color processor fails"""
        if frame is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        try:
            # Scale to 8-bit
            if frame.max() > 255:
                bit_shift = max(0, int(np.log2(frame.max())) - 7)
                scaled_image = frame >> bit_shift
            else:
                scaled_image = frame
                
            # Ensure uint8
            if scaled_image.dtype != np.uint8:
                if scaled_image.max() > 0:
                    normalized = (scaled_image.astype(float) / scaled_image.max() * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(scaled_image, dtype=np.uint8)
            else:
                normalized = scaled_image
                
            # Convert to RGB if grayscale
            if len(normalized.shape) == 2:
                rgb_image = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
                return rgb_image
            else:
                return normalized
                
        except Exception as e:
            print(f"Error in fallback processing: {e}")
            return np.full((480, 640, 3), 128, dtype=np.uint8)
        
    def save_last_frame(self):
        """Save the most recently captured frame in full color"""
        with QMutexLocker(self.processing_lock):
            if self.last_frame is None:
                print("No frame available to save")
                return None
                
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"microscope_image_{timestamp}.tiff"  # Use TIFF for 12-bit data
            
            # Save to downloads folder with full bit depth
            return self._save_full_color_image(self.last_frame, filename)
    
    def _save_full_color_image(self, frame, filename):
        """Save image with full bit depth"""
        try:
            # Determine downloads path
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            
            # Ensure filename has extension
            base, ext = os.path.splitext(filename)
            if not ext:
                filename = base + ".tiff"  # Default to TIFF for high bit depth
                
            save_path = os.path.join(downloads_dir, filename)
            
            # For high bit depth, always save as TIFF with original data
            if self.is_color and self.mono_to_color_processor:
                # Try to use the SDK color processor for best quality
                try:
                    width = frame.shape[1]
                    height = frame.shape[0]
                    
                    # Use 48-bit color processing for full bit depth
                    color_data = self.mono_to_color_processor.transform_to_48(
                        frame, width, height
                    )
                    # Reshape to proper dimensions for a 48-bit color image (16 bits per channel)
                    color_data = color_data.reshape(height, width, 3)
                    
                    # Save as 16-bit TIFF
                    cv2.imwrite(save_path, color_data)
                    print(f"Saved full 16-bit color image to {save_path}")
                    
                except Exception as e:
                    print(f"Error in color transform, using fallback method: {e}")
                    # Save original frame if color transform fails
                    if self.bit_depth > 8:
                        # For higher bit depths (e.g., 12-bit), save as 16-bit TIFF
                        frame_16bit = frame.astype(np.uint16)
                        cv2.imwrite(save_path, frame_16bit)
                        print(f"Saved original {self.bit_depth}-bit data to {save_path}")
                    else:
                        cv2.imwrite(save_path, frame)
            else:
                # For monochrome, save original data
                if self.bit_depth > 8:
                    # Ensure we use 16-bit for TIFF
                    frame_16bit = frame.astype(np.uint16)
                    cv2.imwrite(save_path, frame_16bit)
                else:
                    cv2.imwrite(save_path, frame)
                    
            return save_path
            
        except Exception as e:
            print(f"Error saving full color image: {e}")
            traceback.print_exc()
            return None

class MotionController:
    """Controls the 3-axis motorized stage"""
    
    def __init__(self):
        self.devices = {}
        self.homed_axes = set()
        self.axis_info = {
            "X": "27601188",
            "Y": "27601181",
            "Z": "27265400",
        }
        
        if KINESIS_AVAILABLE:
            self._initialize_devices()
        else:
            print("Kinesis SDK not available, motion control disabled")
            
    def _initialize_devices(self):
        """Initialize connection to motion control devices"""
        try:
            DeviceManagerCLI.BuildDeviceList()
            available_devices = list(DeviceManagerCLI.GetDeviceList())
            print(f"Kinesis devices found: {available_devices}")
            
            for axis, serial_no in self.axis_info.items():
                try:
                    device = KCubeDCServo.CreateKCubeDCServo(serial_no)
                    device.Connect(serial_no)
                    device.StartPolling(250)
                    device.EnableDevice()
                    
                    # Wait for settings initialization
                    if not device.IsSettingsInitialized():
                        device.WaitForSettingsInitialized(10000)
                    if not device.IsSettingsInitialized():
                        print(f"[{axis}] Settings failed to initialize!")
                        continue
                        
                    # Load motor configuration
                    m_config = device.LoadMotorConfiguration(serial_no,
                        DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)
                    m_config.UpdateCurrentConfiguration()
                    device.SetSettings(device.MotorDeviceSettings, True, False)
                    
                    # Check if already homed
                    if device.Status.IsHomed:
                        print(f"[{axis}] Already homed. Current pos: {device.Position}")
                        self.homed_axes.add(axis)
                    elif device.Position == Decimal(0):
                        print(f"Homing axis {axis}...")
                        device.Home(120000)  # 2 minute timeout
                        print(f"[{axis}] Homing complete.")
                        self.homed_axes.add(axis)
                    else:
                        # Not at home but has position
                        print(f"[{axis}] Not at home position but at {device.Position}.")
                        self.homed_axes.add(axis)
                        
                    # Store the device
                    self.devices[axis] = device
                    
                except Exception as e:
                    print(f"[{axis}] Error initializing: {e}")
                    
        except Exception as e:
            print(f"Error initializing motion controller: {e}")
            
    def home_all_axes(self):
        """Home all axes that aren't already homed"""
        for axis in self.axis_info.keys():
            if axis not in self.homed_axes and axis in self.devices:
                try:
                    device = self.devices[axis]
                    print(f"Homing {axis} axis...")
                    device.Home(120000)  # 2 minute timeout
                    self.homed_axes.add(axis)
                    print(f"[{axis}] Homing complete.")
                except Exception as e:
                    print(f"[{axis}] Error homing: {e}")
                    
    def move_relative(self, axis, distance_mm):
        """Move an axis by a relative amount in mm"""
        if axis not in self.devices:
            print(f"[{axis}] Axis not available.")
            return False
            
        try:
            device = self.devices[axis]
            current_pos = float(str(device.Position))
            new_pos = current_pos + distance_mm
            device.MoveTo(Decimal(new_pos), Int32(20000))  # 20 second timeout
            return True
        except Exception as e:
            print(f"[{axis}] Error moving: {e}")
            return False
            
    def move_absolute(self, axis, position_mm):
        """Move an axis to an absolute position in mm"""
        if axis not in self.devices:
            print(f"[{axis}] Axis not available.")
            return False
            
        try:
            device = self.devices[axis]
            device.MoveTo(Decimal(position_mm), Int32(20000))  # 20 second timeout
            return True
        except Exception as e:
            print(f"[{axis}] Error moving to absolute position: {e}")
            return False
            
    def get_position(self, axis):
        """Get the current position of an axis in mm"""
        if axis not in self.devices:
            print(f"[{axis}] Axis not available.")
            return 0.0
            
        try:
            device = self.devices[axis]
            position = float(str(device.Position))
            return position
        except Exception as e:
            print(f"[{axis}] Error getting position: {e}")
            return 0.0
            
    def disconnect(self):
        """Disconnect from all devices"""
        for axis, device in self.devices.items():
            try:
                print(f"Disconnecting {axis} axis...")
                device.StopPolling()
                device.DisableDevice()
                device.Disconnect()
            except Exception as e:
                print(f"[{axis}] Error disconnecting: {e}")
                
        # Clean up simulation manager
        SimulationManager.Instance.UninitializeSimulations()

class LEDController:
    """Controls LEDs for microscope illumination"""
    
    def __init__(self):
        self.led_lib = None
        self.led_handles = {}
        self.led_max_currents = {}
        self.current_led_type = None
        self.led_controllers = {}
        self.enabled = False
        self.current_intensity = 50  # Default 50%
        
        # Try to initialize LEDs
        self._initialize_led_library()
        
    def _initialize_led_library(self):
        """Initialize the LED control library"""
        try:
            # Try to load the DLL
            try:
                self.led_lib = cdll.LoadLibrary("thorlabs_unified_powermeter_library.dll")
                
                # Initialize the library
                ret = self.led_lib.TLUP_init()
                if ret == 0:
                    print("LED library loaded successfully.")
                    self._detect_led_controllers()
                else:
                    print(f"Error initializing LED library: {ret}")
                    self.led_lib = None
            except:
                # Fallback to mock LED implementation if library not available
                print("LED library not available, using mock implementation")
                self.led_controllers = {
                    "White": {"handle": None, "max_intensity": 100},
                    "UV": {"handle": None, "max_intensity": 100}
                }
                print("LED controllers initialized")
                
        except Exception as e:
            print(f"Error initializing LED controllers: {e}")
            # Create fallback LEDs
            self.led_controllers = {
                "White": {"handle": None, "max_intensity": 100},
                "UV": {"handle": None, "max_intensity": 100}
            }
            
    def _detect_led_controllers(self):
        """Auto-detect connected LED controllers"""
        if not self.led_lib:
            return
            
        try:
            print("Auto-detecting LED controllers on all COM ports...")
            
            # Check COM ports 1-19
            for port in range(1, 20):
                resource_name = f"ASRL{port}::INSTR"
                print(f"Checking COM{port} ({resource_name})...")
                
                # Try to open the port
                handle = c_uint32(0)
                ret = self.led_lib.TLUP_openWithResource(resource_name.encode(), byref(handle))
                
                if ret == 0:
                    print(f"Found LED controller on COM{port}")
                    
                    # Get details about the LED
                    model_buffer = create_string_buffer(32)
                    serial_buffer = create_string_buffer(32)
                    wavelength = c_double(0)
                    max_current = c_double(0)
                    
                    self.led_lib.TLUP_getLedModuleModel(handle, model_buffer, 32)
                    self.led_lib.TLUP_getLedModuleSerialNumber(handle, serial_buffer, 32)
                    self.led_lib.TLUP_getLedWavelength(handle, byref(wavelength))
                    self.led_lib.TLUP_getLedMaximumCurrent(handle, byref(max_current))
                    
                    model = model_buffer.value.decode()
                    serial = serial_buffer.value.decode()
                    wave_nm = wavelength.value
                    max_curr = max_current.value
                    
                    # Determine LED type based on wavelength and model
                    led_type = "Unknown"
                    if wave_nm < 0 or "White" in model:
                        led_type = "White"
                    elif 250 <= wave_nm <= 300:
                        led_type = "UV"
                    
                    print(f"  Detected {led_type} on COM{port}")
                    print(f'  Model: "{model}"')
                    print(f'  Serial: "{serial}"')
                    print(f"  Wavelength: {wave_nm}")
                    print(f"  Max Current: {max_curr}")
                    
                    # Store the LED info
                    self.led_controllers[led_type] = {
                        "com_port": f"COM{port}",
                        "resource_name": resource_name,
                        "model": model,
                        "serial": serial,
                        "wavelength": wave_nm,
                        "max_current": max_curr
                    }
                    
                    # Close for now - will reconnect later when needed
                    self.led_lib.TLUP_close(handle)
                    
            # Print summary
            if self.led_controllers:
                print(f"Found {len(self.led_controllers)} LED controller(s):")
                for led_type, info in self.led_controllers.items():
                    print(f'  {led_type}: {info.get("com_port", "Mock")}')
            else:
                print("No LED controllers found, using mock implementation")
                self.led_controllers = {
                    "White": {"handle": None, "max_intensity": 100},
                    "UV": {"handle": None, "max_intensity": 100}
                }
                
        except Exception as e:
            print(f"Error detecting LED controllers: {e}")
            # Create fallback LEDs
            self.led_controllers = {
                "White": {"handle": None, "max_intensity": 100},
                "UV": {"handle": None, "max_intensity": 100}
            }
            
    def connect_led(self, led_type):
        """Connect to specific LED type"""
        if led_type not in self.led_controllers:
            print(f"Unknown LED type: {led_type}")
            return False
            
        # If we're already connected to this LED, do nothing
        if self.current_led_type == led_type:
            return True
            
        # Disconnect any current LED
        if self.current_led_type:
            self.turn_off_led()
            
        # Connect to the new LED
        try:
            if self.led_lib:
                info = self.led_controllers[led_type]
                if "resource_name" in info:
                    resource_name = info["resource_name"]
                    handle = c_uint32(0)
                    
                    ret = self.led_lib.TLUP_openWithResource(resource_name.encode(), byref(handle))
                    if ret == 0:
                        self.led_handles[led_type] = handle
                        self.led_max_currents[led_type] = info.get("max_current", 100)
                    else:
                        print(f"Error connecting to {led_type}: {ret}")
                        
            # Set as current LED regardless of hardware connection success
            self.current_led_type = led_type
            self.enabled = False
            self.current_intensity = 50  # Reset to default
            
            print(f"Connected to {led_type} LED")
            return True
                
        except Exception as e:
            print(f"Error connecting to {led_type}: {e}")
            # Set as current LED anyway for mock implementation
            self.current_led_type = led_type
            self.enabled = False
            return True
        
    def turn_on_led(self):
        """Turn on the current LED"""
        if not self.current_led_type:
            print("No LED selected")
            return False
            
        try:
            if self.led_lib and self.current_led_type in self.led_handles:
                # Use hardware control
                handle = self.led_handles[self.current_led_type]
                ret = self.led_lib.TLUP_switchLedOutput(handle, 1)
                if ret != 0:
                    print(f"Error turning on LED: {ret}")
                    return False
            
            # For both hardware and mock implementations
            print(f"Turning on {self.current_led_type} LED")
            self.enabled = True
            
            # Apply current intensity
            if self.current_intensity > 0:
                self.set_intensity(self.current_intensity)
                
            return True
        except Exception as e:
            print(f"Error turning on LED: {e}")
            # For mock implementation, still consider it successful
            self.enabled = True
            return True
            
    def turn_off_led(self):
        """Turn off the current LED"""
        if not self.current_led_type:
            return True  # No LED to turn off
            
        try:
            if self.led_lib and self.current_led_type in self.led_handles:
                # Use hardware control
                handle = self.led_handles[self.current_led_type]
                ret = self.led_lib.TLUP_switchLedOutput(handle, 0)
                if ret != 0:
                    print(f"Error turning off LED: {ret}")
                    return False
                    
            # For both hardware and mock implementations
            print(f"Turning off {self.current_led_type} LED")
            self.enabled = False
            return True
        except Exception as e:
            print(f"Error turning off LED: {e}")
            # For mock implementation, still consider it successful
            self.enabled = False
            return True
            
    def set_intensity(self, intensity_percent):
        """Set LED intensity as percentage of maximum"""
        if not self.current_led_type:
            print("No LED selected")
            return False
            
        if not 0 <= intensity_percent <= 100:
            print(f"Invalid intensity: {intensity_percent}")
            return False
            
        try:
            # Store the intensity value
            self.current_intensity = intensity_percent
            
            # Only apply if LED is enabled
            if self.enabled:
                if self.led_lib and self.current_led_type in self.led_handles and self.current_led_type in self.led_max_currents:
                    # Use hardware control
                    handle = self.led_handles[self.current_led_type]
                    max_current = self.led_max_currents[self.current_led_type]
                    current = (intensity_percent / 100.0) * max_current
                    
                    ret = self.led_lib.TLUP_setLedCurrentSetpoint(handle, c_double(current))
                    if ret != 0:
                        print(f"Error setting LED intensity: {ret}")
                        return False
                
                # For both hardware and mock implementations
                print(f"Setting {self.current_led_type} LED intensity to {intensity_percent}%")
                
            return True
        except Exception as e:
            print(f"Error setting LED intensity: {e}")
            return True  # For mock implementation
            
    def close(self):
        """Close all LED connections"""
        try:
            # Turn off any active LED
            self.turn_off_led()
            
            # Close all handles
            if self.led_lib:
                for led_type, handle in list(self.led_handles.items()):
                    try:
                        print(f"Closing {led_type} controller...")
                        self.led_lib.TLUP_close(handle)
                    except Exception as e:
                        print(f"Error closing {led_type} controller: {e}")
            
            # Clear all data
            self.led_handles = {}
            self.led_max_currents = {}
            self.led_controllers = {}
            self.current_led_type = None
            self.enabled = False
            
            print("LED controllers closed")
            return True
        except Exception as e:
            print(f"Error closing LED controllers: {e}")
            return False

#######################################################################
# UI Components
#######################################################################

class CameraDisplayWidget(QLabel):
    """Widget to display the live camera feed"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setFrameShape(QLabel.Box)
        
        # FPS tracking
        self.frame_count = 0
        self.last_update_time = time.time()
        self.fps = 0
        self.show_fps = True
        
        # Start timer for FPS updates
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update once per second
        
    def update_frame(self, frame):
        """Update the displayed frame"""
        if frame is None or not isinstance(frame, np.ndarray):
            return
            
        # Convert frame to QImage
        if len(frame.shape) == 3:
            # RGB image
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            # Using RGB888 format - will convert BGR to RGB if needed
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            # Grayscale image
            h, w = frame.shape
            bytes_per_line = w
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            
        # Convert to pixmap and display, maintaining aspect ratio
        pixmap = QPixmap.fromImage(img)
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Update frame count for FPS calculation
        self.frame_count += 1
        
    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if elapsed > 0 and self.frame_count > 0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_update_time = current_time
            self.update()  # Trigger a repaint to update the FPS display
            
    def paintEvent(self, event):
        """Override paint event to add FPS display"""
        super().paintEvent(event)
        
        # Draw FPS counter if enabled
        if self.show_fps and hasattr(self, 'fps'):
            try:
                painter = QPainter(self)
                painter.setPen(Qt.white)
                painter.setFont(QFont("Arial", 10, QFont.Bold))
                painter.drawText(10, 20, f"FPS: {self.fps:.1f}")
                painter.end()
            except Exception as e:
                print(f"Error drawing FPS: {e}")
                
    def toggle_fps_display(self):
        """Toggle FPS display on/off"""
        self.show_fps = not self.show_fps
        self.update()

class MotionControlPanel(QWidget):
    """Panel for motion control with buttons for X, Y, Z axes"""
    
    def __init__(self, motion_controller, parent=None):
        super().__init__(parent)
        self.motion_controller = motion_controller
        self.initUI()
        
        # Initialize move worker
        self.move_worker = None
        
        # Update timer for position display
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_position_display)
        self.update_timer.start(500)  # Update every 500ms
        
    def initUI(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add title
        title_label = QLabel("Motion Control")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        main_layout.addWidget(title_label)
        
        # Create compact grid for all controls
        control_grid = QGridLayout()
        control_grid.setSpacing(3)
        
        # Step size control (top row)
        control_grid.addWidget(QLabel("Step:"), 0, 0)
        self.step_spinbox = QDoubleSpinBox()
        self.step_spinbox.setRange(0.001, 10.0)
        self.step_spinbox.setValue(0.1)
        self.step_spinbox.setSingleStep(0.1)
        self.step_spinbox.setDecimals(3)
        self.step_spinbox.setMaximumWidth(70)
        control_grid.addWidget(self.step_spinbox, 0, 1)
        control_grid.addWidget(QLabel("mm"), 0, 2)
        
        # Home button (top row, right side)
        self.home_btn = QPushButton("Home All")
        self.home_btn.setMaximumWidth(80)
        self.home_btn.clicked.connect(self.home_all)
        control_grid.addWidget(self.home_btn, 0, 4, 1, 2)
        
        # Axis Labels
        control_grid.addWidget(QLabel("X:"), 1, 0)
        control_grid.addWidget(QLabel("Y:"), 2, 0)
        control_grid.addWidget(QLabel("Z:"), 3, 0)
        
        # Position displays
        self.x_pos_label = QLabel("0.000")
        self.y_pos_label = QLabel("0.000")
        self.z_pos_label = QLabel("0.000")
        self.x_pos_label.setMinimumWidth(50)
        self.y_pos_label.setMinimumWidth(50)
        self.z_pos_label.setMinimumWidth(50)
        control_grid.addWidget(self.x_pos_label, 1, 1)
        control_grid.addWidget(self.y_pos_label, 2, 1)
        control_grid.addWidget(self.z_pos_label, 3, 1)
        
        # Axis unit labels
        control_grid.addWidget(QLabel("mm"), 1, 2)
        control_grid.addWidget(QLabel("mm"), 2, 2)
        control_grid.addWidget(QLabel("mm"), 3, 2)
        
        # Movement buttons for X, Y, Z
        # X axis
        self.x_minus_btn = QPushButton("◀")
        self.x_minus_btn.setMaximumWidth(30)
        self.x_minus_btn.clicked.connect(lambda: self.move_axis_relative("X", -self.step_spinbox.value()))
        control_grid.addWidget(self.x_minus_btn, 1, 3)
        
        self.x_plus_btn = QPushButton("▶")
        self.x_plus_btn.setMaximumWidth(30)
        self.x_plus_btn.clicked.connect(lambda: self.move_axis_relative("X", self.step_spinbox.value()))
        control_grid.addWidget(self.x_plus_btn, 1, 4)
        
        # Y axis
        self.y_minus_btn = QPushButton("▼")
        self.y_minus_btn.setMaximumWidth(30)
        self.y_minus_btn.clicked.connect(lambda: self.move_axis_relative("Y", -self.step_spinbox.value()))
        control_grid.addWidget(self.y_minus_btn, 2, 3)
        
        self.y_plus_btn = QPushButton("▲")
        self.y_plus_btn.setMaximumWidth(30)
        self.y_plus_btn.clicked.connect(lambda: self.move_axis_relative("Y", self.step_spinbox.value()))
        control_grid.addWidget(self.y_plus_btn, 2, 4)
        
        # Z axis (focus)
        self.z_minus_btn = QPushButton("▼")
        self.z_minus_btn.setMaximumWidth(30)
        self.z_minus_btn.clicked.connect(lambda: self.move_axis_relative("Z", -self.step_spinbox.value()))
        control_grid.addWidget(self.z_minus_btn, 3, 3)
        
        self.z_plus_btn = QPushButton("▲")
        self.z_plus_btn.setMaximumWidth(30)
        self.z_plus_btn.clicked.connect(lambda: self.move_axis_relative("Z", self.step_spinbox.value()))
        control_grid.addWidget(self.z_plus_btn, 3, 4)
        
        main_layout.addLayout(control_grid)
        main_layout.setStretchFactor(control_grid, 0)
        
        # Add a small spacer to maintain compact size
        main_layout.addStretch(1)
        
    def move_axis_relative(self, axis, distance):
        """Move an axis by a relative distance"""
        if not self.motion_controller:
            print(f"Motion controller not available")
            return
            
        if self.move_worker and self.move_worker.isRunning():
            print(f"Move already in progress")
            return
            
        # Disable all move buttons during movement
        self.set_move_buttons_enabled(False)
        
        # Create and start the worker thread
        self.move_worker = StageMoveWorker(self.motion_controller, axis, distance)
        self.move_worker.finished_move.connect(self.move_complete)
        self.move_worker.move_failed.connect(self.move_failed)
        self.move_worker.start()
        
    def move_complete(self):
        """Handle completion of a move operation"""
        self.set_move_buttons_enabled(True)
        self.update_position_display()  # Update positions immediately
        
    def move_failed(self, error_message):
        """Handle failure of a move operation"""
        self.set_move_buttons_enabled(True)
        print(f"Move failed: {error_message}")
        QMessageBox.warning(self, "Move Failed", error_message)
        
    def set_move_buttons_enabled(self, enabled):
        """Enable or disable all movement buttons"""
        self.x_minus_btn.setEnabled(enabled)
        self.x_plus_btn.setEnabled(enabled)
        self.y_minus_btn.setEnabled(enabled)
        self.y_plus_btn.setEnabled(enabled)
        self.z_minus_btn.setEnabled(enabled)
        self.z_plus_btn.setEnabled(enabled)
        
    def home_all(self):
        """Home all axes"""
        if not self.motion_controller:
            print(f"Motion controller not available")
            return
            
        # Confirm with user
        reply = QMessageBox.question(
            self, "Home All Axes",
            "Are you sure you want to home all axes? This may take several minutes.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Disable buttons during homing
            self.setEnabled(False)
            QApplication.processEvents()  # Update UI
            
            # Create and start the thread
            self.home_worker = QThread()
            worker = HomeWorker(self.motion_controller)
            worker.moveToThread(self.home_worker)
            self.home_worker.started.connect(worker.run)
            worker.finished.connect(self.home_worker.quit)
            worker.finished.connect(worker.deleteLater)
            worker.finished.connect(self.home_complete)
            self.home_worker.finished.connect(self.home_worker.deleteLater)
            self.home_worker.start()
            
    def home_complete(self):
        """Re-enable controls after homing completes"""
        self.setEnabled(True)
        self.update_position_display()  # Update positions immediately
        QMessageBox.information(self, "Homing Complete", "All axes have been homed successfully.")
                
    def update_position_display(self):
        """Update position display labels"""
        if not self.motion_controller:
            return
            
        try:
            # Get current positions
            x_pos = self.motion_controller.get_position("X")
            y_pos = self.motion_controller.get_position("Y")
            z_pos = self.motion_controller.get_position("Z")
            
            # Update labels
            self.x_pos_label.setText(f"{x_pos:.3f}")
            self.y_pos_label.setText(f"{y_pos:.3f}")
            self.z_pos_label.setText(f"{z_pos:.3f}")
        except Exception as e:
            print(f"Error updating position display: {e}")

class StageMoveWorker(QThread):
    """Worker thread that performs stage movement operations"""
    finished_move = pyqtSignal()
    move_failed = pyqtSignal(str)
    
    def __init__(self, motion_controller, axis, distance, parent=None):
        super().__init__(parent)
        self.motion_controller = motion_controller
        self.axis = axis
        self.distance = distance
        
    def run(self):
        """Execute the stage movement"""
        try:
            if self.motion_controller:
                print(f"Moving {self.axis} axis by {self.distance} mm")
                result = self.motion_controller.move_relative(self.axis, self.distance)
                if not result:
                    self.move_failed.emit(f"Failed to move {self.axis} axis by {self.distance} mm")
                self.finished_move.emit()
            else:
                self.move_failed.emit("Motion controller not available")
        except Exception as e:
            self.move_failed.emit(f"Error moving stage: {str(e)}")
            print(f"Error in stage move worker: {e}")
        
class HomeWorker(QObject):
    """Worker for homing operations"""
    finished = pyqtSignal()
    
    def __init__(self, motion_controller):
        super().__init__()
        self.motion_controller = motion_controller
        
    def run(self):
        try:
            if self.motion_controller:
                self.motion_controller.home_all_axes()
        except Exception as e:
            print(f"Error in home worker: {e}")
        finally:
            self.finished.emit()

class ImageCapturePanel(QWidget):
    """Panel for image capture controls"""
    
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.initUI()
        self.capture_thread = None
        
    def initUI(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Add title
        title_label = QLabel("Image Capture")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)
        
        # File format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["TIFF (12-bit)", "PNG (8-bit)"])
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)
        
        # Capture button
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_btn)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Add spacer to push everything up
        layout.addStretch()
        
    def capture_image(self):
        """Capture and save the current image"""
        if not self.camera:
            self.status_label.setText("No camera available")
            return
            
        # Disable button during capture
        self.capture_btn.setEnabled(False)
        self.status_label.setText("Capturing...")
        QApplication.processEvents()  # Update UI
        
        try:
            # Get selected format
            is_tiff = "TIFF" in self.format_combo.currentText()
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = ".tiff" if is_tiff else ".png"
            filename = f"microscope_image_{timestamp}{extension}"
            
            # Create downloads folder path
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            filepath = os.path.join(downloads_dir, filename)
            
            saved_path = None
            
            # Use appropriate saving method based on format
            if is_tiff:
                # Save in full bit depth
                saved_path = self.camera.save_last_frame()
            else:
                # Save with 8-bit depth using the previous method
                with QMutexLocker(self.camera.processing_lock):
                    if self.camera.last_frame is None:
                        self.status_label.setText("No frame available")
                        self.capture_btn.setEnabled(True)
                        return
                        
                    # Process for 8-bit format (PNG)
                    frame = self.camera.last_frame.copy()
                    processed = self.camera.process_frame(frame)  # Use process_frame for color
                    
                    # Save using OpenCV
                    if processed.shape[2] == 3:  # RGB to BGR for OpenCV
                        save_frame = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    else:
                        save_frame = processed
                        
                    cv2.imwrite(filepath, save_frame)
                    saved_path = filepath
            
            if saved_path:
                self.status_label.setText(f"Saved: {os.path.basename(saved_path)}")
                print(f"Image saved to {saved_path}")
            else:
                self.status_label.setText("Failed to save image")
                
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Error capturing image: {e}")
            
        finally:
            self.capture_btn.setEnabled(True)

class LEDControlPanel(QWidget):
    """Panel for LED control"""
    
    def __init__(self, led_controller, parent=None):
        super().__init__(parent)
        self.led_controller = led_controller
        self.led_lock = QMutex()  # Add mutex for thread safety
        self.initUI()
        
    def initUI(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Add title
        title_label = QLabel("LED Control")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)
        
        # LED type and power controls in a single row
        controls_layout = QHBoxLayout()
        
        # LED type selection
        self.led_combo = QComboBox()
        self.led_combo.addItem("None")
        self.led_combo.setMaximumWidth(80)
        
        # Add available LED types
        if hasattr(self.led_controller, 'led_controllers'):
            for led_type in self.led_controller.led_controllers.keys():
                self.led_combo.addItem(led_type)
                
        self.led_combo.currentTextChanged.connect(self.led_selection_changed)
        controls_layout.addWidget(self.led_combo)
        
        # On/Off toggle
        self.led_enable = QCheckBox("Enable")
        self.led_enable.toggled.connect(self.led_enable_toggled)
        self.led_enable.setEnabled(False)  # Disabled until LED selected
        controls_layout.addWidget(self.led_enable)
        
        layout.addLayout(controls_layout)
        
        # Intensity slider
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity:"))
        self.led_slider = QSlider(Qt.Horizontal)
        self.led_slider.setRange(0, 100)
        self.led_slider.setValue(50)
        self.led_slider.valueChanged.connect(self.update_intensity_from_slider)
        self.led_slider.setEnabled(False)  # Disabled until LED selected
        intensity_layout.addWidget(self.led_slider)
        
        # Intensity text input
        self.led_intensity_input = QLineEdit("50")
        self.led_intensity_input.setValidator(QIntValidator(0, 100))
        self.led_intensity_input.setMaximumWidth(40)
        self.led_intensity_input.editingFinished.connect(self.update_intensity_from_text)
        self.led_intensity_input.setEnabled(False)  # Disabled until LED selected
        intensity_layout.addWidget(self.led_intensity_input)
        intensity_layout.addWidget(QLabel("%"))
        
        layout.addLayout(intensity_layout)
        
        # Status display
        self.led_status = QLabel("LED Status: Off")
        self.led_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.led_status)
        
        # Add spacer to keep compact
        layout.addStretch()
        
        # Update initial state
        QTimer.singleShot(100, self.update_led_status)
        
    def led_selection_changed(self, led_type):
        """Handle LED type selection change"""
        if not self.led_controller:
            return
            
        # First turn off any active LED
        if self.led_enable.isChecked():
            self.led_enable.blockSignals(True)
            self.led_enable.setChecked(False)
            self.led_enable.blockSignals(False)
            self.led_controller.turn_off_led()
            
        if led_type == "None":
            self.led_enable.setEnabled(False)
            self.led_slider.setEnabled(False)
            self.led_intensity_input.setEnabled(False)
            self.led_status.setText("LED Status: Not Selected")
            return
        
        # Reset intensity to default
        self.led_slider.setValue(50)
        self.led_intensity_input.setText("50")
        
        # Connect to the selected LED
        success = self.led_controller.connect_led(led_type)
        if success:
            self.led_enable.setEnabled(True)
            self.led_slider.setEnabled(True)
            self.led_intensity_input.setEnabled(True)
            self.led_status.setText(f"LED Status: {led_type} (Off)")
        else:
            self.led_enable.setEnabled(False)
            self.led_slider.setEnabled(False)
            self.led_intensity_input.setEnabled(False)
            self.led_status.setText(f"LED Status: Failed to connect to {led_type}")
                
    def led_enable_toggled(self, enabled):
        """Handle LED enable/disable toggle"""
        if not self.led_controller or not self.led_controller.current_led_type:
            return
            
        led_type = self.led_controller.current_led_type
        
        if enabled:
            success = self.led_controller.turn_on_led()
            if success:
                # Apply current intensity
                intensity = self.led_slider.value()
                self.led_controller.set_intensity(intensity)
                self.led_status.setText(f"LED Status: {led_type} (On - {intensity}%)")
            else:
                # Failed to turn on - reset the checkbox without triggering signals
                self.led_enable.blockSignals(True)
                self.led_enable.setChecked(False)
                self.led_enable.blockSignals(False)
                self.led_status.setText(f"LED Status: {led_type} (Failed to turn on)")
        else:
            self.led_controller.turn_off_led()
            self.led_status.setText(f"LED Status: {led_type} (Off)")
                
    def update_intensity_from_slider(self, value):
        """Update LED intensity from slider value"""
        self.led_intensity_input.setText(str(value))
        
        # Only update the LED if it's enabled
        if self.led_controller and self.led_controller.current_led_type and self.led_enable.isChecked():
            self.led_controller.set_intensity(value)
            self.led_status.setText(f"LED Status: {self.led_controller.current_led_type} (On - {value}%)")
                
    def update_intensity_from_text(self):
        """Update LED intensity from text input"""
        try:
            intensity = int(self.led_intensity_input.text())
            if 0 <= intensity <= 100:
                self.led_slider.setValue(intensity)
                # Update action handled by the slider's signal
            else:
                # Reset to slider value if out of range
                self.led_intensity_input.setText(str(self.led_slider.value()))
        except ValueError:
            # Reset to slider value if invalid
            self.led_intensity_input.setText(str(self.led_slider.value()))
            
    def update_led_status(self):
        """Update the LED status display to reflect current state"""
        if not self.led_controller or not self.led_controller.current_led_type:
            self.led_status.setText("LED Status: Not Selected")
            return
            
        led_type = self.led_controller.current_led_type
        if self.led_controller.enabled:
            intensity = self.led_controller.current_intensity
            self.led_status.setText(f"LED Status: {led_type} (On - {intensity}%)")
        else:
            self.led_status.setText(f"LED Status: {led_type} (Off)")

class CameraControlPanel(QWidget):
    """Panel for camera settings"""
    
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.initUI()
        
    def initUI(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Add title
        title_label = QLabel("Camera Control")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)
        
        # Exposure control
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("Exposure:"))
        
        # Exposure slider
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 300)  # 0.01s to 3.0s
        self.exposure_slider.setValue(10)      # Default 0.1s
        self.exposure_slider.valueChanged.connect(self.update_exposure_from_slider)
        exposure_layout.addWidget(self.exposure_slider)
        
        # Exposure text input
        self.exposure_input = QLineEdit("0.10")
        self.exposure_input.setValidator(QDoubleValidator(0.01, 3.0, 2))
        self.exposure_input.setMaximumWidth(60)
        self.exposure_input.editingFinished.connect(self.update_exposure_from_text)
        exposure_layout.addWidget(self.exposure_input)
        exposure_layout.addWidget(QLabel("s"))
        
        layout.addLayout(exposure_layout)
        
        # Add a separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # FPS display toggle
        fps_layout = QHBoxLayout()
        fps_toggle_btn = QPushButton("Toggle FPS Display")
        fps_toggle_btn.clicked.connect(self.toggle_fps_display)
        fps_layout.addWidget(fps_toggle_btn)
        layout.addLayout(fps_layout)
        
        # Add spacer to push everything up
        layout.addStretch()
        
    def update_exposure_from_slider(self, value):
        """Update exposure from slider value"""
        # Convert slider value (1-300) to seconds (0.01-3.0)
        exposure_sec = value / 100.0
        
        # Update text field
        self.exposure_input.setText(f"{exposure_sec:.2f}")
        
        # Update camera
        if self.camera:
            self.camera.set_exposure(exposure_sec)
            
    def update_exposure_from_text(self):
        """Update exposure from text input"""
        try:
            exposure_sec = float(self.exposure_input.text())
            if 0.01 <= exposure_sec <= 3.0:
                # Update slider (value 1-300)
                slider_value = int(exposure_sec * 100)
                self.exposure_slider.setValue(slider_value)
                
                # Update camera directly
                if self.camera:
                    self.camera.set_exposure(exposure_sec)
        except ValueError:
            # Reset to current slider value if invalid
            exposure_sec = self.exposure_slider.value() / 100.0
            self.exposure_input.setText(f"{exposure_sec:.2f}")
            
    def toggle_fps_display(self):
        """Toggle the FPS display in the main window"""
        # Get the main window
        main_window = self.window()
        if main_window and hasattr(main_window, 'camera_display'):
            main_window.camera_display.toggle_fps_display()

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MUSE Microscopy")
        self.resize(1200, 800)
        
        # Initialize hardware components
        self.camera = MicroscopeCamera()
        self.motion_controller = MotionController()
        self.led_controller = LEDController()
        
        # Initialize UI
        self.initUI()
        
        # Start camera if available
        if self.camera.camera:
            self.camera.start_acquisition()
            
            # Start camera worker thread
            self.camera_worker = CameraWorker(self.camera.camera, self)
            self.camera_worker.new_frame.connect(self.process_new_frame)
            self.camera_worker.error_occurred.connect(self.handle_camera_error)
            self.camera_worker.start()
        else:
            self.statusBar().showMessage("No camera detected")
        
    def initUI(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (horizontal)
        main_layout = QHBoxLayout(central_widget)
        
        # Left side - Camera view
        self.camera_display = CameraDisplayWidget()
        main_layout.addWidget(self.camera_display, 3)  # 3:1 ratio with control panel
        
        # Right side - Control panels
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Motion control panel
        self.motion_panel = MotionControlPanel(self.motion_controller)
        control_layout.addWidget(self.motion_panel)
        
        # LED control panel
        self.led_panel = LEDControlPanel(self.led_controller)
        control_layout.addWidget(self.led_panel)
        
        # Camera control panel
        self.camera_panel = CameraControlPanel(self.camera)
        control_layout.addWidget(self.camera_panel)
        
        # Image capture panel
        self.capture_panel = ImageCapturePanel(self.camera)
        control_layout.addWidget(self.capture_panel)
        
        # Add control panel to main layout
        main_layout.addWidget(control_panel, 1)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
    def process_new_frame(self, frame):
        """Process and display a new camera frame"""
        if frame is None:
            return
            
        try:
            # Process the frame using our camera's processing method
            processed_frame = self.camera.process_frame(frame)
            
            # Update the display
            self.camera_display.update_frame(processed_frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            
    def handle_camera_error(self, error_message):
        """Handle camera errors"""
        self.statusBar().showMessage(f"Camera error: {error_message}")
        QMessageBox.warning(self, "Camera Error", error_message)
        
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop camera thread
        if hasattr(self, 'camera_worker') and self.camera_worker.isRunning():
            self.camera_worker.stop()
            
        # Close hardware connections
        if self.camera:
            self.camera.close()
            
        if self.motion_controller:
            self.motion_controller.disconnect()
            
        if self.led_controller:
            self.led_controller.close()
            
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 