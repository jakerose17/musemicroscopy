from thorcam import TLCameraSDK

def main():
    # Initialize the SDK
    with TLCameraSDK() as sdk:
        # Discover available cameras
        available_cameras = sdk.discover_available_cameras()
        if not available_cameras:
            print("No cameras detected.")
            return

        # Open the first camera
        with sdk.open_camera(available_cameras[0]) as camera:
            print(f"Opened camera with serial number: {available_cameras[0]}")

            # Configure camera settings (example settings)
            camera.frames_per_trigger_zero_for_unlimited = 0  # Continuous mode
            camera.exposure_time_us = 50000  # Set exposure time to 50ms (adjust as needed)

            # Arm and capture a frame
            camera.arm(2)
            camera.issue_software_trigger()

            # Retrieve the frame
            frame = camera.get_pending_frame_or_null()
            if frame is not None:
                print("Captured frame data:", frame.image_buffer)
            else:
                print("No frame captured.")

            # Disarm after capturing
            camera.disarm()

if __name__ == "__main__":
    main()
