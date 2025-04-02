import time
import clr
kinesis_path = r"C:\Program Files\Thorlabs\Kinesis"
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.DCServoCLI.dll")

from Thorlabs.MotionControl.KCube.DCServoCLI import *
from Thorlabs.MotionControl.DeviceManagerCLI import *
from System import Decimal
from System import Decimal, Int32

def test_z_axis():
    serial_no = "27265400"  # Z axis serial number
    DeviceManagerCLI.BuildDeviceList()
    print("Device list:", list(DeviceManagerCLI.GetDeviceList()))

    # Create and connect to the Z-axis device
    z_device = KCubeDCServo.CreateKCubeDCServo(serial_no)
    z_device.Connect(serial_no)
    time.sleep(0.25)
    z_device.StartPolling(250)
    time.sleep(0.25)
    z_device.EnableDevice()
    time.sleep(0.25)

    if not z_device.IsSettingsInitialized():
        z_device.WaitForSettingsInitialized(10000)
    if not z_device.IsSettingsInitialized():
        print("Z axis settings failed to initialize!")
        return

    # Load configuration and set settings
    m_config = z_device.LoadMotorConfiguration(serial_no, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)
    m_config.UpdateCurrentConfiguration()
    z_device.SetSettings(z_device.MotorDeviceSettings, True, False)

    # Home the device
    print("Homing Z axis...")
    z_device.Home(60000)
    time.sleep(1)

    try:
        initial_pos = float(str(z_device.Position))
        print("Initial Z position:", initial_pos, "mm")
    except Exception as e:
        print("Error reading initial position:", e)
        return

    # Command a relative move of 0.01 mm (or adjust as needed)
    commanded_step = 20  # mm
    print("Commanding a relative move of", commanded_step, "mm")
    try:
        # Try relative move (if supported)
        z_device.MoveRelative(Decimal(commanded_step), Int32(20000))
    except Exception as e:
        print("Relative move failed with error:", e)
        # Fallback: compute absolute move
        new_pos = initial_pos + commanded_step
        print("Fallback: Absolute move to", new_pos, "mm")
        z_device.MoveTo(Decimal(new_pos), Int32(20000))

    time.sleep(1)

    try:
        new_pos = float(str(z_device.Position))
        print("New Z position:", new_pos, "mm")
        print("Difference:", new_pos - initial_pos, "mm")
    except Exception as e:
        print("Error reading new position:", e)

    # Clean up
    z_device.StopPolling()
    z_device.DisableDevice()
    z_device.Disconnect()

if __name__ == "__main__":
    test_z_axis()
