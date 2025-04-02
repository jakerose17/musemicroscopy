#!/usr/bin/env python
"""
Standalone Detailed Debugging Script for upLED Devices

This script:
  1. Loads the TLUP_64.dll.
  2. Scans for upLED devices and prints each device’s model, serial number, and resource.
  3. For each device:
      - Attempts to initialize it using TLUP_init.
      - Prints the return code.
      - If TLUP_init returns -1073807360, prints a warning that no EEPROM was detected and falls back to a default current setpoint.
      - Otherwise, if EEPROM is detected (LEDWavelength != 0), sets the LED current to 10% of its limit.
      - Prints out the LED info.
  4. Closes the device.

By comparing output when only the UV LED is connected vs. when both are connected,
you can see why the UV LED is not being recognized in one scenario.
"""

import sys
import os
import time
from ctypes import cdll, c_uint32, c_int, c_double, byref, create_string_buffer

def main():
    dll_path = r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLUP_64.dll"
    try:
        lib = cdll.LoadLibrary(dll_path)
    except OSError as e:
        print(f"Error loading DLL at {dll_path}: {e}")
        sys.exit(1)

    # Find available devices.
    deviceCount = c_uint32()
    lib.TLUP_findRsrc(0, byref(deviceCount))
    print("Number of upLED devices found:", deviceCount.value)
    if deviceCount.value == 0:
        sys.exit(0)

    # List all devices with detailed basic info.
    devices_info = []
    for i in range(deviceCount.value):
        modelName = create_string_buffer(256)
        serialNumber = create_string_buffer(256)
        lib.TLUP_getRsrcInfo(i, 0, modelName, serialNumber, 0, 0)
        resourceName = create_string_buffer(256)
        lib.TLUP_getRsrcName(i, 0, resourceName)
        info = {
            "index": i,
            "model": modelName.value.decode().strip(),
            "serial": serialNumber.value.decode().strip(),
            "resource": resourceName.value.decode().strip()
        }
        devices_info.append(info)
        print(f"Device {i}: Model: {info['model']}, Serial: {info['serial']}, Resource: {info['resource']}")

    # For debugging: print expected serial numbers for each type.
    expected_white = "M00877195"
    expected_uv = "M00920749"
    print("\nExpected White LED Serial:", expected_white)
    print("Expected UV LED Serial:", expected_uv)

    # Enumerate through each device and print detailed LED info.
    uv_found = False
    for dev in devices_info:
        i = dev["index"]
        print("\n=== Enumerating device index", i, "===")
        print(f"Model: {dev['model']}, Serial: {dev['serial']}, Resource: {dev['resource']}")

        led_handle = c_int(0)
        res = lib.TLUP_init(dev["resource"].encode('utf-8'), 0, 0, byref(led_handle))
        print(f"TLUP_init returned {res} for device index {i}.")
        if res != 0:
            if res == -1073807360:
                print("No EEPROM detected on this LED device. Falling back to manual current setpoint.")
            else:
                print(f"Failed to initialize device {i} with resource '{dev['resource']}'. Skipping.")
                continue

        # Configure the device for non-Thorlabs mode and switch output off.
        lib.TLUP_setLedUseNonThorlabsLed(led_handle, 1)
        lib.TLUP_switchLedOutput(led_handle, 0)

        # Retrieve LED info.
        LEDName = create_string_buffer(256)
        LEDSerialNumber = create_string_buffer(256)
        LEDCurrentLimit = c_double()
        LEDForwardVoltage = c_double()
        LEDWavelength = c_double(0)
        lib.TLUP_getLedInfo(led_handle, LEDName, LEDSerialNumber,
                            byref(LEDCurrentLimit),
                            byref(LEDForwardVoltage),
                            byref(LEDWavelength))
        print("LED Info for device", i)
        print("  LED Name:", LEDName.value.decode().strip())
        print("  LED Serial:", LEDSerialNumber.value.decode().strip())
        print("  LED Current Limit (A):", LEDCurrentLimit.value)
        print("  LED Forward Voltage (V):", LEDForwardVoltage.value)
        print("  LED Wavelength (nm):", LEDWavelength.value)

        # Check for EEPROM presence.
        if LEDWavelength.value != 0:
            print("EEPROM on LED detected.")
            currentSetpoint = c_double(0.1 * LEDCurrentLimit.value)
            lib.TLUP_setLedCurrentSetpoint(led_handle, currentSetpoint)
            time.sleep(1)
            lib.TLUP_getLedCurrentSetpoint(led_handle, 0, byref(currentSetpoint))
            print("LED current set to [A]:", currentSetpoint.value)
        else:
            print("No EEPROM on LED detected. Using default 1 mA setpoint.")
            default_current = c_double(0.001)
            lib.TLUP_setLedCurrentSetpoint(led_handle, default_current)
            time.sleep(1)
            lib.TLUP_getLedCurrentSetpoint(led_handle, 0, byref(default_current))
            print("LED current set to [A]:", default_current.value)

        # Check if this device is a UV LED.
        if dev["serial"] == expected_uv:
            uv_found = True

        lib.TLUP_close(led_handle)
        print(f"Device {i} closed.")

    if not uv_found:
        print("\nWARNING: No device with the expected UV LED serial was found during enumeration.")
    else:
        print("\nA device with the expected UV LED serial was detected (though it may not have been initialized successfully).")

if __name__ == '__main__':
    main()
