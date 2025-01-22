import pylablib as pll
pll.par["devices/dlls/thorlabs_tlcam"] = "path/to/dlls"
from pylablib.devices import Thorlabs
cam = Thorlabs.ThorlabsTLCamera()
cameras = Thorlabs.list_cameras_tlcam()
print(cameras)
camera = Thorlabs.ThorlabsTLCamera(serial=cameras[0])
camera.start_acquisition()
k = input("Enter to next")
image = camera.grab(1)
k = input("Enter to next")
camera.stop_acquisition()
k = input("Enter to end")