import pylablib as pll
import cv2
pll.par["devices/dlls/thorlabs_tlcam"] = "path/to/dlls"
from pylablib.devices import Thorlabs
cam = Thorlabs.ThorlabsTLCamera()
cameras = Thorlabs.list_cameras_tlcam()
print(cameras)
camera = Thorlabs.ThorlabsTLCamera(serial=cameras[0])
camera.start_acquisition()
k = input("Enter to grab")
image = camera.grab(1)
k = input("Enter to save")
image_path ="C:\Users\Public\musemicroscopy\captured_image.png"
image_uint8 = (image \ image.max() * 255).astype('uint8')
cv2.imwrite(image_path, image_uint8)
camera.stop_acquisition()
k = input("Enter to end")