import pylablib as pll
import cv2
import numpy as np
from pylablib.devices import Thorlabs

pll.par["devices/dlls/thorlabs_tlcam"] = "path/to/dlls"

cam = Thorlabs.ThorlabsTLCamera()
cameras = Thorlabs.list_cameras_tlcam()
print(cameras)
camera = Thorlabs.ThorlabsTLCamera(serial=cameras[0])

#Adjust camera settings
print("Current Exposure: ", camera.get_exposure())
new_exposure = float(input("New Exposure: "))
camera.set_exposure(new_exposure)

camera.start_acquisition()
k = input("Enter to grab")
image = camera.grab(1)[0]
image_array = np.array(image)
print("Image shape: ", image_array.shape)
k = input("Enter to save")
image_path = r"C:\Users\Public\musemicroscopy\captured_image.png"
image_uint8 = (image_array / image_array.max() * 255).astype('uint8')
cv2.imwrite(image_path, image_uint8)
camera.stop_acquisition()
k = input("Enter to end")
