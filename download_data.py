

from roboflow import Roboflow
rf = Roboflow(api_key="hX8eUnPivDRAfzfdfU5i")
project = rf.workspace("varadworkspace").project("ppe-vest-hardhat")
version = project.version(1)
dataset = version.download("yolov8")
                
print("Downloaded to:", dataset.location)
print("Version:", dataset.version)
                