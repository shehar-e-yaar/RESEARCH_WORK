from roboflow import Roboflow

def download_dataset():
    print("Initializing Roboflow...")
    rf = Roboflow(api_key="nvSMsXnu4b62mWW7RTwG")
    project = rf.workspace("aiexample-c8c2y").project("snake-emula")
    version = project.version(2)
    print("Downloading dataset in yolov8 format...")
    dataset = version.download("yolov8")
    print(f"Dataset downloaded successfully to: {dataset.location}")

if __name__ == "__main__":
    download_dataset()
