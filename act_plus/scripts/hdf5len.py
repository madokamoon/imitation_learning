import json
import pathlib

rootpath = pathlib.Path(__file__).parent.parent.parent
input_path = rootpath.joinpath("imitation_learning_ROS2/data/sample/test")

for epoch_folder in input_path.iterdir():
    if epoch_folder.is_dir():
        state_file = epoch_folder / "state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)
                print(f"{epoch_folder.name}: {len(data)}")
        else:
            print(f"{epoch_folder.name}: state.json not found")
