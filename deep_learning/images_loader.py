import os

input_dir = "/files/images/"
target_dir = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/classification_maps/"
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, file_name)
        for file_name in os.listdir(input_dir)
        if file_name.endswith(".jpg")
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, file_name)
        for file_name in os.listdir(target_dir)
        if file_name.endswith(".png") and not file_name.startswith(".")
    ]
)

print("Number of images:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)
