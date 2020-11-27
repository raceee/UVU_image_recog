import os
import csv
zero_names = []
one_names = []
dir_name = r"*root dir to all images*"
switch = False
for file in os.listdir(dir_name):
    if ".py" not in file:
        sub_folder_name = os.path.join(dir_name, file)
        print(sub_folder_name)
        for image in os.listdir(sub_folder_name):
            print(image)
            if "1.JPG" in image:
                switch = True
            if switch:
                one_names.append((image, 1))
            else:
                zero_names.append((image, 0))
        print(" ")
print(zero_names)
with open("images_data.csv", "w") as f:
    csv_out = csv.writer(f)
    csv_out.writerow(['image_name', 'y_val'])
    csv_out.writerows(zero_names)
    csv_out.writerows(one_names)

