import bpy

file_text = []

with open(bpy.path.abspath("//positions_autogen.txt")) as file:
    file_text = file.readlines()

current_frame = 0
for line in file_text:
    if line.startswith("#"):
        continue
    if line.startswith("Step"):
        current_frame = int(line.split()[1])
#        bpy.context.scene.frame_set(current_frame)
    else:
        elements = line.split()
        body_id = int(elements[0])
        name = f"Cube.{body_id:03d}"
        x = float(elements[1])
        y = float(elements[2])
        z = float(elements[3])
        bpy.data.objects[name].location = (x,y,z)
        w = float(elements[5])
        x = float(elements[6])
        y = float(elements[7])
        z = float(elements[8])
        bpy.data.objects[name].rotation_mode = "QUATERNION"
        bpy.data.objects[name].rotation_quaternion = (w,x,y,z)
        bpy.data.objects[name].keyframe_insert(data_path="location", frame=current_frame)
        bpy.data.objects[name].keyframe_insert(data_path="rotation_quaternion", frame=current_frame)

try:
    with open(bpy.path.abspath("//positions_valid_autogen.txt")) as file:
        file_text = file.readlines()
except:
    print("No validations file; skipping")
    file_text = []

current_frame = 0
for line in file_text:
    if line.startswith("Step"):
        current_frame = int(line.split()[1])
#        bpy.context.scene.frame_set(current_frame)
    else:
        elements = line.split()
        body_id = int(elements[0])
        name = f"Cube_valid.{body_id:03d}"
        x = float(elements[1])
        y = float(elements[2])
        z = float(elements[3])
        bpy.data.objects[name].location = (x,y,z)
        w = float(elements[5])
        x = float(elements[6])
        y = float(elements[7])
        z = float(elements[8])
        bpy.data.objects[name].rotation_mode = "QUATERNION"
        bpy.data.objects[name].rotation_quaternion = (w,x,y,z)
        bpy.data.objects[name].keyframe_insert(data_path="location", frame=current_frame)
        bpy.data.objects[name].keyframe_insert(data_path="rotation_quaternion", frame=current_frame)
