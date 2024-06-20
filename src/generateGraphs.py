import os
import subprocess

current_dir = os.getcwd()
py_files = [f for f in os.listdir(current_dir) if f.endswith('.py')]

generate_puml = False
generate_img = True

if generate_puml:
    for py_file in py_files:
        filename_without_py = py_file[:-3]
        command = f"pyreverse -o puml {py_file} -d ./src_uml --project {filename_without_py}"
        subprocess.run(command, shell=True)
elif generate_img:
    for py_file in py_files:
        filename_without_py = py_file[:-3]
        command = f"pyreverse {py_file} -d ./src_uml --project {filename_without_py}"
        created_filename = f"classes_{filename_without_py}"
        subprocess.run(command, shell=True)
        command = "cd src_uml"
        subprocess.run(command, shell=True)
        command = f"dot - Tpng {created_filename}.dot - o {created_filename}.png"
        subprocess.run(command, shell=True)
