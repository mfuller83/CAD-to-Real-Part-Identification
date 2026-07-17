import os
import shutil

def copy_matching_files(source_dir, destination_dir, prefix="12316", suffix="SLDPRT"):
    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)


    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.startswith(prefix) and file.endswith(suffix):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file)

                try:
                    shutil.copy2(source_path, destination_path)
                    print(f"Copied: {source_path} -> {destination_path}")
                except PermissionError:
                    print(f"Permission denied: {source_path}")
                except FileNotFoundError:
                    print(f"File not found: {source_path}")
                except shutil.SameFileError:
                    print(f"Source and destination are the same: {source_path}")
                except Exception as e:
                    print(f"Failed to copy {source_path}: {e}")


# Example usage:
source_folder = r"C:\Users\martin.fuller\12316-TaylorMade\05.Mechanical Design\05.Design"

destination_folder = r"C:\Users\martin.fuller\OneDrive - Expert Tooling & Automation Ltd\Projects\Python Projects\manufacturing_classifier\Data"

copy_matching_files(source_folder, destination_folder)