import os

def list_dir(path):
    files = []
    for root, _, filenames in os.walk(path):
        for f in sorted(filenames):
            if f.endswith(".md"):
                full = os.path.join(root, f)
                files.append(full)
    return files

