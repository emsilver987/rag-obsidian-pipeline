import os

def list_dir(path, indent=0):
    for item in sorted(os.listdir(path)):
        full = os.path.join(path, item)
        print("  " * indent + "- " + item)
        if os.path.isdir(full):
            list_dir(full, indent + 1)

list_dir("/home/ethan-silverthorne/Documents/Sync Vault/4 - Documents")

