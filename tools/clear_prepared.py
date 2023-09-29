import os
import shutil

if __name__ == '__main__':
    # Do nothing if the prepared data directory does not exist
    if not os.path.exists('data/prepared/'):
        exit(0)
    # Delete all subdirectories in the prepared data directory
    shutil.rmtree('data/prepared/')
    # Recreate the prepared data directory
    os.mkdir('data/prepared/')
