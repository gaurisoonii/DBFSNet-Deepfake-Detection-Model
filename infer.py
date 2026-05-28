"""
infer.py  (root-level entry point)
Run from C:/Users/sanid/Desktop/DBFS :
    python infer.py --input "C:/Users/sanid/Desktop/DBFS/data/original_sequences/youtube/c40/videos/000.mp4"
    python infer.py --input "path/to/face.jpg" --explain
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.infer import main

if __name__ == "__main__":
    main()
