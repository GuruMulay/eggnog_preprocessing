import os
from pathlib import Path

sessions = os.listdir("/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/")
print(sessions)

session_dir = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/s04/"
    
for layout in os.listdir(session_dir):
    print("processing", session_dir, layout, "==============================")
    for vf in os.listdir(os.path.join(session_dir, layout)):
        print("processing video", vf)
        video_dir = os.path.join(session_dir, layout, vf)
        for n in os.listdir(video_dir):
            if n.endswith(".jpg"):  # remove the initial frames or any frames for which the .npy is not gerenrated
                npyfile = os.path.join(video_dir, n.split("_240x320")[0] + ".npy")
                if not Path(npyfile).exists():
                    print("removing", n)
                    os.remove(os.path.join(video_dir, n))