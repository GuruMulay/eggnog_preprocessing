import os

sessions = os.listdir("/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/")
print(sessions)

session_dir = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/s07/"

for layout in os.listdir(session_dir):
    print("processing", session_dir, layout, "==============================")
    for vf in os.listdir(os.path.join(session_dir, layout)):
        print("processing video", vf)
        video_dir = os.path.join(session_dir, layout, vf)
        for n in os.listdir(video_dir):
            if n.endswith(".npy"):
#                 print(video_dir + n, video_dir + n.split("skfr")[0] + "skfr_00000.npy")
                os.rename(os.path.join(video_dir, n), os.path.join(video_dir, n.split("skfr")[0] + "skfr_00000.npy"))

    
