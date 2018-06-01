import os

dataset_dir = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_v1/"
sessions = os.listdir(dataset_dir)
# print(sessions)

Already ran once. Dont run again
# sessions.remove('s21')
# session_dir = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/s04/"

for session in sessions:
    session_dir = os.path.join(dataset_dir, session)
    for layout in os.listdir(session_dir):
        print("processing", session_dir, layout, "==============================")
        for vfolder in os.listdir(os.path.join(session_dir, layout)):
            if os.path.isdir(os.path.join(session_dir, layout, vfolder)):
                print("renaming...", vfolder, "to", vfolder + "_version1")
                os.rename(os.path.join(session_dir, layout, vfolder), os.path.join(session_dir, layout, vfolder + "_version1"))   

print(sessions)