import sys
import os
sys.path.append("../")
from MusicGeneration.api.Engine import EngineAPI

if __name__=="__main__":
    musicEngine = EngineAPI()
    musicEngine.cortexInit()

    # 输入路径
    input_path = "../trainSongs/"

    # 学习
    for composerName in os.listdir(input_path):
        dpath = os.path.join(input_path,composerName)
        if os.path.isdir(dpath):
            for musicName in os.listdir(dpath):
                fileName = (os.path.join(dpath,musicName))
                musicEngine.memorizing(musicName, composerName, 20, fileName)

    musicEngine.recallMusic("mz_331_3.mid")