class Conf():
    def __init__(self, neutype="LIF", task="MusicLearning", dt=0.1):

        self.neuron_type = neutype
        self.task = task
        self.dt = dt
        self.notesMap = {}
        self.GenreMap = {}
        self.RunTimeState = 0

    def readNoteFiles(self):
        f = open("../trainSongs/MIDID.txt", "r")
        while (True):
            line = f.readline()
            if not line:
                break
            else:
                strs = line.split(":")
                index = int(strs[0])
                self.notesMap[index] = strs[1].strip()
        f.close()

    def readGenreFils(self):
        f = open("../trainSongs/Genre.txt", "r")
        while (True):
            line = f.readline()
            if not line:
                break
            else:
                strs = line.split(":")
                g = strs[0].strip()
                ns = strs[1].split(",")
                for n in ns:
                    self.GenreMap[(n.strip()).title()] = g.title()
        f.close()


configs = Conf()
configs.readNoteFiles()
configs.readGenreFils()