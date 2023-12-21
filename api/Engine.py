

from MusicGeneration.conf.conf import *
from MusicGeneration.brainArea.Cortex import Cortex
import pretty_midi
import math
import json

from pygments.lexers import configs


class EngineAPI():

    def __init__(self):
        self.cortex = Cortex(configs.neuron_type, configs.dt)

    def cortexInit(self):
        self.cortex.musicSequenceMemroyInit()

    def rememberMusic(self, muiscName, composerName="None"):
        muiscName = muiscName.title()
        composerName = composerName.title()
        self.cortex.pfc.setTestStates()
        self.cortex.msm.setTestStates()
        self.cortex.addSubGoalToPFC(muiscName)
        self.cortex.addComposerToPFC(composerName)
        genreName = configs.GenreMap.get(composerName)
        self.cortex.addGenreToPFC(genreName)
        self.cortex.pfc.innerLearning(muiscName, composerName, genreName)

        goaldic = {}
        composerdic = {}
        genredic = {}
        if (configs.RunTimeState == 1):
            g = self.cortex.pfc.titles.groups.get(muiscName)
            c = self.cortex.pfc.composers.groups.get(composerName)
            gre = self.cortex.pfc.genres.groups.get(genreName)
            goaldic = g.writeSelfInfoToJson("IPS")
            composerdic = c.writeSelfInfoToJson("Composer")
            genredic = gre.writeSelfInfoToJson("Genre")

        return goaldic, composerdic

    def rememberMIDIMusic(self, musicName, composerName, noteLength, fileName):
        musicName = musicName.title()
        composerName = composerName.title()
        print(musicName + " processing...")
        pm = pretty_midi.PrettyMIDI(fileName)
        genreName = configs.GenreMap.get(composerName)
        for i, ins in enumerate(pm.instruments):
            if (i >= 1): break;
            if (self.cortex.msm.sequenceLayers.get(i + 1) == None):
                # 创建一个新的layer来存储音轨
                self.cortex.msm.createActionSequenceMem(i + 1, self.cortex.neutype)
            self.rememberTrackNotes(musicName, composerName, genreName, i + 1, ins, pm, noteLength)
        print(musicName + " finished!")

    def rememberTrackNotes(self, musicName, composerName, genreName, trackIndex, track, pm, noteLength):
        r_notes = []
        r_intervals = []
        total_dic = {}

        print(track)
        if(noteLength == "ALL"):
            noteLength = len(track.notes)
        order = 1
        i = 0
        while (i < noteLength):
            note = track.notes[i]
            start = pm.time_to_tick(note.start)
            end = pm.time_to_tick(note.end)
            pitches = []
            durations = []
            restFlag = False
            if (i == 0):
                if (start >= 30):
                    pitches.append(-1)
                    durations.append(start / pm.resolution)
                    restFlag = True
            else:
                lastend = pm.time_to_tick(track.notes[i - 1].end)
                if (start - lastend >= 50):
                    pitches.append(-1)
                    durations.append((start - lastend) / pm.resolution)
                    restFlag = True
            if (restFlag == True):
                dic, g = self.rememberANote(musicName, composerName, genreName, trackIndex, pitches[0], order,
                                            durations[0], True)
                if (configs.RunTimeState == 1):
                    jstr = json.dumps(g)
                    self.conn.send('/Queue/SampleQueue', jstr)
                order = order + 1
                pitches = []
                durations = []

                # 和弦识别模块
            pitches.append(note.pitch)
            durations.append((end - start) / pm.resolution)
            j = i + 1
            while (j < len(track.notes)):
                nextstart = pm.time_to_tick(track.notes[j].start)
                nextend = pm.time_to_tick(track.notes[j].end)

                if (math.fabs(start - nextstart) <= 30 or end - nextstart >= 30):
                    pitches.append(track.notes[j].pitch)
                    durations.append((nextend - nextstart) / pm.resolution)
                    j = j + 1
                else:
                    break
            i = j

            if (i < noteLength):
                dic, g = self.rememberANote(musicName, composerName, genreName, trackIndex, pitches[0], order,
                                            durations[0], True)
                str1 = str(order) + ":("
                for t in range(len(pitches)):
                    str1 += str(pitches[t]) + "," + str(durations[t]) + ";"

                order = order + 1
                if (configs.RunTimeState == 1):
                    jstr = json.dumps(g)
                    self.conn.send('/Queue/SampleQueue', jstr)
                    nlist = dic.get('MSMSpike')
                    ns = []
                    for l in nlist:
                        n = l.get('Index')
                        ns.append(n)
                    r_notes.append(ns)
                    tlist = dic.get('MSMTSpike')
                    ts = []
                    for l in tlist:
                        t = l.get('Index')
                        ts.append(t * 60)
                    r_intervals.append(ts)
        return total_dic

    def rememberNotes(self, MusicName, notes, intervals, tempo=True):
        jStr = ''

        notesStr = notes.split(",")
        intervalsStr = intervals.split(",")
        intervaltimes = []
        for i in range(len(intervalsStr) - 1):
            intervaltimes.append(int(intervalsStr[i]))
        print(intervaltimes)
        for i, note in enumerate(notesStr):
            note = int(note)
            if (i < len(notesStr) - 1):
                tinterval = intervalsStr[i]
                tinterval = int(intervalsStr[i])
            self.rememberANote(MusicName, note, i + 1, tinterval, tempo)
        return jStr

    def rememberANote(self, MusicName, ComposerName, genreName, TrackIndex, NoteIndex, order, tinterval, tempo=False):
        if (tempo == False):
            dic = self.cortex.rememberANote(MusicName, NoteIndex, order)
            jsonStr = json.dumps(dic)
            return jsonStr
        else:
            dic, g = self.cortex.rememberANoteandTempo(MusicName, ComposerName, genreName, TrackIndex, NoteIndex, order,
                                                       tinterval)
            return dic, g

    def memorizing(self,MusicName, ComposerName, noteLength, fileName):
        self.rememberMusic(MusicName, ComposerName)
        self.rememberMIDIMusic(MusicName,ComposerName,noteLength, fileName)

    def recallMusic(self, musicName):
        print("Recall the " + musicName + " ......")
        musicName = musicName.title()
        result = self.cortex.recallMusicPFC(musicName)
        noteResult = {}
        for tindex,track in result.items():
            ns = track.get('N')
            ts = track.get('T')
            tmp = []
            for key in ns.keys():
                dic = {}
                dic['N']=ns.get(key)
                dic['T']=ts.get(key)
                tmp.append(dic)
            noteResult[tindex] = tmp
        self.writeMidiFile(musicName+"_recall",noteResult)
        print("Recall " + musicName + " finished!")
        return noteResult


    def generateEx_Nihilo(self, firstNote, durations, length,gen_fName):
        print("生成无风格旋律中...........")
        result = self.cortex.generateEx_Nihilo2(firstNote, durations, length)
        self.writeMidiFile(gen_fName,result)
        print("生成完成！")
        return result

    def generateEx_NihiloAccordingToGenre(self, genreName, firstNote, durations, length,gen_fName):

        print("生成 "+ genreName+"\'风格歌曲............")
        result = self.cortex.generateEx_NihiloAccordingToGenre(genreName, firstNote, durations, length)
        self.writeMidiFile(gen_fName,result)
        print("生成完成!")
        return result

    def generateEx_NihiloAccordingToComposer(self, composerName, firstNote, durations, length,gen_fName):

        print("生成 " + composerName + "'风格歌曲............")
        result = self.cortex.generateEx_NihiloAccordingToComposer(composerName, firstNote, durations, length)
        self.writeMidiFile(gen_fName,result)
        print("生成结束!")
        return result

    def generate2TrackMusic(self, firstNotes, durations, lengths):
        result = self.cortex.generate2TrackMusic(firstNotes, durations, lengths)
        return result


    def writeMidiFile(self,fileName, mudic):

        fileName += ".mid"
        pm = pretty_midi.PrettyMIDI()

        for values in mudic.values():

            piano = pretty_midi.Instrument(program=0)

            start = 0
            end = 0
            for i, n in enumerate(values):

                end = start + n.get('T')
                note_name = n.get('N')
                if (note_name == -1):
                    note = pretty_midi.Note(
                        velocity=0, pitch=0, start=start, end=end)
                else:
                    note = pretty_midi.Note(
                        velocity=100, pitch=note_name, start=start, end=end)

                piano.notes.append(note)
                start = end

            pm.instruments.append(piano)

        pm.write(fileName)