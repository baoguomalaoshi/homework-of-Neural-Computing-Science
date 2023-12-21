# 听觉皮层构建
from braincog.base.brainarea.BrainArea import  BrainArea
from MusicGeneration.Modal.sequencememory import SequenceMemory
from MusicGeneration.Modal.notesequencelayer import NoteSequenceLayer
from MusicGeneration.Modal.temposequencelayer import TempoSequenceLayer

class PAC(BrainArea,SequenceMemory):
    # 初始化
    def __init__(self,neutype):
        SequenceMemory.__init__(self, neutype)

    def forward(self, x):
        # do noting
        pass

    def doRemembering_note_only(self, note, order, dt, t):
        # remember note
        sl = self.sequenceLayers.get(1)
        sgroup = sl.groups.get(order)
        dt = 0.1
        for n in sgroup.neurons:
            n.I_ext = note.frequence
            n.computeFilterCurrent()
            n.update(dt, t, 'Learn')

    def doRemembering(self, trackIndex, noteIndex, order, dt, t, tinterval=0):
        # 记忆结点的构建
        iTrack = self.sequenceLayers.get(trackIndex)
        sl = iTrack.get("N")
        sgroup = sl.groups.get(order)
        dt = 0.1
        for n in sgroup.neurons:
            n.I_ext = noteIndex
            n.computeFilterCurrent()
            n.update(dt, t, 'Learn')

        # remember tempo
        tl = iTrack.get("T")
        tgroup = tl.groups.get(order)
        dt = 0.1
        for n in tgroup.neurons:
            n.I_ext = tinterval
            n.computeFilterCurrent()
            n.update(dt, t, 'Learn')

    def doConnectToTitle(self, title, track, order):
        for sl in track.values():
            self.doConnecting(title, sl, order)

    def doConnectToComposer(self, composer, track, order):
        for sl in track.values():
            self.doConnecting(composer, sl, order)

    def doConnectToGenre(self, genre, track, order):
        for sl in track.values():
            self.doConnecting(genre, sl, order)

    def generateEx_Nihilo(self, firstNote, durations, order, dt, t):
        ns = self.sequenceLayers.get(1).get("N")
        ts = self.sequenceLayers.get(1).get("T")
        nneurons = ns.groups.get(order + 1).neurons
        tneurons = ts.groups.get(order + 1).neurons
        # firstNode指定了触发后续音符的起始音符
        if (order < len(firstNote)):
            i = firstNote[order]
            nneu = nneurons[i + 1]
            nneu.I = 20
            nneu.update_normal(dt, t)

            d = int(durations[order] / 0.125) - 1
            tneu = tneurons[d]
            tneu.I = 20

            tneu.update_normal(dt, t)
        else:  # 生成下一个结点
            for nn in nneurons:
                nn.updateCurrentOfLowerAndUpperLayer(t)
                nn.update(dt, t, 'test')
            for tn in tneurons:
                tn.updateCurrentOfLowerAndUpperLayer(t)
                tn.update(dt, t, 'test')

    def generateSimgleTrackNotes(self, trackIndex, firstNote, durations, order, dt, t):
        ns = self.sequenceLayers.get(trackIndex).get("N")
        ts = self.sequenceLayers.get(trackIndex).get("T")
        nneurons = ns.groups.get(order + 1).neurons
        tneurons = ts.groups.get(order + 1).neurons
        if (order < len(firstNote)):
            i = firstNote[order]
            nneu = nneurons[i + 1]
            nneu.I = 20
            nneu.update_normal(dt, t)

            d = int(durations[order] / 0.125) - 1
            tneu = tneurons[d]
            tneu.I = 20
            tneu.update_normal(dt, t)
        else:
            for nn in nneurons:
                nn.updateCurrentOfLowerAndUpperLayer(t)
                nn.update(dt, t, 'test')
            for tn in tneurons:
                tn.updateCurrentOfLowerAndUpperLayer(t)
                tn.update(dt, t, 'test')