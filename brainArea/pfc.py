import numpy as np
import math
from braincog.base.brainarea.PFC import PFC
from MusicGeneration.Modal.synapse import Synapse
from MusicGeneration.Modal.titlelayer import TitleLayer
from MusicGeneration.Modal.composerlayer import ComposerLayer
from MusicGeneration.Modal.genrelayer import GenreLayer


class PFC(PFC):


    def __init__(self, neutype):

        super().__init__()
        self.neutype = neutype
        # 存储音乐名
        self.goals = TitleLayer(self.neutype)
        # 存储作者
        self.composers = ComposerLayer(self.neutype)
        # 存储流派
        self.genres = GenreLayer(self.neutype)

    def forward(self,x):
        pass

    def addNewSubGoal(self, goalname):
        if (self.goals.groups.get(goalname) == None):
            self.goals.addNewGroups(len(self.goals.groups) + 1, 1, 1, goalname)

    def addNewComposer(self, composername):
        if (self.composers.groups.get(composername) == None):
            self.composers.addNewGroups(len(self.composers.groups) + 1, 1, 1, composername)

    def addNewGenre(self, genrename):
        if (self.genres.groups.get(genrename) == None):
            self.genres.addNewGroups(len(self.genres.groups) + 1, 1, 1, genrename)

    def setTestStates(self):
        self.goals.setTestStates()
        self.composers.setTestStates()
        self.genres.setTestStates()

    def doRecalling(self, goalname, asm):
        goal = self.goals.groups.get(goalname)
        result = {}
        sequences = asm.sequenceLayers.get(1).groups
        dt = 0.1
        time = np.arange(0, len(sequences) * 5, dt)

        for t in time:
            order = math.floor(t / 5) + 1

            for neu in goal.neurons:
                neu.I = 30
                neu.update_normal(dt, t)
            sg = sequences.get(order)
            for neu in sg.neurons:
                neu.updateCurrentOfLowerAndUpperLayer(t)
                neu.update(dt, t, 'test')
                if (neu.spike == True and result.get(order) == None):
                    result[int(order)] = neu.selectivity
        return result

    def doRecalling2(self, goalname, asm):
        goal = self.goals.groups.get(goalname)
        result = {}
        for tindex, strack in asm.sequenceLayers.items():
            nsequences = strack.get("N").groups
            tsequences = strack.get("T").groups
            dic = {}
            ndic = {}
            tdic = {}
            dt = 0.1
            time = np.arange(0, len(nsequences) * 5, dt)
            for t in time:
                order = math.floor(t / 5) + 1
                for neu in goal.neurons:
                    neu.I = 30
                    neu.update_normal(dt, t)
                nsg = nsequences.get(order)
                for neu in nsg.neurons:
                    neu.updateCurrentOfLowerAndUpperLayer(t)
                    neu.update(dt, t, 'test')
                    if (neu.spike == True and ndic.get(order) == None):
                        ndic[int(order)] = neu.selectivity

                tsg = tsequences.get(order)
                for neu in tsg.neurons:
                    neu.updateCurrentOfLowerAndUpperLayer(t)
                    neu.update(dt, t, 'test')
                    if (neu.spike == True and tdic.get(order) == None):
                        tdic[int(order)] = neu.selectivity

            dic["N"] = ndic
            dic["T"] = tdic
            result[tindex] = dic
        return result

    def doRemebering(self, goalname, dt, t):
        # 存储歌曲信息
        goal_group = self.goals.groups.get(goalname)
        for neu in goal_group.neurons:
            neu.I = 10
            neu.update_normal(dt, t)

    def doRememberingComposer(self, composername, dt, t):
        composer_group = self.composers.groups.get(composername)
        for neu in composer_group.neurons:
            neu.I = 10
            neu.update_normal(dt, t)

    def doRememberingGenre(self, genrename, dt, t):
        genre_group = self.genres.groups.get(genrename)
        for neu in genre_group.neurons:
            neu.I = 10
            neu.update_normal(dt, t)

    def innerLearning(self, goalname, composer, genre):
        g = self.goals.groups.get(goalname)
        c = self.composers.groups.get(composer)
        gre = self.genres.groups.get(genre)
        if (g != None and c != None):
            for n1 in c.neurons:
                if (len(n1.spiketime) > 0):
                    for n2 in g.neurons:
                        if (len(n2.spiketime) > 0):
                            temp = 0
                            for sp1 in n1.spiketime:
                                for sp2 in n2.spiketime:
                                    if (abs(sp1 - sp2) <= n1.tau_ref):
                                        temp += 1
                            if (temp >= 4):
                                syn = Synapse(n1, n2)
                                syn.type = 2
                                syn.weight = 5
                                n2.synapses.append(syn)
                                n2.pre_neurons.append(n1)

        if (gre != None):
            for n1 in gre.neurons:
                if (len(n1.spiketime) > 0):
                    if (c != None):
                        for n2 in c.neurons:
                            if (len(n2.spiketime) > 0):
                                temp = 0
                                for sp1 in n1.spiketime:
                                    for sp2 in n2.spiketime:
                                        if (abs(sp1 - sp2) <= n1.tau_ref):
                                            temp += 1
                                if (temp >= 4):
                                    syn = Synapse(n1, n2)
                                    syn.type = 2
                                    syn.weight = 5
                                    n2.synapses.append(syn)
                                    n2.pre_neurons.append(n1)
                    if (g != None):
                        for n2 in g.neurons:
                            if (len(n2.spiketime) > 0):
                                temp = 0
                                for sp1 in n1.spiketime:
                                    for sp2 in n2.spiketime:
                                        if (abs(sp1 - sp2) <= n1.tau_ref):
                                            temp += 1
                                if (temp >= 4):
                                    syn = Synapse(n1, n2)
                                    syn.type = 2
                                    syn.weight = 5
                                    n2.synapses.append(syn)
                                    n2.pre_neurons.append(n1)

    def inhibiteGoals(self, dt, t):
        goal_group = self.goals.groups
        for g in goal_group.values():
            for neu in g.neurons:
                neu.I = -100
                neu.update(dt, t)

    def inhibitComposers(self, dt, t):
        com_group = self.composers.groups
        for g in com_group.values():
            for neu in g.neurons:
                neu.I = -100
                neu.update(dt, t)

