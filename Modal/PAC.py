'''
Primary Auditory Cortex
'''
import torch
from braincog.base.node.node import *
from braincog.base.brainarea.BrainArea import *
from braincog.base.connection import CustomLinear
from braincog.base.learningrule.STDP import *
from MusicGeneration.Modal.notelifneuron import NoteLIFNeuron


class PAC(BrainArea):

    def __int__(self,w,mask):
        self.noteNetworks = NoteLIFNeuron()
        self.connection = [CustomLinear(w,mask),CustomLinear(w2)]
        self.stdp = []
        self.internalinputs = torch.zeros(640,640)
        self.stdp.append(MutliInputSTDP(self.noteNetworks, self.connection))

    def forward(self, x):
        self.internalinputs,dw = self.stdp[0](x,self.internalinputs)
        return self.internalinputs, dw