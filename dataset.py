import numpy as np
from tensorflow.keras.utils import Sequence
import ROOT



class TauDatasetTrain(Sequence):
    def __init__(self, batch_size=6*50):
        self.batch_size = batch_size
        num_entries = []
        self.Z2qqbar = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2qqbar_train.root', 'r')
        self.Z2qqbarTemp = self.Z2qqbar.Get('data')
        num_entries.append(self.Z2qqbarTemp.GetEntries())
        
        self.Z2tau2pi = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pi_train.root')
        self.Z2tau2piTemp = self.Z2tau2pi.Get('data')
        num_entries.append(self.Z2tau2piTemp.GetEntries())
        
        self.Z2tau2pipizero = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipizero_train.root')
        self.Z2tau2pipizeroTemp = self.Z2tau2pipizero.Get('data')
        num_entries.append(self.Z2tau2pipizeroTemp.GetEntries())
        
        self.Z2tau2pipizeropizero = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipizeropizero_train.root')
        self.Z2tau2pipizeropizeroTemp = self.Z2tau2pipizeropizero.Get('data')
        num_entries.append(self.Z2tau2pipizeropizeroTemp.GetEntries())
        
        self.Z2tau2pipipi = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipipi_train.root')
        self.Z2tau2pipipiTemp = self.Z2tau2pipipi.Get('data')
        num_entries.append(self.Z2tau2pipipiTemp.GetEntries())
        
        self.Z2tau2pipipipizero = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipipipizero_train.root')
        self.Z2tau2pipipipizeroTemp = self.Z2tau2pipipipizero.Get('data')
        num_entries.append(self.Z2tau2pipipipizeroTemp.GetEntries())
        
        self.length = int(np.ceil((min(num_entries)*6)/self.batch_size))

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        start = int(index * self.batch_size/6)
        end = int((index + 1) * self.batch_size/6)
        images = []
        labels = []
        self.Z2qqbarTree = self.Z2qqbar.Get('data')
        self.Z2tau2piTree = self.Z2tau2pi.Get('data')
        self.Z2tau2pipizeroTree = self.Z2tau2pipizero.Get('data')
        self.Z2tau2pipizeropizeroTree = self.Z2tau2pipizeropizero.Get('data')
        self.Z2tau2pipipiTree = self.Z2tau2pipipi.Get('data')
        self.Z2tau2pipipipizeroTree = self.Z2tau2pipipipizero.Get('data')
        
        for i in range(start, end):
            self.Z2qqbarTree.GetEntry(i)
            self.Z2tau2piTree.GetEntry(i)
            self.Z2tau2pipizeroTree.GetEntry(i)
            self.Z2tau2pipizeropizeroTree.GetEntry(i)
            self.Z2tau2pipipiTree.GetEntry(i)
            self.Z2tau2pipipipizeroTree.GetEntry(i)
            
            Z2qqbarImage = np.concatenate(
                (np.array(self.Z2qqbarTree.S1).reshape(256,256,1),
                 np.array(self.Z2qqbarTree.S2).reshape(256,256,1),
                 np.array(self.Z2qqbarTree.C1).reshape(256,256,1),
                 np.array(self.Z2qqbarTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2qqbarLabel = np.array([0,0,0,0,0,1])
            images.append(Z2qqbarImage)
            labels.append(Z2qqbarLabel)
            
            Z2tau2piImage = np.concatenate(
                (np.array(self.Z2tau2piTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2piTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2piTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2piTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2piLabel = np.array([1,0,0,0,0,0])
            images.append(Z2tau2piImage)
            labels.append(Z2tau2piLabel)
            
            Z2tau2pipizeroImage = np.concatenate(
                (np.array(self.Z2tau2pipizeroTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeroTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeroTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeroTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipizeroLabel = np.array([0,0,0,1,0,0])
            images.append(Z2tau2pipizeroImage)
            labels.append(Z2tau2pipizeroLabel)
            
            Z2tau2pipizeropizeroImage = np.concatenate(
                (np.array(self.Z2tau2pipizeropizeroTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeropizeroTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeropizeroTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeropizeroTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipizeropizeroLabel = np.array([0,0,0,0,1,0])
            images.append(Z2tau2pipizeropizeroImage)
            labels.append(Z2tau2pipizeropizeroLabel)
            
            Z2tau2pipipiImage = np.concatenate(
                (np.array(self.Z2tau2pipipiTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipiTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipipiTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipiTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipipiLabel = np.array([0,1,0,0,0,0])
            images.append(Z2tau2pipipiImage)
            labels.append(Z2tau2pipipiLabel)
            
            Z2tau2pipipipizeroImage = np.concatenate(
                (np.array(self.Z2tau2pipipipizeroTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipipizeroTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipipipizeroTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipipizeroTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipipipizeroLabel = np.array([0,0,1,0,0,0])
            images.append(Z2tau2pipipipizeroImage)
            labels.append(Z2tau2pipipipizeroLabel)
        return np.array(images), np.array(labels)  
        
class TauDatasetValidation(Sequence):
    def __init__(self, batch_size=6*20):
        self.batch_size = batch_size
        num_entries = []
        self.Z2qqbar = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2qqbar_validation.root', 'r')
        self.Z2qqbarTemp = self.Z2qqbar.Get('data')
        num_entries.append(self.Z2qqbarTemp.GetEntries())
        
        self.Z2tau2pi = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pi_validation.root')
        self.Z2tau2piTemp = self.Z2tau2pi.Get('data')
        num_entries.append(self.Z2tau2piTemp.GetEntries())
        
        self.Z2tau2pipizero = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipizero_validation.root')
        self.Z2tau2pipizeroTemp = self.Z2tau2pipizero.Get('data')
        num_entries.append(self.Z2tau2pipizeroTemp.GetEntries())
        
        self.Z2tau2pipizeropizero = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipizeropizero_validation.root')
        self.Z2tau2pipizeropizeroTemp = self.Z2tau2pipizeropizero.Get('data')
        num_entries.append(self.Z2tau2pipizeropizeroTemp.GetEntries())
        
        self.Z2tau2pipipi = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipipi_validation.root')
        self.Z2tau2pipipiTemp = self.Z2tau2pipipi.Get('data')
        num_entries.append(self.Z2tau2pipipiTemp.GetEntries())
        
        self.Z2tau2pipipipizero = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipipipizero_validation.root')
        self.Z2tau2pipipipizeroTemp = self.Z2tau2pipipipizero.Get('data')
        num_entries.append(self.Z2tau2pipipipizeroTemp.GetEntries())
        
        self.length = int(np.ceil((min(num_entries)*6)/self.batch_size))

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        start = int(index * self.batch_size/6)
        end = int((index + 1) * self.batch_size/6)
        images = []
        labels = []
        self.Z2qqbarTree = self.Z2qqbar.Get('data')
        self.Z2tau2piTree = self.Z2tau2pi.Get('data')
        self.Z2tau2pipizeroTree = self.Z2tau2pipizero.Get('data')
        self.Z2tau2pipizeropizeroTree = self.Z2tau2pipizeropizero.Get('data')
        self.Z2tau2pipipiTree = self.Z2tau2pipipi.Get('data')
        self.Z2tau2pipipipizeroTree = self.Z2tau2pipipipizero.Get('data')
        
        for i in range(start, end):
            self.Z2qqbarTree.GetEntry(i)
            self.Z2tau2piTree.GetEntry(i)
            self.Z2tau2pipizeroTree.GetEntry(i)
            self.Z2tau2pipizeropizeroTree.GetEntry(i)
            self.Z2tau2pipipiTree.GetEntry(i)
            self.Z2tau2pipipipizeroTree.GetEntry(i)
            
            Z2qqbarImage = np.concatenate(
                (np.array(self.Z2qqbarTree.S1).reshape(256,256,1),
                 np.array(self.Z2qqbarTree.S2).reshape(256,256,1),
                 np.array(self.Z2qqbarTree.C1).reshape(256,256,1),
                 np.array(self.Z2qqbarTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2qqbarLabel = np.array([0,0,0,0,0,1])
            images.append(Z2qqbarImage)
            labels.append(Z2qqbarLabel)
            
            Z2tau2piImage = np.concatenate(
                (np.array(self.Z2tau2piTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2piTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2piTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2piTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2piLabel = np.array([1,0,0,0,0,0])
            images.append(Z2tau2piImage)
            labels.append(Z2tau2piLabel)
            
            Z2tau2pipizeroImage = np.concatenate(
                (np.array(self.Z2tau2pipizeroTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeroTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeroTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeroTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipizeroLabel = np.array([0,0,0,1,0,0])
            images.append(Z2tau2pipizeroImage)
            labels.append(Z2tau2pipizeroLabel)
            
            Z2tau2pipizeropizeroImage = np.concatenate(
                (np.array(self.Z2tau2pipizeropizeroTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeropizeroTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeropizeroTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeropizeroTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipizeropizeroLabel = np.array([0,0,0,0,1,0])
            images.append(Z2tau2pipizeropizeroImage)
            labels.append(Z2tau2pipizeropizeroLabel)
            
            Z2tau2pipipiImage = np.concatenate(
                (np.array(self.Z2tau2pipipiTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipiTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipipiTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipiTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipipiLabel = np.array([0,1,0,0,0,0])
            images.append(Z2tau2pipipiImage)
            labels.append(Z2tau2pipipiLabel)
            
            Z2tau2pipipipizeroImage = np.concatenate(
                (np.array(self.Z2tau2pipipipizeroTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipipizeroTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipipipizeroTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipipizeroTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipipipizeroLabel = np.array([0,0,1,0,0,0])
            images.append(Z2tau2pipipipizeroImage)
            labels.append(Z2tau2pipipipizeroLabel)
        return np.array(images), np.array(labels)  
        
        
class TauDatasetTest(Sequence):
    def __init__(self, batch_size=6*20):
        self.batch_size = batch_size
        num_entries = []
        self.Z2qqbar = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2qqbar_test.root', 'r')
        self.Z2qqbarTemp = self.Z2qqbar.Get('data')
        num_entries.append(self.Z2qqbarTemp.GetEntries())
        
        self.Z2tau2pi = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pi_test.root')
        self.Z2tau2piTemp = self.Z2tau2pi.Get('data')
        num_entries.append(self.Z2tau2piTemp.GetEntries())
        
        self.Z2tau2pipizero = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipizero_test.root')
        self.Z2tau2pipizeroTemp = self.Z2tau2pipizero.Get('data')
        num_entries.append(self.Z2tau2pipizeroTemp.GetEntries())
        
        self.Z2tau2pipizeropizero = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipizeropizero_test.root')
        self.Z2tau2pipizeropizeroTemp = self.Z2tau2pipizeropizero.Get('data')
        num_entries.append(self.Z2tau2pipizeropizeroTemp.GetEntries())
        
        self.Z2tau2pipipi = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipipi_test.root')
        self.Z2tau2pipipiTemp = self.Z2tau2pipipi.Get('data')
        num_entries.append(self.Z2tau2pipipiTemp.GetEntries())
        
        self.Z2tau2pipipipizero = ROOT.TFile('/store/ml/dual-readout/TauID/hadronic/Z2tau2pipipipizero_test.root')
        self.Z2tau2pipipipizeroTemp = self.Z2tau2pipipipizero.Get('data')
        num_entries.append(self.Z2tau2pipipipizeroTemp.GetEntries())
        
        self.length = int(np.ceil((min(num_entries)*6)/self.batch_size))

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        start = int(index * self.batch_size/6)
        end = int((index + 1) * self.batch_size/6)
        images = []
        labels = []
        self.Z2qqbarTree = self.Z2qqbar.Get('data')
        self.Z2tau2piTree = self.Z2tau2pi.Get('data')
        self.Z2tau2pipizeroTree = self.Z2tau2pipizero.Get('data')
        self.Z2tau2pipizeropizeroTree = self.Z2tau2pipizeropizero.Get('data')
        self.Z2tau2pipipiTree = self.Z2tau2pipipi.Get('data')
        self.Z2tau2pipipipizeroTree = self.Z2tau2pipipipizero.Get('data')
        
        for i in range(start, end):
            self.Z2qqbarTree.GetEntry(i)
            self.Z2tau2piTree.GetEntry(i)
            self.Z2tau2pipizeroTree.GetEntry(i)
            self.Z2tau2pipizeropizeroTree.GetEntry(i)
            self.Z2tau2pipipiTree.GetEntry(i)
            self.Z2tau2pipipipizeroTree.GetEntry(i)
            
            Z2qqbarImage = np.concatenate(
                (np.array(self.Z2qqbarTree.S1).reshape(256,256,1),
                 np.array(self.Z2qqbarTree.S2).reshape(256,256,1),
                 np.array(self.Z2qqbarTree.C1).reshape(256,256,1),
                 np.array(self.Z2qqbarTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2qqbarLabel = np.array([0,0,0,0,0,1])
            images.append(Z2qqbarImage)
            labels.append(Z2qqbarLabel)
            
            Z2tau2piImage = np.concatenate(
                (np.array(self.Z2tau2piTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2piTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2piTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2piTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2piLabel = np.array([1,0,0,0,0,0])
            images.append(Z2tau2piImage)
            labels.append(Z2tau2piLabel)
            
            Z2tau2pipizeroImage = np.concatenate(
                (np.array(self.Z2tau2pipizeroTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeroTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeroTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeroTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipizeroLabel = np.array([0,0,0,1,0,0])
            images.append(Z2tau2pipizeroImage)
            labels.append(Z2tau2pipizeroLabel)
            
            Z2tau2pipizeropizeroImage = np.concatenate(
                (np.array(self.Z2tau2pipizeropizeroTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeropizeroTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeropizeroTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipizeropizeroTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipizeropizeroLabel = np.array([0,0,0,0,1,0])
            images.append(Z2tau2pipizeropizeroImage)
            labels.append(Z2tau2pipizeropizeroLabel)
            
            Z2tau2pipipiImage = np.concatenate(
                (np.array(self.Z2tau2pipipiTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipiTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipipiTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipiTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipipiLabel = np.array([0,1,0,0,0,0])
            images.append(Z2tau2pipipiImage)
            labels.append(Z2tau2pipipiLabel)
            
            Z2tau2pipipipizeroImage = np.concatenate(
                (np.array(self.Z2tau2pipipipizeroTree.S1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipipizeroTree.S2).reshape(256,256,1),
                 np.array(self.Z2tau2pipipipizeroTree.C1).reshape(256,256,1),
                 np.array(self.Z2tau2pipipipizeroTree.C2).reshape(256,256,1)),
                axis=-1)
            Z2tau2pipipipizeroLabel = np.array([0,0,1,0,0,0])
            images.append(Z2tau2pipipipizeroImage)
            labels.append(Z2tau2pipipipizeroLabel)
        return np.array(images), np.array(labels)  
        
if __name__ == '__main__':
    trainset = TauDatasetTrain()
    valset = TauDatasetValidation()
    testset = TauDatasetTest()
    print(len(trainset))
    #print(len(valset))
    #print(len(testset))
    x, y = trainset[0]
    print(x.shape)
    print(y.shape)
