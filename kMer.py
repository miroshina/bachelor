#! /usr/bin/python

import sys
import subprocess
from datetime import datetime
from sklearn import preprocessing
import Bio
from Bio import SeqIO, SeqUtils,bgzf
from Bio.SeqUtils import lcc
import gzip
import math
import pandas as pd
import datatable as dt
import os
import random
import seaborn as sn
import numpy as np
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import arange

class Kmers(object):

    def __init__(self, pathToSamples,kmerLength,disease):

        self.pathabundanceDir = ""
        self.kmerLength = kmerLength
        self.disease = disease
        self.pathwayMatrix = pd.DataFrame()
        self.pathToSamples = pathToSamples
        self.kmerMatrix = pd.DataFrame()

    def create_count_table(self, listOfKmerCountFiles, nsamples):
        kmerDB = {}
        pf = pd.DataFrame()
        sample = 0
        listOfKmerCountFiles=["merge.txt"]
        for file in listOfKmerCountFiles:
            filepath = './'+file
            with open(filepath) as file1:
                start_time = datetime.now()
                for line in file1:
                    if line.startswith("*"):
                        sample+=1
                        end_time = datetime.now()
                        print('Duration: {}'.format(end_time - start_time))
                        print("processed sample "+str(sample-1))
                    else:
                        kmer=line.split("\t")[0].rstrip()
                        count=int(line.split("\t")[1].rstrip())
                        if Bio.SeqUtils.lcc.lcc_simp(kmer)>1.5:
                            if kmer not in kmerDB.keys():
                                counts = [0] * nsamples
                                counts[sample]=count
                                kmerDB[kmer] = counts
                            else:
                                kmerDB[kmer][sample]=count
            file1.close()
        print("start saving to table")
        dt1 = pd.DataFrame.from_dict(kmerDB, orient='index')
        dt1= dt1.T
        dt1 = dt1.loc[:, ((dt1 == 0).sum(axis=0) < nsamples // 2)]
        dt1 = dt1.T
        dt1 = dt1.loc[~(dt1 <= 1).all(axis=1)]
        print("table is ready: ")
        print(dt1)
        dt1.to_csv('count.tsv', sep="\t")
        self.kmerMatrix = dt1
        self.concat()

    def count_kmers(self, listOfFiles,nsamples):
        sample = 0
        listOfKmersFiles=[]
        kmerDB={}
        for filepath in sorted(listOfFiles):
            kmerCountFile = 'kmers_in_sample_'+ str(sample)
            os.system('kmc -cs100000 -k' + str(self.kmerLength) + ' -ci20 ' + filepath +' '+ kmerCountFile + ' . >>kmc.log')
            sample += 1
            kmerTextFile = kmerCountFile + '.txt'
            os.system('kmc_tools transform '+kmerCountFile+' dump '+kmerTextFile)
            os.system('rm ./*_suf')
            os.system('rm ./*_pre')
            kmPath='./'+kmerTextFile
            tsv_data = (pd.read_csv(kmPath, sep='\t', names=["kmer", "count"]))
            tsv_data["compl"] = list(map(lambda x:  Bio.SeqUtils.lcc.lcc_simp(x), tsv_data["kmer"].values))
            tsv_data = tsv_data[tsv_data.compl > 1.5]
            tmp_dict=dict(zip(tsv_data["kmer"], tsv_data["count"]))
            for kmer in tmp_dict.keys():
                count = tmp_dict[kmer]
                if kmer not in kmerDB.keys():
                    counts = [0] * (nsamples-1)
                    counts[sample] = count
                    kmerDB[kmer] = counts
                else:
                    valueAsNP = np.array(counts)
                    kmerDB[kmer][sample] = count
            tmp_dict.clear()
            os.system('rm '+kmerTextFile)
            print("processed file "+str(sample-1))
        self.kmerMatrix = pd.DataFrame.from_dict(kmerDB, orient='index')
        dt1= self.kmerMatrix.T
        dt1 = dt1.loc[:, ((dt1 == 0).sum(axis=0) < nsamples*2 // 100)]
        dt1 = dt1.T
        dt1 = dt1.loc[~(dt1 <= 1).all(axis=1)]
        dt1.to_csv('count_matrix.tsv', sep="\t")

        print(dt1)

        print("----- done counting -----\n")
#        print("----- start processing -----")
#        self.create_count_table(listOfKmersFiles, nsamples-1)

    def get_files_fromdir(self):
        inputDir = self.pathToSamples
        inputDisease = self.disease
        kmerLength = self.kmerLength
        listWithGzFiles = []
        dirList = os.listdir(inputDir)
        nsamples=1
        for file in dirList:
            name = os.path.basename(file)
            listWithGzFiles.append(inputDir + "/" + name)
            nsamples+=1
        print("----- start counting kmers in files-----")
        self.count_kmers(listWithGzFiles,nsamples)

    def concat (self):
        self.kmerMatrix = self.kmerMatrix.rename(columns={"Unnamed: 0" : "kmer"})
        self.kmerMatrix = self.kmerMatrix.set_index(self.kmerMatrix["kmer"])
        del self.kmerMatrix["kmer"]
        print(self.kmerMatrix)
        pathMatrixTrans = self.pathwayMatrix.iloc[[1]].T
        kmerMatrixTrans = self.kmerMatrix.T
        kmerMatrixTrans.insert(0, "Sample", pathMatrixTrans.index, True)
        pathMatrixTrans.insert(0, "Sample", pathMatrixTrans.index, True)
        merged=pathMatrixTrans.merge(kmerMatrixTrans, on='Sample', how="outer")
        merged.to_csv('merged_path_count.tsv', sep="\t")
        merged = merged.set_index(merged["Sample"])
        del merged["Sample"]
        print(merged)
        X, y = merged.iloc[:,1:], merged.iloc[:,0]
        model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        scores = absolute(scores)
        print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))


    def read_pathcoverage(self):
        print("----- start reading path abundance files-----")
        os.system('rm -r '+self.pathabundanceDir+ '/*_temp')
        os.system('rm '+self.pathabundanceDir+ '/*_genefamilies.tsv')
        os.system('rm '+self.pathabundanceDir+ '/*_pathcoverage.tsv')
        dirList = os.listdir(self.pathabundanceDir)
        sample = 0
        count = 1
        new = pd.DataFrame()
        for inputfile in sorted(dirList):
            sample += 1
            ids = []
            func = {}
            name = self.pathabundanceDir + "/" + os.path.basename(inputfile)
            with open(name, "r") as file:
                for line in file:
                    if not "PWY" in line:
                        continue
                    else:
                        splitted = line.split("\t")
                        firstIndex = line.index(":")
                        function = str(splitted[0][firstIndex + 2:]).rstrip()
                        abundance = splitted[1].rstrip()
                        id = splitted[0][:firstIndex]
                        if id not in ids:
                            func[function] = round(float(abundance), 2)
                            ids.append(id)
                pf1 = pd.DataFrame(func.items(), columns=["function", sample-1])
                if sample == 1:
                    new = pd.concat([new, pf1], axis=1)
                else:
                    new = new.merge(pf1, on='function', how="outer").fillna(0)

            print(str(count) + " file done")
            count += 1

        new = new.set_index(new["function"])
        del new["function"]
        new = new.T
        new = new.loc[:, ((new == 0).sum(axis=0) <= (sample) // 2)]
        new = new.T
        print(new)
        self.pathwayMatrix = new
        file.close()
#        self.get_files_fromdir()

    def concatFiles(self):
        dirList = os.listdir(self.pathToSamples)
        names = []
        os.system('mkdir concatFiles_2')
        for file in dirList:
            if file.split('_')[0] not in names:
                names.append(file.split('_')[0])
            else:
                os.system('cat '+self.pathToSamples+'/'+ file.split('_')[0] + '*.fastq.gz > concatFiles_2/' + file.split('_')[0] + '.fastq.gz')

        self.pathToSamples="./concatFiles_2"
        print("----- paired-end fastq files successfully concatenated -----")
        self.get_data_fromdir()

    def get_data_fromdir(self):
        print("----- starting humann-----")
        start_time = datetime.now()
        dirList = os.listdir(self.pathToSamples)
        self.pathabundanceDir="humann_out_2"
        for file in dirList:
            pathToFile = self.pathToSamples+"/"+os.path.basename(file)
            os.system('humann --input '+pathToFile+' --output humann_out_2 --resume --threads 8 --bypass-translated-search')
        end_time = datetime.now()
        print('Duration humann: {}'.format(end_time - start_time))
        self.read_pathcoverage()




if __name__ == "__main__":
    start_time = datetime.now()
    kmer=Kmers("/nfs/home/students/a.miroshina/all_100/",22,"demo")
    kmer.pathToSamples="./concatFiles"
    kmer.pathabundanceDir = "/nfs/home/students/a.miroshina/bachelor/humann_out_last"
    kmer.get_files_fromdir()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
