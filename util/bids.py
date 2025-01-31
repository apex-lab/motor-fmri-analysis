from mne_bids import BIDSPath
from mne_bids.config import BIDS_VERSION
import os.path as op
import json
import re

class DataSink():
    '''
    Utility for saving processed files to derivative directories.
    '''

    def __init__(self, derivatives_dir, workflow):
        '''
        Inputs
        ------
        derivatives_dir: string, path to main derivatives directory
        workflow: string, name of current workflow
        '''
        self.deriv_root = op.join(derivatives_dir, workflow)
        self.workflow = workflow

    def get_path(self, subject, task, desc, suffix, extension, **kwargs):
        '''
        Builds a directory to save a derivatives file in and
        returns the filepath to save your data at (as string).
        Currently assumes you're saving an EEG file.
        '''
        path = BIDSPath(
            root = self.deriv_root,
            subject = subject,
            task = task,
            extension = extension,
            suffix = suffix,
            check = False,
            **kwargs
            )
        path.mkdir()
        self.check_description()
        fpath = str(path).replace('_%s'%suffix, '_desc-%s'%desc + '_%s'%suffix)
        run_lab = re.findall(r'run-\d+', fpath)
        if run_lab: # format run number as integer
            run_lab = run_lab[0]
            run_num = int(run_lab.split('-')[-1])
            new_run_lab = 'run-%d'%run_num
            fpath = fpath.replace(run_lab, new_run_lab)
        return fpath

    def check_description(self):
        '''
        Checks whether a dataset description file exists in the workflow
        directory, and makes one if not.
        '''
        fpath = op.join(self.deriv_root, 'dataset_description.json')
        try:
            f = open(fpath, "r")
            f.close()
        except:
            desc = {}
            desc['Name'] = self.workflow
            desc['BIDSVersion'] = BIDS_VERSION
            desc['DatasetType'] = 'derivative'
            desc['GeneratedBy'] = [{'Name': self.workflow}]
            with open(fpath, 'w') as f:
                json.dump(desc, f, indent = 4)
