import mne
import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from scipy.io import loadmat, savemat
from sklearn.pipeline import make_pipeline

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset

SMR_URL = "https://www.nature.com/articles/s41597-021-00883-1/" #"~/mne_data/MNE-SMR-data/database/data-sets/"


def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    """Download the data from one subject"""
    return [dl.data_path(url, "SMR", path, force_update, update_path, verbose)]

class SMR_202104(BaseDataset):
    """
    Dataset used to exemplify the creation of a dataset class in MOABB.
    The data samples have been simulated and has no physiological meaning
    whatsoever.
    """

    def __init__(self):
        super().__init__(
            subjects=[i for i+1 for i in range(62)],
            sessions_per_subject=7, #minimum number of sessions for each subject
            events={"LR": 1, "UD": 2, "2D": 3},
            code="202104",
            interval=[4,8],
            paradigm="imagery",
            doi="",
        )


    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        sessions = load_data(subject=subject, dataset=self.code, verbose=False)
        return sessions

    def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
        """Download the data from one subject"""
        return [dl.data_path(url, "SMR", path, force_update, update_path, verbose)]

@verbose
def load_data(
    subject,
    dataset="202104",
    path=None,
    force_update=False,
    update_path=None,
    base_url=SMR_URL,
    verbose=None,
):  # noqa: D301
    """Get paths to local copies of a SMR dataset files.

    This will fetch data for a given SMR dataset. Report to the bnci website
    for a complete description of the experimental setup of each dataset.

    Parameters
    ----------
    subject : int
        The subject to load.
    dataset : string
        The bnci dataset name.
    path : None | str
        Location of where to look for the SMR data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_SMR_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the SMR dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_SMR_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raws : list
        List of raw instances for each non consecutive recording. Depending
        on the dataset it could be a BCI run or a different recording session.
    event_id: dict
        dictonary containing events and their code.
    """
    dataset_list = {
        "202104": _load_data_202104,
    }

    baseurl_list = {
        "202104": SMR_URL,
    }

    if dataset not in dataset_list.keys():
        raise ValueError(
            "Dataset '%s' is not a valid SMR dataset ID. "
            "Valid dataset are %s." % (dataset, ", ".join(dataset_list.keys()))
        )

    return dataset_list[dataset](
        subject, path, force_update, update_path, baseurl_list[dataset], verbose
    )


@verbose
def _load_data_202104(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=SMR_URL,
    verbose=None,
):
    """Load data for 001-2014 dataset."""
    if (subject < 1) or (subject > 1):
        raise ValueError("Subject must be between 1 and 62. Got %d." % subject)

    # fmt: off
    ch_names = [
        "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
        "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
        "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2", #"CB1", #"CB2"
    ]
    # fmt: on
    ch_types = ["eeg"] * 60
    
    sessions = {}
    for r in range(1,10+1):
        url = "{u}S1_Session_{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)
        runs, ev = _convert_mi(filename[0], ch_names, ch_types)
        # FIXME: deal with run with no event (1:3) and name them
        sessions["session_%s" % r] = {"run_%d" % ii: run for ii, run in enumerate(runs)}
    return sessions

def _convert_mi(filename, ch_names, ch_types):
    """
    Processes (Graz) motor imagery data from MAT files, returns list of
    recording runs.
    """
    runs = []
    event_id = {}
    #data = loadmat(filename, struct_as_record=False, squeeze_me=False)

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    #print(np.array(list(data.keys())))
    #print(data)
    #print(data["BCI"])
    if isinstance(data["BCI"].data, np.ndarray):
        run_array = data["BCI"].data
    else:
        run_array = [data["BCI"].data]

    for i in range(len(run_array)):
        raw, evd = _convert_run(data['BCI'], i, ch_names, ch_types, None)
        if raw is None:
            continue
        runs.append(raw)
        event_id.update(evd)
    # change labels to match rest
    standardize_keys(event_id)
    return runs, event_id

@verbose
def _convert_run(BCI,run_nb, ch_names=None, ch_types=None, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    run = BCI.data[run_nb]
    run = np.concatenate((run[:57],run[58:61]))
    event_id = {}
    n_chan = run.shape[0]
    montage = make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run
    sfreq = BCI.SRATE

    if not ch_names:
        ch_names = ["EEG%d" % ch for ch in range(1, n_chan + 1)]
        montage = None  # no montage

    if not ch_types:
        ch_types = ["eeg"] * n_chan

    trigger = np.zeros((eeg_data.shape[1], 1))
    # some runs does not contains trials i.e baseline runs
    #print(BCI.TrialData[run_nb].trialnumber)
    if len([BCI.TrialData[run_nb].trialnumber]) > 0:
        trigger[BCI.TrialData[run_nb].trialnumber - 1, 0] = BCI.TrialData[run_nb].tasknumber
    else:
        return None, None
    eeg_data_reshaped = np.zeros((eeg_data.shape[1],eeg_data.shape[0]))
    for i in range(eeg_data.shape[1]):
        for j in range(eeg_data.shape[0]):
            eeg_data_reshaped[i][j] = eeg_data[j][i]
    eeg_data = np.c_[eeg_data_reshaped, trigger]
    ch_names = ch_names + ["stim"]
    ch_types = ch_types + ["stim"]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(["LR","UD","2D"])}
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage,match_case=False)
    return raw, event_id

def standardize_keys(d):
    master_list = [
        ["both feet", "feet"],
        ["left hand", "left_hand"],
        ["right hand", "right_hand"],
        ["FEET", "feet"],
        ["HAND", "right_hand"],
        ["NAV", "navigation"],
        ["SUB", "subtraction"],
        ["WORD", "word_ass"],
        ["UD", "UD"],
        ["LR", "LR"],
        ["2D", "2D"],
    ]
    for old, new in master_list:
        if old in d.keys():
            d[new] = d.pop(old)

