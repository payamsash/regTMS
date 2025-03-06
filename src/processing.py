import os
import os.path as op
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne_icalabel import label_components
from mne_connectivity import spectral_connectivity_epochs
from learn_graph import log_degree_barrier


## initialize parameters
compute_power = True
compute_conn = True
compute_graph = True
create_report = True

sfreq = 250.0
(l_freq, h_freq, no_freq) = (0.1, 100.0, 50.0)
montage = mne.channels.make_standard_montage("standard_1020")
verbose = False
fs_dir = fetch_fsaverage(verbose=verbose)
src_fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
bands_dict = {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 13],
            "beta": [13, 30], "gamma": [30, 80]}
snr = 1.0  
lambda2 = 1.0 / snr**2
brain_labels = mne.read_labels_from_annot(subject="fsaverage", parc="aparc", verbose=verbose)[:-1]
bl_names = [bl.name for bl in brain_labels]
power_keys = ["subject_ID", "hemisphere", "protocol", "run", "frequency_band", "brain_label", "power"]

## loop over subjects
folder_name = Path.cwd().parent / "data" / "EEG_data"
for file in tqdm(sorted(os.listdir(folder_name))):
    if file.endswith(".set"):
        fname = folder_name / file
        subject_id = file[:5]
        if file[8:9] == "R": hemisphere = "right"
        if file[8:9] == "L": hemisphere = "left"
        if file[-7:-4] == "pre": run = "pre"
        if file[-8:-4] == "post": run = "post"
        if "0.1Hz" in file: protocol = "0.1 Hz"
        if "1Hz" in file: protocol = "1 Hz"
        if "10Hz" in file: protocol = "10 Hz"
        if "20Hz" in file: protocol = "20 Hz"

        power_dict = {key: [] for key in power_keys}
        con_dict = {}
        graphs_dict = {}

        ## read raw eeg files
        raw = mne.io.read_raw_eeglab(input_fname=fname, preload=True, verbose=verbose)
        raw.annotations.delete(range(len(raw.annotations)))

        ## set montage (FPz -> Fpz)
        raw.set_montage(montage=montage, match_case=False, on_missing="raise")

        ## resample and filter, re-referencing
        raw.resample(sfreq=sfreq, verbose=verbose)
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)
        raw.notch_filter(freqs=no_freq, verbose=verbose) 
        raw.set_eeg_reference("average", projection=False, verbose=verbose)

        ## ICA
        ica = mne.preprocessing.ICA(n_components=0.95, max_iter=800, method='infomax',
                                    fit_params=dict(extended=True))
        try:
            ica.fit(raw, verbose=verbose)
        except:
            ica = mne.preprocessing.ICA(n_components=5, max_iter=800, method='infomax',
                                    fit_params=dict(extended=True))
            ica.fit(raw, verbose=verbose)
        
        ic_dict = label_components(raw, ica, method="iclabel")
        ic_labels = ic_dict["labels"]
        ic_probs = ic_dict["y_pred_proba"]
        eog_indices = []
        for idx, label in enumerate(ic_labels):
            if label == "eye blink" and ic_probs[idx] > 0.70:
                eog_indices.append(idx)
        
        if len(eog_indices) > 0:
            eog_components = ica.plot_properties(raw, picks=eog_indices, verbose=verbose, show=False)
            for fig_idx, fig in enumerate(eog_components):
                pred_value = int(ic_probs[eog_indices[fig_idx]] * 1e2)
                fig.axes[0].set_title(f"pred score: 0.{pred_value}")

        ica.apply(raw, exclude=eog_indices, verbose=verbose)

        ## sensor to source
        # assert len(raw) == 45000, f"the recording duration {len(raw) / sfreq} s should be 180 seconds."
        info = raw.info
        fwd = mne.make_forward_solution(info, trans="fsaverage", src=src_fname, bem=bem, eeg=True,
                                        meg=False, verbose=verbose)
        noise_cov = mne.make_ad_hoc_cov(info, std=None, verbose=verbose)
        inverse_operator = make_inverse_operator(info, fwd, noise_cov, verbose=verbose)
        src = inverse_operator["src"]
        raw.set_eeg_reference("average", projection=True, verbose=verbose)
        
        for band, (freq_l, freq_h) in bands_dict.items(): 
            print(f"working on {file} at frequency range {band}.")
            raw_filt = raw.copy().filter(freq_l, freq_h, verbose=verbose)
            epochs = mne.make_fixed_length_epochs(raw_filt, duration=5.0, preload=True, verbose=verbose)
            
            ## compute power in labels (try to not fill memory)
            if compute_power:
                stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method="dSPM",
                                            label=None, return_generator=True, verbose=verbose)
                label_ts = mne.extract_label_time_course(stcs, brain_labels, src, mode="mean",
                                                        return_generator=False, verbose=verbose)
                powers = np.array(label_ts).mean(axis=(0, 2))
                del stcs, label_ts
                
                ## fill in the dataframe
                for label, power in zip(brain_labels, powers):
                    power_dict["subject_ID"].append(subject_id)
                    power_dict["hemisphere"].append(hemisphere)
                    power_dict["protocol"].append(protocol)
                    power_dict["run"].append(run)
                    power_dict["frequency_band"].append(band)
                    power_dict["brain_label"].append(label.name)
                    power_dict["power"].append(power)
            
            ## compute connectivity between labels (wPLI, Coh)
            if compute_conn:
                stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method="dSPM",
                                            label=None, return_generator=True, verbose=verbose)
                label_ts = mne.extract_label_time_course(stcs, brain_labels, src, mode="mean",
                                                        return_generator=False, verbose=verbose)
                conns = spectral_connectivity_epochs(data=label_ts, names=bl_names,
                                                    method=["wpli", "coh"], sfreq=sfreq,
                                                    fmin=freq_l, fmax=freq_h, faverage=True,
                                                    verbose=verbose)
                del stcs
                con_data = np.array([con.get_data(output="dense")[:, :, 0] for con in conns])
                con_dict[band] = con_data

            ## learn graph on FC data
            if compute_graph:
                label_ts_reshaped = np.array(label_ts).transpose(1, 0, 2)
                X = label_ts_reshaped.reshape(label_ts_reshaped.shape[0], -1)
                graph = log_degree_barrier(X=X, dist_type='sqeuclidean', alpha=1,
                                                beta=1, step=0.5, w0=None, maxit=10000, rtol=1e-16,
                                                retall=False, verbosity='NONE')
                graphs_dict[band] = graph
        
        ## fill and save the report
        if compute_power: 
            fname_save = Path.cwd().parent / "data" / "dataframes" / "powers" / f"{file[:-4]}.csv"
            df_power = pd.DataFrame(power_dict)
            df_power.to_csv(fname_save)

        if compute_conn: 
            fname_save = Path.cwd().parent / "data" / "dataframes" / "conns" / f"{file[:-4]}.pkl"
            with open(fname_save, "wb") as file_pkl:
                pickle.dump(con_dict, file_pkl)

        if compute_graph:
            fname_save = Path.cwd().parent / "data" / "dataframes" / "graphs" / f"{file[:-4]}.pkl"
            with open(fname_save, "wb") as file_pkl:
                pickle.dump(graphs_dict, file_pkl)
        
        if create_report:
            report = mne.Report(title=f"report_subject_{subject_id}", verbose=verbose)
            report.add_raw(raw=raw, title="recording info", butterfly=False, psd=False) 
            if len(eog_indices) > 0:
                report.add_figure(fig=eog_components, title="EOG components", image_format="PNG")
            fname_report = Path.cwd().parent / "data" / "dataframes" / "reports" / f"{file[:-4]}.html"
            report.save(fname=fname_report, open_browser=False, overwrite=True, verbose=verbose)



