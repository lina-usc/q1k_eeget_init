import mne
import numpy as np
import mne
import mne_bids
import plotly.express as px
import os
import re
import glob

VALID_TASKS = ['rest', 'RS', 'as', 'AS', 'ssvep', 'vp', 'VEP', 'vs', 'ap', 'AEP',
               'go', 'GO', 'plr', 'mn', 'TO', 'nsp', 'fsp', 'PLR']

# define the function for generating the input path and file name
def generate_session_ids (dataset_group, project_path, site_code, task_id_in, subject_id_in,  run_id):
    if dataset_group == "control": 
        session_path_eeg = project_path + 'sourcefiles/' + subject_id_in + '/' + subject_id_in + '_eeg/'
        session_file_name_eeg = glob.glob(session_path_eeg + '*_' + task_id_in + '_*.mff')

    elif dataset_group == "experimental":
        session_path_eeg = project_path + 'sourcedata/eeg/Q1K_' + site_code + '_' + subject_id_in + '/'
        session_file_name_eeg = glob.glob(session_path_eeg + '*' + task_id_in + '_*.mff')
        print(session_path_eeg)
        print(session_file_name_eeg)

        session_path_et = project_path + 'sourcedata/et/Q1K_' + site_code + '_' + subject_id_in + '/'
        session_file_name_et = glob.glob(session_path_et + '*' + task_id_in + '.asc')
        print(session_path_et)
        print(session_file_name_et)

    return session_file_name_eeg, session_file_name_et


def set_din_str (task_id_out):
    # Create the task specific event string list (basically.. visual tasks = ('DIN2','DIN3') and auditory tasks = ('DIN4','DIN5'))
    if task_id_out == 'AEP':
        event_dict_offset = 1
        din_str = ('DIN4','DIN5')
    if task_id_out == 'AS':
        event_dict_offset = 1
        din_str = ('DIN2','DIN2')
    if task_id_out == 'GO':
        event_dict_offset = 1
        din_str = ('DIN2','DIN3')
    if task_id_out == 'TO':
        event_dict_offset = 1
        din_str = ('DIN4','DIN5')
    if task_id_out == 'VEP':
        event_dict_offset = 1
        din_str = ('DIN2','DIN3')
    if task_id_out == 'PLR':
        event_dict_offset = 1
        din_str = ('DIN2','DIN3')
    if task_id_out == 'RS':
        event_dict_offset = 1
        din_str = ('DIN2','DIN3')
        
    return din_str, event_dict_offset


def get_din_diff(events, event_dict, din_str):
    # get the distance between DIN events of interest
    din_diffs = []
    din_diffs_time = []
    last_din = 0
    # Iterate over the array
    for row in events:
        # Check if the third column corresponds to e1 or e2 values|
        if row[2] == event_dict[din_str[0]] or row[2] == event_dict[din_str[0]]:
            # Calculate the sequential difference between values in the first column
            if last_din > 0:
                din_diffs.append(row[0] - last_din)
                din_diffs_time.append(row[0])
                last_din = row[0]
            else:
                last_din = row[0]
                #print(last_din)

    return din_diffs, din_diffs_time          
                
                
def din_check(event_dict, din_str):
    # this is a patch that will become obsolete... accounting for missing DIN types in the recording..
    exists_in_dict = [din in event_dict for din in din_str]
    print(din_str)
    print(exists_in_dict)
    if all(exists_in_dict):
        print("Both strings exist in eeg_event_dict.")
    elif any(exists_in_dict):
        existing_string = din_str[exists_in_dict.index(True)]
        din_str = (existing_string, existing_string)
        print(f"Only one string exists. din_str updated to: {din_str}")
    else:
        din_str=()
        print("Neither DIN string exists in eeg_event_dict.")
    print(din_str)

    return din_str


def et_read(path,blink_interp,fill_nans,resamp):

    #read the asc eye tracking data and convert it to a dataframe...
    et_raw = mne.io.read_raw_eyelink(path)
    et_raw.load_data()
    
    #get the events from the annotation structure
    et_annot_events, et_annot_event_dict = mne.events_from_annotations(et_raw)
    #et_events = mne.find_events(et_raw, min_duration=0.01, shortest_event=1, uint_cast=True)

    if blink_interp:
        #read the raw et asc file again this time with the blinks annotation enabled.. (this should be combined into a single read) 
        print("Interpolating blinks.")
        #et_raw = mne.io.read_raw_eyelink(path,create_annotations=["blinks"])
        et_raw = mne.io.read_raw_eyelink(path,create_annotations=["blinks"])
        et_raw.load_data()

        #interpolate the signals during blinks
        #mne.preprocessing.eyetracking.interpolate_blinks(et_raw, match=('BAD_blink','BAD_ACQ_SKIP'),buffer=(0.05, 0.2), interpolate_gaze=True)
        mne.preprocessing.eyetracking.interpolate_blinks(et_raw,buffer=(0.05, 0.2), interpolate_gaze=True)
    
    if fill_nans:
        print("Filling NaNs with zeros.")
        #replace Nans with 0s..
        data = et_raw.get_data()
        data[np.isnan(data)] = 0
        et_raw._data = data
    
    if resamp:
        print("Resampling the data.")
        #resample the data..
        et_raw.resample(1000, npad="auto")

    #create the dataframe
    et_raw_df = et_raw.to_data_frame()
   

    return et_raw, et_raw_df, et_annot_events, et_annot_event_dict


def get_event_dict(raw, events, offset):

    stim_names = raw.copy().pick('stim').info['ch_names']
    event_dict = {event: int(i) + offset
                  for i, event in enumerate(stim_names)
                  if event != 'STI 014'}

    """ #method for building the event_dict is precarious, but it seems to satisfy all of the cases... to be reworked later
    if raw.info.ch_names[-1] == 'VBeg':
        print('VBeg method')
        adjuster = len(raw.info.ch_names) - 129 - events[0,2] # this assumes that 'VBeg' is the last stim channel and the first event in the recording.
        event_dict = {}
        for i in range(129,len(raw.info.ch_names)):
            event_dict[raw.info.ch_names[i]] = i-129 + 1 - adjuster

    if raw.info.ch_names[-1] == 'STI 014':
        print('STI 014 method')
        #adjuster = len(raw.info.ch_names) - 130 - events[1,2] # this assumes that 'STI 014' is the last stim channel and the first event in the recording.
        event_dict = {}
       for i in range(129,len(raw.info.ch_names)):
            event_dict[raw.info.ch_names[i]] = i-129 + 1# - adjuster

    # check for dstr label and if found remove it
    try:
        event_dict['dstr']
        print('found dstr label.. removing it')
        del event_dict['dstr']
        for k in event_dict:
            event_dict[k] -= 1
    except:
        print('no dstr label found.. continuing')"""

    return event_dict            


def eeg_clean_events(eeg_events, eeg_event_dict, din_str):
    
    # remove TSYN events...this might have to happen for all tasks.. because this is not used for anything and they appear in arbitrary locations...
    print('Removing TSYN events...')
    mask = np.isin(eeg_events[:,2],[eeg_event_dict['TSYN']])
    eeg_events = eeg_events[~mask]
    #new_events = np.empty((0, 3))
    if 'TSYN' in eeg_event_dict:
        del eeg_event_dict['TSYN']

    #remove 'DIN*' events that are not included in din_str    
    #filter the dictionary
    filtered_dict = {k: v for k, v in eeg_event_dict.items() if not k.startswith('DIN') or k in din_str}
    #reassign dict values sequentially
    filtered_dict = {key: i + 1 for i, (key, _) in enumerate(filtered_dict.items())}
    #update the events array
    updated_events = np.array([
        [row[0], row[1], filtered_dict[key]]
        for row in eeg_events if (key := next((k for k, v in eeg_event_dict.items() if v == row[2]), None)) in filtered_dict
    ])

    #update eeg_event_dict and eeg_events
    eeg_event_dict = filtered_dict
    eeg_events = updated_events

    # print result
    print("Updated Dictionary:", eeg_event_dict)
        
    return eeg_events, eeg_event_dict



def eeg_task_events(eeg_events, eeg_event_dict, din_str, task_name=None):
    
    din_offset = []
    new_events = np.empty((0, 3))
    
    if not task_name:
        raise ValueError(f'please pass one of {VALID_TASKS}'
                         ' to the task_name keyword argument.')

    ## handle visual DIN tasks
    #if task_name == 'PLR' or task_name == 'VEP':
        
    if task_name == 'PLR':
        #find the first din_str event following plro events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e == eeg_event_dict['plro']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        #append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()] 
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['plro_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['plro_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])


    elif task_name == 'VEP':

        # find the first DIN3 event following either sv06 or sv15 events and add new *d events
        # DIN3 is used here eventhough it is the second DIN in the visual stimulus inversion animations... 
        # DIN3 is used because because it is more reliable that DIN2 at the HSJ site and its offset from the intial DIN2 event is exactly determined by stimulus condition
        # This is handled differently it the et_task_event function.. but because the stimulus sequence is exactly deterministic these always line up in practice
        # I would continue to keep an eye on this...
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e == eeg_event_dict['sv06']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]] and din_str[1] == 'DIN3':
                            cor_val = 166
                        else:
                            cor_val = 0
                        new_row = np.array([[eeg_events[i[0] + 1, 0] - cor_val, 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e == eeg_event_dict['sv15']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]] and din_str[1] == 'DIN3':
                            cor_val = 66
                        else:
                            cor_val = 0
                        new_row = np.array([[eeg_events[i[0] + 1, 0] - cor_val, 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()] 
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['sv06_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['sv15_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['sv06_d'],eeg_event_dict['sv15_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])

        
        
        
    elif task_name == 'ap' or task_name == 'AEP':

        # remove TSYN events...this might have to happen for all tasks.. because this is not used for anything and they appear in arbitrary locations...
        print('Removing TSYN events...')
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['TSYN']])
        eeg_events = eeg_events[~mask]
        new_events = np.empty((0, 3))

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e == eeg_event_dict['ae06']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN4']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e == eeg_event_dict['ae40']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN4']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()] 
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['ae06_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['ae40_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['ae06_d'],eeg_event_dict['ae40_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])


    #elif task_name == 'go':

    elif task_name=='go'or task_name == 'GO':

        # remove TSYN events...this might have to happen for all tasks.. because this is not used for anything and they appear in arbitrary locations...
        print('Removing TSYN events...')
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['TSYN']])
        eeg_events = eeg_events[~mask]
        new_events = np.empty((0, 3))

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e == eeg_event_dict['dtoc']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e == eeg_event_dict['dtbc']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

            if e == eeg_event_dict['dtgc']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])


            if e == eeg_event_dict['dsoc']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e == eeg_event_dict['dsbc']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

            if e == eeg_event_dict['dsgc']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()] 
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['dtoc_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['dtbc_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['dtgc_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['dfoc_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['dfbc_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['dfgc_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['dtoc_d'],eeg_event_dict['dtbc_d'],eeg_event_dict['dtgc_d'],eeg_event_dict['dfoc_d'],eeg_event_dict['dfbc_d'],eeg_event_dict['dfgc_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])


    elif task_name == 'as' or task_name == 'AS':
        
        # remove TSYN events...this might have to happen for all tasks.. because this is not used for anything and they appear in arbitrary locations...
        print('Removing TSYN events...')
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['TSYN']])
        eeg_events = eeg_events[~mask]
        new_events = np.empty((0, 3))

        d_ind = [value for key, value in eeg_event_dict.items() if key.startswith('dd')]
        t_ind = [value for key, value in eeg_event_dict.items() if key.startswith('dt')]

        # find the first DIN3 or DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e in d_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:# or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e in t_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:# or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()] 
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['dd_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['dt_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['dd_d'],eeg_event_dict['dt_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])
        

    elif task_name=='mn' or task_name=='TO':
        
        # remove TSYN events...this might have to happen for all tasks.. because this is not used for anything and they appear in arbitrary locations...
        print('Removing TSYN events...')
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['TSYN']])
        eeg_events = eeg_events[~mask]
        new_events = np.empty((0, 3))
        
        s_ind = [value for key, value in eeg_event_dict.items() if key.startswith('SO')]
        t_ind = [value for key, value in eeg_event_dict.items() if key.startswith('Dev')]

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e in s_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN4']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e in t_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN4']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
                        #eeg_events[i[0]+1, 2] = len(eeg_event_dict) + 2 #mmnt DIN onset
                        #new_events.append([eeg_events[i[0], 0], 0 , len(eeg_event_dict) + 2])
                        #new_events = np.append(new_events,[eeg_events[i[0], 0], 0, len(eeg_event_dict) + 2], axis=0)
                        #din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()] 
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['to_s_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['to_t_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['to_s_d'],eeg_event_dict['to_t_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])

        
        ##for the plr task it is more simple to select trials based on DIN2 occurences
        #mask = np.isin(eeg_events[:,2],[eeg_event_dict['DIN4']])
        #eeg_stims = eeg_events[mask]
        #print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        ##calculate the inter trial interval between stimulus onset DIN events
        #eeg_iti = np.diff(eeg_stims[:,0])


    elif task_name=='rest' or task_name=='RS':

        # remove TSYN events...this might have to happen for all tasks.. because this is not used for anything and they appear in arbitrary locations...
        print('Removing TSYN events...')
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['TSYN']])
        eeg_events = eeg_events[~mask]
        new_events = np.empty((0, 3))
        
        v_ind = [value for key, value in eeg_event_dict.items() if key.startswith('vs')]
        b_ind = [value for key, value in eeg_event_dict.items() if key.startswith('dbrk')]

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e in v_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN2']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e in b_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN2']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
                        #eeg_events[i[0]+1, 2] = len(eeg_event_dict) + 2 #mmnt DIN onset
                        #new_events.append([eeg_events[i[0], 0], 0 , len(eeg_event_dict) + 2])
                        #new_events = np.append(new_events,[eeg_events[i[0], 0], 0, len(eeg_event_dict) + 2], axis=0)
                        #din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()] 
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['vs_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['brk_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['vs_d'],eeg_event_dict['brk_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])

        
    elif task_name in ['vs', 'fsp', 'nsp']:
        raise NotImplemented
    else:
        raise ValueError('Could not determine task name.'
                         f' Expected one of {VALID_TASKS} but got {task_name}')


    #rename DIN* events to eeg_DIN*
    renamed_dict = {
        (f"eeg_{key}" if key.startswith('DIN') else key): value
        for key, value in eeg_event_dict.items()
    }
    eeg_event_dict = renamed_dict

    # print result
    print("Updated Dictionary:", eeg_event_dict)

    return eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events



def show_sync_offsets(eeg_stims,et_stims):
    #eeg_et_offset = eeg_stims[:,0] - et_stims['index'][:]
    eeg_et_offset = eeg_stims[:,0] - et_stims[:,0]

    fig = px.scatter(y=eeg_et_offset)
    fig.show()
    
    return fig



#def eeg_clean_events(eeg_annot_event_dict, eeg_annot_events, task_id_out):

#    if task_id_out == "PLR"   
    
#    return eeg_annot_event_dict, eeg_annot_events
    
    
    
def et_clean_events(et_annot_event_dict, et_annot_events):

    #remove dictionary keys that start with "TRACKER_TIME"
    filtered_dict = {k: v for k, v in et_annot_event_dict.items() if not k.startswith("TRACKER_TIME")}
    #update dictionary values to be consecutive
    updated_dict = {k: i + 1 for i, (k, _) in enumerate(filtered_dict.items())}
    #remove rows from the array corresponding to removed keys
    valid_values = set(filtered_dict.values())
    filtered_events = np.array([row for row in et_annot_events if row[2] in valid_values])
    #update array values to match the updated dictionary
    value_map = {v: updated_dict[k] for k, v in filtered_dict.items()}
    updated_events = np.array([[row[0], row[1], value_map[row[2]]] for row in filtered_events])
    # Results
    et_annot_event_dict = updated_dict
    et_annot_events = updated_events

    
    #clean keys by removing numeric prefixes
    cleaned_dict = {}
    index_map = {}
    for key, value in et_annot_event_dict.items():
        #remove leading numbers and dashes/spaces using regex
        clean_key = re.sub(r'^[-\d\s]+', '', key)  # Remove leading digits, dashes, and spaces
        if clean_key not in cleaned_dict:
            cleaned_dict[clean_key] = value
        else:
            cleaned_dict[clean_key] = min(cleaned_dict[clean_key], value)  # Keep smallest index
        index_map[value] = clean_key  # Map original indices to cleaned keys
    #create a new dictionary with cleaned keys and assign unique indices
    new_dict = {key: idx + 1 for idx, key in enumerate(sorted(cleaned_dict.keys()))}
    #build a full mapping of old indices to new indices
    old_to_new_mapping = {old_idx: new_dict[cleaned_key] for old_idx, cleaned_key in index_map.items()}
    #update et_annot_events
    updated_column = [old_to_new_mapping.get(value, -1) for value in et_annot_events[:, 2]]
    #check for unmatched values
    if -1 in updated_column:
        unmatched = [value for value in et_annot_events[:, 2] if value not in old_to_new_mapping]
        raise ValueError(f"Unmatched values in et_annot_events[:, 2]: {unmatched}")
    #apply the updates
    et_annot_events[:, 2] = updated_column
    # Final results
    et_annot_event_dict = new_dict
    
    return et_annot_event_dict, et_annot_events


def et_task_events(et_raw_df, et_annot_event_dict, et_annot_events, task_id):
    # fill NaNs in DIN channel with zeros
    et_raw_df['DIN']=et_raw_df['DIN'].fillna(0)

    # Correct blips to zero for a single sample while DIN8 is on.
    for ind, row in et_raw_df.iterrows():
        if ind < len(et_raw_df)-1:
            if ind > 0:
                if et_raw_df['DIN'][ind] == 0:
                    if et_raw_df['DIN'][ind-1] == 8:
                        if et_raw_df['DIN'][ind+1] == 8:
                            et_raw_df['DIN'].loc[ind] = 8

    # convert the ET DIN channel into ET events
    # find when the DIN channel changes values
    et_raw_df['DIN_diff']=et_raw_df['DIN'].diff()
    # select all non-zero DIN changes
    et_din_events=et_raw_df.loc[et_raw_df['DIN_diff']>0]

    if task_id == 'VEP' or task_id == 'PLR':
        # there should only be DIN 2 and 4 in the Q1K visual tasks.. however there are frequently binary values greater than 4 indicating that there are anomalous pin4 and pin5 pulses
        # bin2=pin2, bin4=pin3, bin8=pin4, bin16=pin5, bin18=pin2+pin5, bin20=pin3+pin5, bin24=pin4+pin5, bin26=pin2+pin4+pin5, bin28=pin3+pin4+pin5
        # given these anomalous pin4 and pin5 pulses the conversion at pin change time is: binary 2,18,26 = 2, and binary 4,20,28 = 4

        # perform the anomalous DIN conversion
        et_din_events = et_din_events.copy()
        et_din_events['DIN'].loc[et_din_events['DIN'].isin([2,18,26])] = 2
        et_din_events['DIN'].loc[et_din_events['DIN'].isin([4,20,28])] = 4

        et_din_events = et_din_events.copy()
        et_din_events=et_din_events.loc[et_raw_df['DIN'].isin([2,4])]
        et_din_events = et_din_events.reset_index()
        et_din_events['DIN_diff'] = et_din_events['DIN_diff'].astype(int)
        et_din_events    
    
    
    #convert DIN_diff to integers
    et_din_events['DIN_diff'] = et_din_events['DIN_diff'].astype(int)

    #add DIN events to et_annot_event_dict with the next available small integer
    existing_indices = set(et_annot_event_dict.values())
    next_index = max(existing_indices) + 1

    for din_diff in et_din_events['DIN_diff']:
        din_key = f'DIN{din_diff}'
        if din_key not in et_annot_event_dict:
            et_annot_event_dict[din_key] = next_index
            next_index += 1

    #create new rows for et_annot_events based on et_din_events
    #map DIN_diff to the new dictionary indices
    et_din_events['mapped_value'] = et_din_events['DIN_diff'].map(lambda x: et_annot_event_dict[f'DIN{x}'])

    #add new rows to et_annot_events
    new_events = np.array([[row['index'], 0, row['mapped_value']] for _, row in et_din_events.iterrows()])
    et_annot_events = np.vstack((et_annot_events, new_events))

    #sort the updated et_annot_events array by the first column (timestamps)
    et_annot_events = et_annot_events[np.argsort(et_annot_events[:, 0])]
    et_annot_events = et_annot_events.astype(int)


    if task_id == 'VEP':
        target_values = {et_annot_event_dict['STIM'], et_annot_event_dict['CS_SPIN']}
        #initialize results and tracking for pruning
        result_events = []
        pruned_indices = set()
        #iterate through rows and apply pruning for 'STIM' and 'CS_SPIN'
        for i, row in enumerate(et_annot_events):
            if i in pruned_indices:
                continue  #skip rows already excluded
            if row[2] in target_values:
                #add the first occurrence of 'STIM' or 'CS_SPIN'
                result_events.append(row)
                #exclude rows of the same type within +500 range
                pruned_indices.update(
                    j for j, other_row in enumerate(et_annot_events)
                    #if abs(other_row[0] - row[0]) <= 500 and other_row[2] == row[2]
                    if other_row[0] - row[0] <= 1000 and other_row[2] == row[2]
                )
            else:
                #retain rows unrelated to 'STIM' or 'CS_SPIN'
                result_events.append(row)
        #convert results back to a numpy array
        result_events = np.array(result_events)
        et_annot_events=result_events

        # add a new key for 'STIM_d' in the dictionary
        stim_d_value = max(et_annot_event_dict.values()) + 1
        et_annot_event_dict['STIM_d'] = stim_d_value

        #process rows to handle 'DIN2' and 'DIN4' for each 'STIM'
        new_rows = []
        used_indices = set()  # To ensure only the first 'DIN2' or 'DIN4' is used

        for stim_index, stim_row in enumerate(et_annot_events):
            if stim_row[2] == et_annot_event_dict['STIM']:
                stim_time = stim_row[0]  # First column of the 'STIM' row
                stim_d_time = None

                # Look for the first 'DIN2' within 1000 ms after this 'STIM'
                for i in range(stim_index + 1, len(et_annot_events)):
                    din2_row = et_annot_events[i]
                    if 'DIN2' in et_annot_event_dict:
                        if (
                            din2_row[2] == et_annot_event_dict['DIN2'] and
                            i not in used_indices and
                            0 <= din2_row[0] - stim_time <= 1000
                        ):
                            stim_d_time = din2_row[0]  # Use 'DIN2' time directly
                            new_rows.append([stim_d_time, 0, stim_d_value])
                            used_indices.add(i)
                            break

                # If no 'DIN2' is found, look for the first 'DIN4' and calculate midpoint if necessary
                if stim_d_time is None:
                    first_din4_time = None
                    second_din4_time = None

                    for i in range(stim_index + 1, len(et_annot_events)):
                        din4_row = et_annot_events[i]
                        if (
                            din4_row[2] == et_annot_event_dict['DIN4'] and
                            i not in used_indices and
                            0 <= din4_row[0] - stim_time <= 1000
                        ):
                            if first_din4_time is None:
                                first_din4_time = din4_row[0]
                                used_indices.add(i)
                            elif second_din4_time is None:
                                second_din4_time = din4_row[0]
                                break

                    # If two DIN4s are found, calculate the midpoint
                    if first_din4_time is not None and second_din4_time is not None:
                        stim_d_time = first_din4_time - (second_din4_time - first_din4_time) // 2
                        new_rows.append([stim_d_time, 0, stim_d_value])

        #add the new rows to the existing events
        et_annot_events = np.vstack([et_annot_events, new_rows])

        #sort the array by the first column for clarity (optional)
        et_annot_events = et_annot_events[et_annot_events[:, 0].argsort()]

    if task_id == 'PLR':
        target_values = {et_annot_event_dict['STIM'], et_annot_event_dict['FIX'], et_annot_event_dict['ISI']}
        #initialize results and tracking for pruning
        result_events = []
        pruned_indices = set()
        #iterate through rows and apply pruning for target)values
        for i, row in enumerate(et_annot_events):
            if i in pruned_indices:
                continue  #skip rows already excluded
            if row[2] in target_values:
                #add the first occurrence of target_values
                result_events.append(row)
                #exclude rows of the same type within +500 range
                pruned_indices.update(
                    j for j, other_row in enumerate(et_annot_events)
                    #if abs(other_row[0] - row[0]) <= 500 and other_row[2] == row[2]
                    if other_row[0] - row[0] <= 1000 and other_row[2] == row[2]
                )
            else:
                #retain rows unrelated to target_values
                result_events.append(row)
        #convert results back to a numpy array
        result_events = np.array(result_events)
        et_annot_events=result_events

        # add a new key for 'STIM_d' in the dictionary
        stim_d_value = max(et_annot_event_dict.values()) + 1
        et_annot_event_dict['STIM_d'] = stim_d_value

        #process rows to handle 'DIN2' and 'DIN4' for each 'STIM'
        new_rows = []
        used_indices = set()  # To ensure only the first 'DIN2' or 'DIN4' is used

        for stim_index, stim_row in enumerate(et_annot_events):
            if stim_row[2] == et_annot_event_dict['STIM']:
                stim_time = stim_row[0]  # First column of the 'STIM' row
                stim_d_time = None

                # Look for the first 'DIN2' within 1000 ms after this 'STIM'
                if 'DIN2' in et_annot_event_dict:
                    for i in range(stim_index + 1, len(et_annot_events)):
                        din2_row = et_annot_events[i]
                        if (
                            din2_row[2] == et_annot_event_dict['DIN2'] and
                            i not in used_indices and
                            0 <= din2_row[0] - stim_time <= 1000
                        ):
                            stim_d_time = din2_row[0]  # Use 'DIN2' time directly
                            new_rows.append([stim_d_time, 0, stim_d_value])
                            used_indices.add(i)
                            break

                # If no 'DIN2' is found, look for the first 'DIN4' and calculate midpoint if necessary
                if stim_d_time is None:
                    if 'DIN4' in et_annot_event_dict:
                        first_din4_time = None
                        second_din4_time = None

                        for i in range(stim_index + 1, len(et_annot_events)):
                            din4_row = et_annot_events[i]
                            if (
                                din4_row[2] == et_annot_event_dict['DIN4'] and
                                i not in used_indices and
                                0 <= din4_row[0] - stim_time <= 1000
                            ):
                                stim_d_time = din4_row[0]  # Use 'DIN2' time directly
                                new_rows.append([stim_d_time, 0, stim_d_value])
                                used_indices.add(i)
                                break

        #add the new rows to the existing events
        et_annot_events = np.vstack([et_annot_events, new_rows])

        #sort the array by the first column for clarity (optional)
        et_annot_events = et_annot_events[et_annot_events[:, 0].argsort()]


    #rename DIN* events to eeg_DIN*
    renamed_dict = {
        (f"et_{key}" if key.startswith('DIN') else key): value
        for key, value in et_annot_event_dict.items()
    }
    et_annot_event_dict = renamed_dict

    # print result
    print("Updated Dictionary:", et_annot_event_dict)
    
    return et_annot_event_dict, et_annot_events, et_raw_df
    
    
def eeg_et_align(eeg_event_dict, et_event_dict, eeg_events, et_events, eeg_stims, et_stims, eeg_sfreq, et_sfreq):
    eeg_times = eeg_stims[:, 0] / eeg_sfreq
    et_times = et_stims[:, 0] / et_sfreq

    n_eeg_times = len(eeg_times)
    n_et_times = len(et_times)
    
    if n_eeg_times > n_et_times:
        print("there are more eeg_times and there are et_times.. attempting align")
        eeg_times = times_align(eeg_times,et_times)
    elif n_eeg_times < n_et_times:
        print("there are more et_times and there are eeg_times.. attempting align")
        et_times = times_align(et_times,eeg_times)
    else:
        print("there are the same number of eeg_times and et_times.. continuing")
        
    #check if alignment was successfull..
    n_eeg_times = len(eeg_times)
    n_et_times = len(et_times)
    if n_eeg_times != n_et_times:
        print("EEG and ET times alignment was not successful... abandoning sync procedures...")
        et_syn = False
    else:
        #create the sync_time events for the EEG and ET data.
        #convert times to samples..
        #eeg_samps = eeg_stims[:, 0]
        #et_samps = et_stims[:, 0]
        eeg_samps = eeg_times * eeg_sfreq        
        et_samps = et_times * et_sfreq                
        #add "*_sync_time" to the dictionary
        eeg_event_dict['eeg_sync_time'] = len(eeg_event_dict) + 1
        et_event_dict['et_sync_time'] = len(et_event_dict) + 1
        #add rows to the events array for "*_sync_time"
        eeg_sync_time_rows = [[samp, 0, eeg_event_dict['eeg_sync_time']] for samp in eeg_samps]
        et_sync_time_rows = [[samp, 0, et_event_dict['et_sync_time']] for samp in et_samps]
        #combine the new rows with the existing events
        eeg_events = np.vstack([eeg_events, eeg_sync_time_rows])
        eeg_events = eeg_events[eeg_events[:, 0].argsort()]  # Sort by the first column (time)
        et_events = np.vstack([et_events, et_sync_time_rows])
        et_events = et_events[et_events[:, 0].argsort()]  # Sort by the first column (time)

        print("Updated EEG event dictionary:")
        eeg_event_dict
        
    return eeg_event_dict, et_event_dict, eeg_events, et_events, eeg_times, et_times

    
    
    
    
    
def eeg_et_combine(eeg_raw, et_raw, eeg_times, et_times, eeg_events, eeg_event_dict, et_events, et_event_dict):

    eeg_raw.load_data()
    et_raw.load_data()

    # Align the data
    mne.preprocessing.realign_raw(eeg_raw, et_raw, eeg_times, et_times, verbose="error")

    # Add EEG channels to the eye-tracking raw object
    eeg_raw.add_channels([et_raw], force_update_info=True)

    # update the annotations...
    eeg_event_dict_r = {value: key for key, value in eeg_event_dict.items()}
    eeg_annots = mne.annotations_from_events(
        events=eeg_events,
        event_desc=eeg_event_dict_r,
        sfreq=eeg_raw.info["sfreq"],
        orig_time=eeg_raw.info["meas_date"],
    )
    eeg_raw.set_annotations(eeg_annots)

    return eeg_raw



def times_align(a_times, b_times):

    b_intervals = np.diff(b_times)

    # Function to find the best index to remove
    def find_best_index_to_remove(a_times, target_intervals):
        min_deviation = float('inf')
        best_index = None

        for i in range(1, len(a_times) - 1):  # Skip first and last indices
            modified_a_times = np.delete(a_times, i)  # Remove element at index i
            modified_intervals = np.diff(modified_a_times)  # Compute new intervals
            deviation = np.sum(np.abs(modified_intervals - target_intervals))  # Deviation from target
            if deviation < min_deviation:
                min_deviation = deviation
                best_index = i  # Store the index with the smallest deviation

        return best_index

    # Find the best index to remove
    index_to_remove = find_best_index_to_remove(a_times, b_intervals)
    print("Index to remove:", index_to_remove)

    a_times = np.delete(a_times, index_to_remove)
    return a_times
    

    
def write_eeg(eeg_raw, eeg_event_dict, eeg_events, subject_id_out, session_id, task_id_out, project_path, device_info):
    # write the BIDS output files
    # specify power line frequency as required by BIDS
    eeg_raw.info["line_freq"] = 60
    eeg_raw.info['device_info']=device_info
    eeg_raw.info['device_info']['type'] = eeg_raw.info['device_info']['type'].replace(' ', '-')

    #THIS SHOULD BE MOVED TO QIT.FILLNA if it is needed...
    def fillna(raw, fill_val=0):
        return mne.io.RawArray(np.nan_to_num(raw.get_data(), nan=fill_val), raw.info)
    eeg_raw=fillna(eeg_raw,fill_val=0)

    eeg_bids_path = mne_bids.BIDSPath(
        subject=subject_id_out, session=session_id, task=task_id_out, run="1", datatype="eeg", root=project_path
    )

    print(eeg_bids_path)
    mne_bids.write_raw_bids(
        raw=eeg_raw,
        bids_path=eeg_bids_path,
        events=eeg_events,
        event_id=eeg_event_dict,
        format = "EDF",
        overwrite=True,
        allow_preload=True,
    )
    
    return eeg_bids_path


def et_events_to_annot(et_raw, et_event_dict, et_events):
    
    et_event_dict_r = {value: key for key, value in et_event_dict.items()}
    event_annotations = mne.annotations_from_events(
        events=et_events,
        event_desc=et_event_dict_r,
        sfreq=et_raw.info['sfreq'],
        orig_time=et_raw.info['meas_date']
    )

    #filter existing annotations
    existing_annotations = et_raw.annotations
    selected_annotations = mne.Annotations(
        onset=[
            existing_annotations.onset[i]
            for i, desc in enumerate(existing_annotations.description)
            if desc in ['BAD_blink', 'BAD_ACQ_SKIP']
        ],
        duration=[
            existing_annotations.duration[i]
            for i, desc in enumerate(existing_annotations.description)
            if desc in ['BAD_blink', 'BAD_ACQ_SKIP']
        ],
        description=[
            desc for desc in existing_annotations.description if desc in ['BAD_blink', 'BAD_ACQ_SKIP']
        ],
        orig_time=existing_annotations.orig_time
    )

    #combine both annotation sets
    combined_annotations = event_annotations + selected_annotations    #et_event_dict_r = {value: key for key, value in et_event_dict.items()}

    
    # Get all channel types and names
    channel_types = et_raw.get_channel_types()
    channel_names = et_raw.ch_names

    # Find channels with types 'eyegaze' or 'pupil'
    eyegaze_channels = [name for name, ch_type in zip(channel_names, channel_types) if ch_type == 'eyegaze']
    pupil_channels = [name for name, ch_type in zip(channel_names, channel_types) if ch_type == 'pupil']

    # Combine the lists if needed
    eye_channels = tuple(eyegaze_channels + pupil_channels)

    print("Eyegaze channels:", eyegaze_channels)
    print("Pupil channels:", pupil_channels)
    print("All eye channels:", eye_channels)

    #eye_chans = ('xpos_left', 'ypos_left', 'pupil_left')        #event_annotations = mne.annotations_from_events(

    combined_annotations.ch_names = np.array([
        eye_channels if desc in ['fixation', 'saccade', 'BAD_blink', 'BAD_ACQ_SKIP'] else ()
        for desc in combined_annotations.description
    ], dtype=object)

    #replace raw's annotations
    et_raw.set_annotations(combined_annotations)    

    return et_raw



def write_et(et_raw, eeg_bids_path):

    
    et_out_path = str(eeg_bids_path)
    et_out_path = et_out_path.replace("/eeg/", "/et/")
    et_out_path = et_out_path.replace("_eeg.edf", "_et.fif")
    #et_out_path = et_out_path.split(".")[0] + ".fif"
    # Extract the directory path
    directory = os.path.dirname(et_out_path)

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    #write the updated Raw object to the specified path
    et_raw.save(et_out_path, overwrite=True)

    return et_out_path




''' this should be obsolete... replaced with et_clean_events and et_task_events...
def et_event_test(et_raw_df, task_name=''):

    # fill NaNs in DIN channel with zeros
    et_raw_df['DIN']=et_raw_df['DIN'].fillna(0)

    # Correct blips to zero for a single sample while DIN8 is on.
    for ind, row in et_raw_df.iterrows():
        if ind < len(et_raw_df)-1:
            if ind > 0:
                if et_raw_df['DIN'][ind] == 0:
                    if et_raw_df['DIN'][ind-1] == 8:
                        if et_raw_df['DIN'][ind+1] == 8:
                            et_raw_df['DIN'].loc[ind] = 8

    # convert the ET DIN channel into ET events
    # find when the DIN channel changes values
    et_raw_df['DIN_diff']=et_raw_df['DIN'].diff()
    # select all non-zero DIN changes
    et_events=et_raw_df.loc[et_raw_df['DIN_diff']>0]

    # there should only be DIN 2 and 4 in the Q1K visual tasks.. however there are frequently binary values greater than 4 indicating that there are anomalous pin4 and pin5 pulses
    # bin2=pin2, bin4=pin3, bin8=pin4, bin16=pin5, bin18=pin2+pin5, bin20=pin3+pin5, bin24=pin4+pin5, bin26=pin2+pin4+pin5, bin28=pin3+pin4+pin5
    # given these anomalous pin4 and pin5 pulses the conversion at pin change time is: binary 2,18,26 = 2, and binary 4,20,28 = 4

    # perform the anomalous DIN conversion
    et_events = et_events.copy()
    et_events['DIN'].loc[et_events['DIN'].isin([2,18,26])] = 2
    et_events['DIN'].loc[et_events['DIN'].isin([4,20,28])] = 4


    if task_name == 'vp' or task_name == 'VEP':

        #select only the DIN 2 and 4 rows.. and reset the index
        et_events = et_events.copy()
        et_events=et_events.loc[et_raw_df['DIN'].isin([2,4])]
        et_events = et_events.reset_index()
    
        # Search for a DIN4 (fixation) followed by a DIN2 (stimulus) within 180 to 3000ms.
        for ind, row in et_events.iterrows():
            if et_events['DIN'][ind] == 4:
                if ind < len(et_events)-1:
                    if et_events['DIN'][ind+1] == 2:
                        if et_events['index'][ind+1] - et_events['index'][ind] > 180:
                            if et_events['index'][ind+1] - et_events['index'][ind] < 3000:
                                et_events['DIN_diff'][ind+1] = 5
                    if et_events['DIN'][ind+1] == 4:
                        if et_events['index'][ind+1] - et_events['index'][ind] > 180:
                            if et_events['index'][ind+1] - et_events['index'][ind] < 3000:
                                et_events['DIN_diff'][ind+1] = 5

        et_stims=et_events.loc[et_events['DIN_diff'].isin([5])]
        print('Number of eye-tracking stimulus onset DIN events: ' + str(len(et_stims))) #the length of this array should equal the number of stimulus trials in the task.. and the length of eeg_stims

        #calculate the inter trial interval between eye-tracking stimulus onset DIN events
        et_iti=et_stims['index'].diff()

    if task_name=='ssaep':

        #select only the DIN 8 rows.. and reset the index
        et_events = et_events.copy()
        et_stims=et_events.loc[et_events['DIN_diff'].isin([8])]
        et_events = et_events.reset_index()

        # Search for the beginning of each stimulus sequence.. previous event is more than 300ms away and following stimulus is less than 300ms
        for ind, row in et_events.iterrows():
            if ind == 0:
                et_events['DIN_diff'][ind] = 9
            if ind > 0 and ind < len(et_events)-1:
                if et_events['index'][ind] - et_events['index'][ind-1] > 300:
                        et_events['DIN_diff'][ind] = 9

        et_stims=et_events.loc[et_events['DIN_diff'].isin([9])]
        print('Number of eye-tracking stimulus onset DIN events: ' + str(len(et_stims))) #the length of this array should equal the number of stimulus trials in the task.. and the length of eeg_stims

        #calculate the inter trial interval between eye-tracking stimulus onset DIN events
        et_iti=et_stims['index'].diff()

    if task_name=='plr' or task_name=='PLR':

        #select only the DIN 2 rows.. and reset the index
        et_events=et_events.loc[et_raw_df['DIN_diff'].isin([2,4])]
        et_events = et_events.reset_index()
    
        et_stims=et_events.loc[et_events['DIN_diff'].isin([2,4])]
        print('Number of eye-tracking stimulus onset DIN events: ' + str(len(et_stims))) #the length of this array should equal the number of stimulus trials in the task.. and the length of eeg_stims

        #calculate the inter trial interval between eye-tracking stimulus onset DIN events
        et_iti=et_stims['index'].diff()

    if task_name=='as':

        et_events = et_events.copy()
        et_events = et_events.reset_index()
    
        #Search for the DIN marker of the target stimulus, where where DIN4 is followed by DIN8 then DIN2.. replace the DIN2 with new mark..
        for ind, row in et_events.iterrows():
            if et_events['DIN_diff'][ind] == 4:
                if ind < len(et_events)-2:
                    if et_events['DIN_diff'][ind+1] == 8:
                        if et_events['DIN_diff'][ind+2] == 2:
                            et_events['DIN_diff'][ind+2] = 9

        et_stims=et_events.loc[et_events['DIN_diff'].isin([9])]
        print('Number of eye-tracking stimulus onset DIN events: ' + str(len(et_stims))) #the length of this array should equal the number of stimulus trials in the task.. and the length of eeg_stims

        #calculate the inter trial interval between eye-tracking stimulus onset DIN events
        et_iti=et_stims['index'].diff()

    if task_name=='go':

        # correct anomalous din 12s
        for ind, row in et_events.iterrows():
            if et_events['DIN_diff'][ind] == 12:
                et_events['DIN_diff'][ind] = 4

        #select only the DIN 2 or 4 rows.. and reset the index
        et_events = et_events.copy()
        et_events=et_events.loc[et_raw_df['DIN_diff'].isin([2,4])]
        et_events = et_events.reset_index()

        for ind, row in et_events.iterrows():
            if et_events['DIN_diff'][ind] == 4:
                if ind > 0:
                    if et_events['DIN_diff'][ind-1] == 2:
                        if ind < len(et_events)-1:
                            if et_events['DIN_diff'][ind+1] == 2:
                                et_events['DIN_diff'][ind+1] = 3

        et_stims=et_events.loc[et_events['DIN_diff'].isin([3])]
        print('Number of eye-tracking stimulus onset DIN events: ' + str(len(et_stims))) #the length of this array should equal the number of stimulus trials in the task.. and the length of eeg_stims

        #calculate the inter trial interval between eye-tracking stimulus onset DIN events
        et_iti=et_stims['index'].diff()


    if task_name=='mmn':

        #make a copy of et_events and reset the index
        et_events = et_events.copy()
        et_events = et_events.reset_index()
    
        et_stims=et_events.loc[et_events['DIN_diff'].isin([8])]
        print('Number of eye-tracking stimulus onset DIN events: ' + str(len(et_stims))) #the length of this array should equal the number of stimulus trials in the task.. and the length of eeg_stims

        #calculate the inter trial interval between eye-tracking stimulus onset DIN events
        et_iti=et_stims['index'].diff()


    if task_name=='rest' or task_name=='RS':

        #make a copy of et_events and reset the index
        et_events = et_events.copy()
        et_events = et_events.reset_index()
    
        for ind, row in et_events.iterrows():
            if (ind % 2) != 0:
                et_events['DIN_diff'][ind] = 3

        et_stims=et_events.loc[et_events['DIN_diff'].isin([3])]
        print('Number of eye-tracking stimulus onset DIN events: ' + str(len(et_stims))) #the length of this array should equal the number of stimulus trials in the task.. and the length of eeg_stims

        #calculate the inter trial interval between eye-tracking stimulus onset DIN events
        et_iti=et_stims['index'].diff()


    return et_raw_df, et_events, et_stims, et_iti
'''