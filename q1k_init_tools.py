import mne
import numpy as np
import plotly.express as px

VALID_TASKS = ['rest', 'as', 'AS', 'ssvep', 'vp', 'VEP', 'vs', 'ap', 'AEP',
               'go', 'GO', 'plr', 'mn', 'TO', 'nsp', 'fsp']


def get_event_dict(raw, events, offset):

    stim_names = raw.copy().pick('stim').info['ch_names']
    event_dict = {event: int(i) + offset
                  for i, event in enumerate(stim_names)
                  if event != 'STI 014'}

    """ #method for building building the event_dict is precarious, but it seems to satisfy all of the cases... to be reworked later
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


def eeg_event_test(eeg_events, eeg_event_dict, din_str, task_name=None):
    
    din_offset = []
    
    if not task_name:
        raise ValueError(f'please pass one of {VALID_TASKS}'
                         ' to the task_name keyword argument.')

    if task_name == 'ap' or task_name == 'AEP':

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

        ##look for 'dtoc'>'DIN2', 'dtbc'>'DIN2', 'dtgc'>'DIN2'
        #for i, e in np.ndenumerate(eeg_events[:,2]):
        #    if e == eeg_event_dict['dtoc']:
        #        if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN2']:
        #            eeg_events[i[0]+1, 2] = len(eeg_event_dict) + 1 #ae06 DIN onset
        #    if e == eeg_event_dict['dtbc']:
        #        if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN2']:
        #            eeg_events[i[0]+1, 2] = len(eeg_event_dict) + 2 #ae40 DIN onset
        #    if e == eeg_event_dict['dtgc']:
        #        if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN2']:
        #            eeg_events[i[0]+1, 2] = len(eeg_event_dict) + 3 #ae40 DIN onset

        # add the new stimulus onset DIN labels to the event_dict..
        #eeg_event_dict['dtoc_d'] = len(eeg_event_dict) + 1
        #eeg_event_dict['dtbc_d'] = len(eeg_event_dict) + 1
        #eeg_event_dict['dtgc_d'] = len(eeg_event_dict) + 1

        ## select all of the newly categorized stimulus DIN events
        #mask = np.isin(eeg_events[:,2],[eeg_event_dict['dtoc_d'],eeg_event_dict['dtbc_d'],eeg_event_dict['dtgc_d']])
        #eeg_stims = eeg_events[mask]
        #print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        ##calculate the inter trial interval between stimulus onset DIN events
        #eeg_iti = np.diff(eeg_stims[:,0])


    elif task_name == 'vp' or task_name == 'VEP':

        # remove TSYN events...this might have to happen for all tasks.. because this is not used for anything and they appear in arbitrary locations...
        print('Removing TSYN events...')
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['TSYN']])
        eeg_events = eeg_events[~mask]
        new_events = np.empty((0, 3))

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e == eeg_event_dict['sv06']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e == eeg_event_dict['sv15']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
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

    elif task_name == 'plr':

        # for the plr task it is more simple to select trials based on DIN2 occurences
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['DIN2']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        # calculate the inter trial interval between stimulus onset DIN events
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


    elif task_name=='rest':

        #for the plr task it is more simple to select trials based on DIN2 occurences
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['DIN2']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])
    elif task_name in ['vs', 'fsp', 'nsp']:
        raise NotImplemented
    else:
        raise ValueError('Could not determine task name.'
                         f' Expected one of {VALID_TASKS} but got {task_name}')


    return eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events


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


    if task_name=='vp':

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

    if task_name=='plr':

        #select only the DIN 2 rows.. and reset the index
        et_events=et_events.loc[et_raw_df['DIN_diff'].isin([2])]
        et_events = et_events.reset_index()
    
        et_stims=et_events.loc[et_events['DIN_diff'].isin([2])]
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


    if task_name=='rest':

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

def show_sync_offsets(eeg_stims,et_stims):
    eeg_et_offset = eeg_stims[:,0] - et_stims['index'][:]

    fig = px.scatter(y=eeg_et_offset)
    fig.show()


def eeg_et_combine(eeg_raw, et_raw, eeg_stims, et_stims):

    eeg_times=eeg_stims[:,0]/1000
    et_times=et_stims['time'].reset_index(drop=True).to_numpy()

    mne.preprocessing.realign_raw(et_raw, eeg_raw, et_times, eeg_times, verbose=None)


    eeg_names = eeg_raw.copy().pick_types(eeg=True).info['ch_names']
    eeg_types = eeg_raw.copy().pick_types(eeg=True).get_channel_types()
    eeg_raw_array = eeg_raw.copy().pick_types(eeg=True).get_data()

    eeg_stim_names = eeg_raw.copy().pick_types(stim=True).info['ch_names']
    eeg_stim_types = eeg_raw.copy().pick_types(stim=True).get_channel_types()
    eeg_stim_raw_array = eeg_raw.copy().pick_types(stim=True).get_data()

    et_names = et_raw.copy().info['ch_names']
    et_types = et_raw.copy().get_channel_types()
    et_raw_array = et_raw.copy().get_data()

    eeg_et_array = np.vstack((eeg_raw_array, et_raw_array, eeg_stim_raw_array))

    info = mne.create_info(ch_names = eeg_names + et_names + eeg_stim_names,
                    sfreq = 1000,
                    ch_types=eeg_types + et_types + eeg_stim_types)

    eeg_et_raw = mne.io.RawArray(eeg_et_array, info)

    return eeg_et_raw

