import mne
import numpy as np
import plotly.express as px

def eeg_event_test(eeg_events, task_name=''):

    if task_name=='ssvep':
        eeg_event_dict = {
                            "sv06": 1,
                            "fvsb": 2,
                            "sv15": 3,
                            "fvct": 4,
                            "fvsr": 5,
                            "TSYN": 6,
                            "fvcr": 7,
                            "dbrk": 8,
                            "dend": 9,
                            "DIN4": 10,
                            "DIN3": 11,
                            "DIN2": 12,
                            "DIN5": 13,
                            "VBeg": 14,
                            "sd06": 15,
                            "sd15": 16,
                            }
        #find DIN marker for stimulus onsets (this will be a specific event index combination for each task)
        #   this should be moved to a function that accepts the task name as an input..
        for i, x in np.ndenumerate(eeg_events[:,2]):
            if x == 1:
                if eeg_events[i[0]+1, 2] == 12:
                    eeg_events[i[0]+1, 2] = 15 #sv06 DIN onset
            if x == 3:
                if eeg_events[i[0]+1, 2] == 12:
                    eeg_events[i[0]+1, 2] = 16 #sv15 DIN onset

        #select all of the newly categorized stimulus DIN events
        #mask = (eeg_events[:,2] == 15)
        mask = np.isin(eeg_events[:,2],[15,16])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of thsi array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])

        return eeg_events, eeg_stims, eeg_iti, eeg_event_dict


def et_event_test(et_raw_df, task_name=''):
    #convert the ET DIN channel into ET events
    #find when the DIN channel changes values
    et_raw_df['DIN_diff']=et_raw_df['DIN'].diff()
    #select all non-zero DIN changes
    et_events=et_raw_df.loc[et_raw_df['DIN_diff']!=0]

    #   there should only be DIN 2 and 4 in the Q1K visual tasks.. however there are frequently binary values greater than 4 indicating that there are anomalous pin4 and pin5 pulses
    #   bin2=pin2, bin4=pin3, bin8=pin4, bin16=pin5, bin18=pin2+pin5, bin20=pin3+pin5, bin24=pin4+pin5, bin26=pin2+pin4+pin5, bin28=pin3+pin4+pin5
    #   given these anomalous pin4 and pin5 pulses the conversion at pin change time is: binary 2,18,26 = 2, and binary 4,20,28 = 4
    #perform the anomalous DIN conversion
    et_events = et_events.copy()
    et_events['DIN'].loc[et_events['DIN'].isin([2,18,26])] = 2
    et_events['DIN'].loc[et_events['DIN'].isin([4,20,28])] = 4

    #now select only the DIN 2 and 4 rows.. and reset the index
    et_events=et_events.loc[et_raw_df['DIN'].isin([2,4])]
    et_events = et_events.reset_index()
 
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

    return et_events, et_stims, et_iti

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

    et_names = et_raw.info['ch_names']
    et_types = et_raw.get_channel_types()
    et_raw_array = et_raw.get_data()

    eeg_et_array = np.vstack((eeg_raw_array, et_raw_array))

    info = mne.create_info(ch_names = eeg_names + et_names,
                    sfreq = 1000,
                    ch_types=eeg_types + et_types )

    eeg_et_raw = mne.io.RawArray(eeg_et_array, info)

    return eeg_et_raw

