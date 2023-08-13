from .fish import Fish
from caImageAnalysis.mesm import load_mesmerize
from caImageAnalysis.utils import get_injection_frame


class VolumeFish(Fish):
    def __init__(self, folder_path):
        super().__init__(folder_path)

        self.process_volumetric_filestructure()

    def split_volume(self, len_thresh=50):
        '''Splits raw volumetric image to separate .tif files of individual planes'''
        self.raw_text_frametimes_to_df()
        self.raw_text_logfile_to_df()
        frametimes = self.align_frames_to_steps(time_offset=.05)

        if 'flipped' in self.data_paths.keys():
            img = cm.load(self.data_paths['flipped'])
        elif 'cropped' in self.data_paths.keys():
            img = cm.load(self.data_paths['cropped'])
        else:
            img = cm.load(self.data_paths['raw_image'])

        for n, s in enumerate(frametimes.step.unique()):
            img_inds = frametimes[frametimes.step == s].index
            new_fts = frametimes[frametimes.step == s].drop(columns='step')
            new_fts = self.align_injection_to_frames(new_fts)

            folderpath = self.data_paths['raw_image'].parents[0].joinpath(f'img_stack_{n}')
            if not os.path.exists(folderpath):
                os.mkdir(folderpath)
            sub_img = img[img_inds]
            if len(sub_img) >= len_thresh:
                new_img_path = folderpath.joinpath(f'image.tif')
                imwrite(new_img_path, sub_img)
                print(f'saved {new_img_path}')

                new_framet_path = folderpath.joinpath('frametimes.h5')
                new_fts.to_hdf(new_framet_path, 'frametimes')
                print(f'saved {new_framet_path}')

        self.process_volumetric_filestructure()

    def raw_text_logfile_to_df(self):
        '''Returns piezo steps in a log txt file as a dataframe'''
        with open(self.data_paths['log']) as file:
            contents = file.read()
        parsed = contents.split("\n")

        movesteps = []
        times = []
        for line in range(len(parsed)):
            if (
                    "piezo" in parsed[line]
                    and "connected" not in parsed[line]
                    and "stopped" not in parsed[line]
            ):
                t = parsed[line].split(" ")[0][:-1]
                z = parsed[line].split(" ")[6]
                try:
                    if isinstance(eval(z), float):
                        times.append(dt.strptime(t, "%H:%M:%S.%f").time())
                        movesteps.append(z)
                except NameError:
                    continue
        else:
            # last line is blank and likes to error out
            pass

        log_steps = pd.DataFrame({"time": times, "steps": movesteps})

        if hasattr(self, "frametimes_df"):
            log_steps = self.trim_log(log_steps)

        self.log_steps = log_steps
        return log_steps

    def trim_log(self, log_steps):
        '''Trims log piezo steps to only the ones within the frametimes'''
        trimmed_logsteps = log_steps[
            (log_steps.time >= self.frametimes_df.iloc[0].values[0])
            & (log_steps.time <= self.frametimes_df.iloc[-1].values[0])
            ]
        return trimmed_logsteps

    def align_frames_to_steps(self, intermediate_return=False, time_offset=0.1):
        '''Aligns image frames to log steps
        time_offset: milliseconds off between the log/step information and frametimes time stamp'''

        logtimes_mod = []  # modified logtimes list
        missed_steps = []

        for t in range(len(self.frametimes_df)):
            listed_time = str(self.frametimes_df.values[t][0]).split(':')
            time_val = float(listed_time[-1])

            seconds_min = time_val - time_offset
            seconds_max = time_val + time_offset
            # clip function to make sure the min is 0, no negative times
            seconds_min = np.clip(seconds_min, a_min=0, a_max=999)

            min_listed_time = listed_time.copy()
            min_listed_time[-1] = str(np.float32(seconds_min))

            max_listed_time = listed_time.copy()
            max_listed_time[-1] = str(np.float32(seconds_max))

            if seconds_max > 60:
                seconds_max -= 60
                max_listed_time[-1] = str(np.float16(seconds_max))
                new_seconds = int(max_listed_time[1]) + 1
                max_listed_time[1] = str(int(new_seconds))

            if seconds_max >= 60:
                seconds_max -= 60
                max_listed_time[-1] = str(np.float16(seconds_max))
                new_seconds = int(max_listed_time[1]) + 1
                max_listed_time[1] = str(int(new_seconds))

            mintime = dt.strptime(':'.join(min_listed_time), '%H:%M:%S.%f').time()

            maxtime = dt.strptime(':'.join(max_listed_time), '%H:%M:%S.%f').time()

            temp = self.log_steps[(self.log_steps.time >= mintime) & (self.log_steps.time <= maxtime)]

            ## sometimes there are missed steps (no frame with the next step in the stack) so we need to take that out
            if len(temp) != 0:
                logtimes_mod.append(temp)
            else:
                missed_steps.append(t)
        ## this is a check here, so if intermediate_return is true, then it will stop here and return the frametimes and logtimes_mod dataframes
        if intermediate_return:
            return self.frametimes_df, logtimes_mod

        frametimes_with_steps = []
        for df_row in logtimes_mod:
            frametimes_with_steps.append(df_row.steps.values[0])

        self.frametimes_df.drop(missed_steps, inplace=True)
        self.frametimes_df.loc[:, 'step'] = frametimes_with_steps
        self.frametimes_df.loc[:, 'step'] = self.frametimes_df.step.astype(np.float32)
        return self.frametimes_df

    def process_volumetric_filestructure(self):
        '''Appends volumetric file paths to the data_paths'''
        self.data_paths['volumes'] = dict()
        with os.scandir(self.data_paths['postgavage_path']) as entries:
            for entry in entries:
                if entry.name.startswith('img_stack_'):
                    volume_ind = entry.name[entry.name.rfind('_')+1:]
                    self.data_paths['volumes'][volume_ind] = dict()

                    with os.scandir(entry.path) as subentries:
                        for sub in subentries:
                            if sub.name == 'image.tif':
                                self.data_paths['volumes'][volume_ind]['image'] = Path(sub.path)
                            elif sub.name == 'frametimes.h5':
                                self.data_paths['volumes'][volume_ind]['frametimes'] = Path(sub.path)
                                self.data_paths['volumes'][volume_ind]['inj_frame'] = get_injection_frame(Path(sub.path))
                            elif 'C_frames' in sub.name:
                                self.data_paths['volumes'][volume_ind]['C_frames'] = Path(sub.path)
                            elif 'F_frames' in sub.name:
                                self.data_paths['volumes'][volume_ind]['F_frames'] = Path(sub.path)
        self.process_mesmerize_filestructure()

    def process_mesmerize_filestructure(self):
        '''TODO: mesmerize filestructure for non-volumetric images'''
        '''Appends volumetric mesmerize uuid paths to the data_paths'''
        if 'mesmerize' in self.data_paths.keys():
            mes_df = load_mesmerize(self)
            for i, row in mes_df.iterrows():
                if row.algo == 'mcorr':
                    try:
                        plane = row.item_name[row.item_name.rfind('_')+1:]
                        mesm_path = self.data_paths['mesmerize']
                        self.data_paths['volumes'][plane]['mcorr'] = self.data_paths['mesmerize'].joinpath(row.outputs['mcorr-output-path'])
                    except:
                        pass

                elif row.algo == 'cnmf':
                    try:
                        plane = row.item_name[row.item_name.rfind('_')+1:]
                        self.data_paths['volumes'][plane]['cnmf'] = self.data_paths['mesmerize'].joinpath(row.outputs['cnmf-memmap-path'])
                    except:
                        pass
