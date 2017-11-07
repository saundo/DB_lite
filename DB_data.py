### Keen Video and Interaction
import datetime
import numpy as np
import pandas as pd
from retrying import retry
from queue import Queue
from threading import Thread
from functools import wraps
from keen.client import KeenClient

def API_log(func):
    '''
    Decorator that keeps track of what succeeds and what fails
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            print('S', end='|')
            return result
        except:
            e2 = sys.exc_info()[1]
            e2 = e2.args[0]['error_code']
            failure_log.append(tuple(args) + (e2,))
            print(e2, end='|')
    return wrapper

################################## Threading ###################################
class DownloadWorker1(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            func, start, end, kwargs = self.queue.get()
            run_func(func, start, end, kwargs)
            self.queue.task_done()

def run_func(func, start, end, kwargs):
    """
    """
    key = func.__name__ + '-' + str(start)
    thread_storage[key] = func(start, end, **kwargs)

def run_thread(func, timeframe, kwargs):
    """
    """
    global thread_storage
    thread_storage = {}
    queue = Queue()
    for x in range(8):
        worker = DownloadWorker1(queue)
        worker.daemon = True
        worker.start()

    for start,end in timeframe:
        queue.put((func, start, end, kwargs))

    queue.join()
    return thread_storage

################################## Time generator ##############################
def timeframe_gen(start, end, hour_interval=24, tz='US/Eastern'):
    """creates timeframe for use in making Keen API calls
    args:
        start - start date (str - '2017-08-04'); inclusive
        end - end date (str - '2017-08-04'); inclusive
    kwargs:
        hour_interval - interval for breaking up start, end tuple
        tz - timezone

    returns:
        List of tuples, tuple - (start, end)
    """
    freq = str(hour_interval) + 'H'
    start_dates = pd.date_range(start, end, freq=freq, tz=tz)
    start_dates = start_dates.tz_convert('UTC')
    end_dates = start_dates.shift(1)

    start_times = [datetime.datetime.strftime(i, '%Y-%m-%dT%H:%M:%S.000Z') for i in start_dates]
    end_times = [datetime.datetime.strftime(i, '%Y-%m-%dT%H:%M:%S.000Z') for i in end_dates]
    timeframe = [(start_times[i], end_times[i]) for i in range(len(start_times))]
    return timeframe

################################## KEEN API calls  and unpacking ###############
class API_calls():
    def __init__(self, name):
        self.name = name

    def initialize_keen(self, Keen_API_credentials, Keen_silo):
        projectID = Keen_API_credentials[Keen_silo]['projectID']
        readKey = Keen_API_credentials[Keen_silo]['readKey']
        self.keen = KeenClient(project_id=projectID, read_key=readKey)

    @API_log
    def ad_interaction(self, start, end, **kwargs):
        event = 'ad_interaction'
        timeframe = {'start':start, 'end':end}
        interval = 'every_24_hours'
        timezone = None
        group_by = (
            'user.cookie.session.id',
            'interaction.name',
            'interaction.target',
            'creative_placement.dfp.creative.id',
            'creative_placement.dfp.line_item.id',
            'creative_placement.device',
            'creative.type',

            'user.ip_address',
            'parsed_page_url.domain',
            'url.domain',
            )

        data = self.keen.count(event,
                          timeframe=timeframe, interval=interval, timezone=timezone,
                          group_by=group_by,
                          filters=None)
        return data

    @API_log
    def ad_video_progress(self, start, end, **kwargs):
        event = 'ad_video_progress'
        timeframe = {'start':start, 'end':end}
        interval = 'every_24_hours'
        timezone = None
        group_by = (
            'user.cookie.session.id',
            'video.progress.percent_viewed',
            'creative_placement.dfp.creative.id',
            'creative_placement.dfp.line_item.id',
            'creative.type',
            'creative_placement.device',

            'user.ip_address',
            'parsed_page_url.domain',
            'url.domain',
            )

        data = self.keen.count(event,
                          timeframe=timeframe, interval=interval, timezone=timezone,
                          group_by=group_by,
                          filters=None)
        return data

    @API_log
    def ad_impression(self, start, end, **kwargs):
        event = 'ad_impression'
        timeframe = {'start':start, 'end':end}
        interval = 'every_72_hours'
        timezone = None
        group_by = (
            'creative_placement.dfp.creative.id',
            'creative_placement.dfp.line_item.id',
            'creative_placement.device',
            'creative_placement.versions.this.name',
            'creative.type',
            'creative.name',
            'creative.context',
            'creative.value',
            'parsed_page_url.domain',
            'url.domain',
            )

        data = self.keen.count(event,
                          timeframe=timeframe, interval=interval, timezone=timezone,
                          group_by=group_by,
                          filters=None)
        return data

def unpack_keen(data):
    """
    unpacks keen data BUT ALSO renames some of the long ass names that keen uses
        'date': 'Date',
        'creative_placement.dfp.creative.id': 'Creative ID',
        'creative_placement.dfp.line_item.id': 'Line item ID',
        'creative_placement.device': 'device',
        'user.cookie.session.id': 'cookie_s',
        'interaction.name': 'interaction',
        'user.ip_address':'ip_address',
        'creative_placement.versions.this.name':'version'
    """
    #parameter dictionary; DFP to conform with STAQ & easy language
    p_dict = {
        'date': 'Date',
        'creative_placement.dfp.creative.id': 'Creative ID',
        'creative_placement.dfp.line_item.id': 'Line item ID',
        'creative_placement.device': 'device',
        'user.cookie.session.id': 'cookie_s',
        'interaction.name': 'interaction',
        'user.ip_address':'ip_address',
        'creative_placement.versions.this.name':'version',
        }
    s1 = []
    for key in data.keys():
        for d1 in data[key]:
            start = d1['timeframe']['start']
            value = d1['value']
            df = pd.DataFrame(value)
            df['date'] = start
            df = df[df['creative_placement.dfp.creative.id'].notnull()]
            df = df[df['result'] > 0]
            s1.append(df)
    df = pd.concat(s1)
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    df = df.rename(index=str, columns=p_dict)

    # check for weird results  --> verify with Apostolis
    dfx = df[df['Creative ID'] != str('%ecid!')].copy()
    dfx['Creative ID'] = dfx['Creative ID'].astype(int)
    dfx['Line item ID'] = dfx['Line item ID'].astype(int)
    print('unpacked')
    print((len(df) - len(dfx)), ' rows with anomalies')
    return dfx

################################## STAQ ########################################
class STAQ_prep():
    def __init__(self, file_name, directory='/users/csaunders/Downloads/'):
        dtypes = {
            'DFP Creative ID Impressions': int,
            'DFP Creative ID Clicks': int,
            'Normalized 3P Impressions': int,
            'Normalized 3P Clicks': int,
            'Ad server downloaded impressions': int,
            'Ad server Active View viewable impressions': int,
        }

        self.STAQ = pd.read_csv(directory + file_name, encoding = "ISO-8859-1",
            thousands=",", dtype=dtypes)
        self.STAQ['Date'] = pd.to_datetime(self.STAQ['Date'])

        self.device_lookup = {
            'Desktop' : 'desktop',
            'Feature Phone': 'mobile',
            'Smartphone': 'mobile',
            'Tablet' : 'tablet'
        }

        self.STAQ_cols = {
            'categories':
                ['site', 'Date', 'Advertiser', 'Order', 'Ad unit', 'Line item ID',
                 'Line item', 'Creative ID',   'Creative', 'placement', 'device'],
            'values':
                ['DFP Creative ID Impressions', 'DFP Creative ID Clicks',
                 'Normalized 3P Impressions', 'Normalized 3P Clicks',
                 'Ad server Active View viewable impressions']
        }
        print(np.shape(self.STAQ))

    def clean_numbers(self):
        """
        ensure the STAQ columns are integers
        this may no longer be necessary
        """
        for c in self.STAQ.loc[:,'DFP Drop ID impressions':].columns:
            if self.STAQ[c].dtype == 'O':
                self.STAQ[c] = self.STAQ[c].str.replace(',','').fillna(0).astype(int)
            else:
                self.STAQ[c] = self.STAQ[c].fillna(0)

    def split_into_sites(self):
        site_lookup = {
            'Android_App':'android',
            'chart':'chart',
            'flipboard_qz_fsa': 'flipboard',
            'iPhone_app':'iphone',
            'qz':'qz',
            'work':'wrk',
            'email-card':'oth',
            'email-logo':'oth',
            'email-content':'oth',
            'flash-brief': 'oth',
            'iphone_test':'oth',
            'newsstand':'Google News ads',
            'qztest':'qztest'
        }

        try:
            self.STAQ['site'] = self.STAQ['Ad unit'].apply(lambda x: site_lookup[x])
        except:
            raise ValueError("matching Ad unit to site failed")

    def apply_placement_names_qz(self):
        """
        match placement name to the placement sizes; along with other logic
        WILL NEED TO REVIST AFTER QUARTZ at WORK
        """
        #qz
        placement_size = ['1600 x 520', '1400 x 520', '640 x 360',
                          '1600 x 521', '1400 x 521', '600 x 431',
                          '640 x 363', '520 x 293', '360 x 203',
                          '1600 x 1600']

        placement_name = ['marquee desktop', 'marquee tablet', 'marquee mobile',
                          'engage desktop', 'engage tablet', 'engage mobile',
                          'inline desktop', 'inline tablet', 'inline mobile',
                          'Bulletin']

        placement_size_to_name = dict(zip(placement_size, placement_name))

        def placement_logic(x):
            if x == 'Total':
                return 'Total'
            if x == '300 x 250':
                return 'Atlas'
            if x == '1920 x 1080':
                return 'Video sponsor unit'
            if x == '600 x 430':
                return 'legacy engage mobile'
            try:
                return placement_size_to_name[x]
            except:
                return 'oth'

        self.STAQ_qz = self.STAQ[self.STAQ['site'] == 'qz'].copy()
        self.STAQ_qz['placement'] = self.STAQ_qz['Creative size'].apply(placement_logic)
        self.STAQ_qz['device'] = self.STAQ_qz['Device category'].apply(lambda x: self.device_lookup[x])

    def apply_placement_names_wrk(self):
        """
        match placement name to the placement sizes; along with other logic
        WILL NEED TO REVIST AFTER QUARTZ at WORK
        """
        #qz

        creative_size_lookup = {
            '1600 x 520': 'marquee',
            '1600 x 521': 'engage',
            '640 x 363': 'inline',
            '1 x 1': 'spotlight',
            'Out-of-page': 'oop'
        }

        self.STAQ_wrk = self.STAQ[self.STAQ['site'] == 'wrk'].copy()

        self.STAQ_wrk['adunit'] = self.STAQ_wrk['Creative size'].apply(lambda x: creative_size_lookup[x])
        self.STAQ_wrk['device'] = self.STAQ_wrk['Device category'].apply(lambda x: self.device_lookup[x])
        self.STAQ_wrk['placement'] = self.STAQ_wrk['adunit'] + ' ' + self.STAQ_wrk['device']
        del self.STAQ_wrk['adunit']

    def compile(self):
        # need to do something with this raw, as it contains sites other than
        # qz and wrk --> see site_lookup
        self.STAQ_raw = self.STAQ

        self.STAQ = self.STAQ_qz.append(self.STAQ_wrk)
        self.STAQ = self.STAQ[self.STAQ_cols['categories'] + self.STAQ_cols['values']]

        #combine duplicate mobile devices (smartphone and feature phone)
        #not combining creates issues when merging Keen to this
        mob = self.STAQ[self.STAQ['device'] == 'mobile']
        not_mob = self.STAQ[self.STAQ['device'] != 'mobile']
        mob = mob.groupby(self.STAQ_cols['categories'], as_index=False).sum()
        self.STAQ = mob.append(not_mob)
        print(np.shape(self.STAQ))

################################## VID #########################################
class VID_calc():
    def __init__(self, VID):
        self.VID = VID

        self.KeyID = ('Date', 'Creative ID', 'Line item ID', 'device')
        self.KeyID_creative = ('Date', 'Creative ID', 'Line item ID', 'creative.type')

    def split_into_sites(self):
        sp_logic = {
            'parsed_page_url.domain':'work.qz.com',
            'url.domain':'qz.com'
            }

        # check if Apostolis changed the domain splitting
        if 'work.qz.com' in set(self.VID['url.domain']):
            raise ValueError('uhoh - Apostolis did something without telling us!'
                             'it looks like he updated the url.domain to include'
                             'to include quartz at work')

        # qz
        x = 'url.domain'
        self.VID_qz = self.VID[self.VID[x] == sp_logic[x]].copy()

        #wrk
        x = 'parsed_page_url.domain'
        self.VID_wrk = self.VID[self.VID[x] == sp_logic[x]].copy()

    def wrangle_vid(self):
        def wrangling(df, session_key='cookie_s'):
            KeyID_session = list(self.KeyID) + [session_key]
            param = 'video.progress.percent_viewed'
            v5 = df[df[param] == 5]
            v75 = df[df[param] == 75]
            v90 = df[df[param] == 90]
            v100 = df[df[param] == 100]
            cols = KeyID_session + ['result']

            v5_75_session = pd.merge(v5[cols], v75[cols],
                on=KeyID_session, how='outer')
            v5_75_views = v5_75_session.groupby(self.KeyID, as_index=False).sum()
            v5_75_views.columns = ['Date', 'Creative ID', 'Line item ID',
                                   'device', 'result_5', 'result_75']

            v90_100_session = pd.merge(v90[cols], v100[cols],
                on=KeyID_session, how='outer')
            v90_100_views = v90_100_session.groupby(self.KeyID, as_index=False).sum()
            v90_100_views.columns = ['Date', 'Creative ID', 'Line item ID',
                                     'device', 'result_90', 'result_100']

            vid_views = pd.merge(v5_75_views, v90_100_views, on=self.KeyID, how='outer')

            self.tester = vid_views
            vid_views['Date'] = pd.to_datetime(vid_views['Date'])
            return vid_views

        self.VID_qz_views = wrangling(self.VID_qz)
        self.VID_wrk_views = wrangling(self.VID_wrk,
            session_key='ip_address')
        #zty #### TK
        print('vid wrangled!')

################################## IR ##########################################
class INT_calc():
    def __init__(self, VID, INT):
        self.VID = VID
        self.INT = INT

        self.KeyID = ('Date', 'Creative ID', 'Line item ID', 'device')
        self.KeyID_creative = ('Date', 'Creative ID', 'Line item ID', 'creative.type')

    def filter_module(self, df, **kwargs):
        if 'interaction.target' in kwargs:
            ignore = kwargs['interaction.target']
        else:
            ignore = None

        if ignore is not None:
            dfx = df[df['interaction.target'] != ignore]
            return dfx
        else:
            return dfx

    def filter_interactions(self, removal_items):
        """
        argument: 'removal_items' is a dictionary with Creative ID as the key
        and interaction to remove as the value
            ex: {138210468919:'clicked', 138210549824:'hover'}
            Interaction value will run a contains lookup within the dataframe so
            the value does not have to be exact.

        Returns DataFrame without those Creative ID/interaction combinations
        """
        orig_list = self.INT[['Creative ID', 'interaction']].apply(tuple, axis=1)
        print('original length ', len(self.INT))
        print('\033[1m'+'Removal inputs:'+'\033[0m')

        df_store = pd.DataFrame()
        for key, value in removal_items.items():
            dft = self.INT.copy()
            dft = dft[(dft['Creative ID']==key) &
                      (dft['interaction'].str.contains(value))]
            df_store = df_store.append(
                dft[['Creative ID','interaction']]).drop_duplicates()
            df_store['marker'] = 1
            print(key, value)

        df_clean = pd.merge(self.INT, df_store,
                            on=['Creative ID', 'interaction'], how='left')
        df_clean = df_clean[pd.isnull(df_clean['marker'])][self.INT.columns]
        new_list = df_clean[['Creative ID', 'interaction']].apply(tuple, axis=1)
        self.INT = df_clean

        print('\033[1m'+'Removed interactions:'+'\033[0m',
            list(set(orig_list)-set(new_list)))
        print('\033[1m'+"New DataFrame created with length: ", len(self.INT))

    def split_into_sites(self):
        sp_logic = {
            'parsed_page_url.domain':'work.qz.com',
            'url.domain':'qz.com'
            }
        # qz
        x = 'url.domain'
        self.VID_qz = self.VID[self.VID[x] == sp_logic[x]].copy()
        self.INT_qz = self.INT[self.INT[x] == sp_logic[x]].copy()

        #wrk
        x = 'parsed_page_url.domain'
        self.VID_wrk = self.VID[self.VID[x] == sp_logic[x]].copy()
        self.INT_wrk = self.INT[self.INT[x] == sp_logic[x]].copy()

        #zty
        #### TK

    def wrangle_vid(self):
        """
        """
        ### this on IR returns sessions
        def wrangling(df, session_key='cookie_s'):
            KeyID_session = list(self.KeyID) + [session_key]
            param = 'video.progress.percent_viewed'
            v5 = df[df[param] == 5]
            v75 = df[df[param] == 75]
            cols = KeyID_session + ['result']

            v5_75_session = pd.merge(v5[cols], v75[cols],
                on=KeyID_session, how='outer')
            return v5_75_session

        self.VID_qz_session = wrangling(self.VID_qz)
        self.VID_wrk_session= wrangling(self.VID_wrk,
            session_key='ip_address')
        print('vid & int wrangled!')

    def wrangle_int(self):
        """
        """
        self.filter_kwargs = {'interaction.target':'external'}
        self.INT_qz = self.filter_module(self.INT_qz, **self.filter_kwargs)
        self.INT_wrk = self.filter_module(self.INT_wrk, **self.filter_kwargs)

        #qz
        KeyID_session = list(self.KeyID) + ['cookie_s']
        self.INT_qz = self.INT_qz.groupby(KeyID_session, as_index=False).sum()

        #wrk
        KeyID_session = list(self.KeyID) + ['ip_address']
        self.INT_wrk = self.INT_wrk.groupby(KeyID_session, as_index=False).sum()

    def combine_vid_int(self):
        """
        """
        def merge_it(df1, df2, onkey, howkey):
            merged = pd.merge(df1, df2, on=onkey, how=howkey)

            int_tot = merged.groupby(self.KeyID, as_index=False).sum()
            int_tot = int_tot[list(self.KeyID) + ['result']]

            int_ses = merged.groupby(self.KeyID, as_index=False).count()
            int_ses = int_ses[KeyID_session]
            int_ses = pd.merge(int_ses, int_tot, on=self.KeyID, how='outer')
            int_ses.columns = ['Date', 'Creative ID', 'Line item ID',
                               'device', 'int sessions', 'interactions']
            int_ses['Date'] = pd.to_datetime(int_ses['Date'])
            return int_ses

        #qz
        KeyID_session = list(self.KeyID) + ['cookie_s']
        self.INT_ses_qz = merge_it(self.VID_qz_session, self.INT_qz,
            KeyID_session, 'outer')
        #wrk
        KeyID_session = list(self.KeyID) + ['ip_address']
        self.INT_ses_wrk = merge_it(self.VID_wrk_session, self.INT_wrk,
            KeyID_session, 'outer')

################################## Creative types ##############################
class creative_types():
    def __init__(self, IMP):
        self.IMP = IMP
        self.KeyID_creative = (
            'creative.type',
            'creative.name',
            'device',
            'Creative ID',
            'Line item ID',
            'version'
        )

    def make_lookups(self):
        df = self.IMP.sort_values('Date', ascending=False)
        df = df[~df[list(self.KeyID_creative)].duplicated()]
        df = df[list(self.KeyID_creative)]

        df['version'] = df['version'].fillna('')
        def version_scrub(x):
            if isinstance(x, float):
                return int(x)
            else:
                return x
        df['version'] = df['version'].apply(version_scrub).astype(str)
        df['creative.name.version'] = df['creative.name'] + '.' + df['version']

        self.creative_lookup = df

################################## Assemble ####################################
class assemble():
    def __init__(self, creative_lookup):
        self.creative_lookup = creative_lookup
        self.KeyID = ('Date', 'Creative ID', 'Line item ID', 'device')

    def assemble_qz(self, STAQ_qz, VID_qz, IR_qz):
        """
        STAQ_qz
        VID_qz
        IR_qz
        """
        KeyID = ('Date', 'Creative ID', 'Line item ID', 'device')

        df1 = STAQ_qz
        df2 = VID_qz.fillna(0)
        df_qz = pd.merge(df1, df2, on=KeyID, how='left')

        df3 = IR_qz.fillna(0)
        self.df_qz = pd.merge(df_qz, df3, on=KeyID, how='left')
        print(np.shape(df1))
        print(np.shape(df2))
        print(np.shape(df3))
        print(np.shape(df_qz))

    def assemble_wrk(self, STAQ_wrk, VID_wrk, IR_wrk):
        """
        STAQ_wrk
        VID_wrk
        IR_wrk
        """
        KeyID = ('Date', 'Creative ID', 'Line item ID', 'device')

        df1 = STAQ_wrk
        df2 = VID_wrk.fillna(0)
        df_wrk = pd.merge(df1, df2, on=KeyID, how='left')

        df3 = IR_wrk.fillna(0)
        self.df_wrk = pd.merge(df_wrk, df3, on=KeyID, how='left')
        print(np.shape(df1))
        print(np.shape(df2))
        print(np.shape(df3))
        print(np.shape(df_wrk))

    def assemble_zty(self):
        """
        STAQ_zty
        VID_zty
        IR_zty
        """
        pass

    def assemble_all(self):
        self.df = self.df_qz.append(self.df_wrk)
        print(np.shape(self.df))
        self.df_master = pd.merge(self.df, self.creative_lookup,
            on=('Creative ID', 'Line item ID', 'device'), how='left')
        print(np.shape(self.df_master))

    def apply_descriptors(self):
        """
        Keep it at creative.type
        THIS ALSO WILL NEED SOME TLC after Quartz at Work
        """

        def adunit(x):
            if 'engage' in x:
                return 'engage'
            elif 'marquee' in x:
                return 'marquee'
            elif 'inline' in x:
                return 'inline'
            else:
                return 'oth'

        self.df_master['adunit'] = self.df_master['placement'].apply(adunit)
        #fill creative.type with values other than NaN
        self.df_master['creative.type'] = self.df_master['creative.type'].fillna('no match')
