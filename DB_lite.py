import os
import pandas as pd
import numpy as np
import ipywidgets
import datetime
import IPython
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

################################## Excel_exporter ##############################
def loop_exports(self, directory='/users/csaunders/Desktop/', number='all',
    exact=None):
    """
    """
    #for error testing
    if exact is not None:
        advert = exact
        df = self.df[self.df['Advertiser'] == advert]
        self.execute_export(df, export, directory)
        print('completed -', export, end=' ||| ')

    #for error testing several
    elif number == 'all':
        for export in self.export_list:
            print('start ', export, end='-')
            sb_advert = self.df['Advertiser'] == export[0]
            sb_order = self.df['Order'] == export[1]
            df = self.df[(sb_advert) & (sb_order)]
            self.execute_export(df, export, directory)
            print('completed -', export, end=' ||| ')

    #the actual shibang
    else:
        options = [i for i in range(len(self.export_list))]
        for i in range(number):
            export = self.export_list[np.random.choice(options)]
            print('start ', export, end='-')
            sb_advert = self.df['Advertiser'] == export[0]
            sb_order = self.df['Order'] == export[1]
            df = self.df[(sb_advert) & (sb_order)]
            self.execute_export(df, export, directory)
            print('completed -', export, end=' ||| ')

class excel_exporter():
    def __init__(self, df, benchmarks,  d1, d2, d2_7, custom_tab):
        """
        df - dataframe to export to excel --> single advert / order
        """
        self.df = df
        self.benchmarks = benchmarks
        self.d1 = d1
        self.d2 = d2
        self.d2_7 = d2_7

        self.creative_types = (
            'traffic driver',
            'interactive non video',
            'branded driver',
            'video',
            'interactive video',
            'no match'
        )

        if custom_tab is not None:
            self.custom_tab = custom_tab

    def return_benchmark(self, KPI, placement, source):
        """
        returns benchmarks based upon args
        """
        df_bm = self.benchmarks
        sb1 = df_bm['KPI'] == KPI
        sb2 = df_bm['placement'] == placement
        sb3 = df_bm['source'] == source

        ### NEED TO FIX BMs
        try:
            if (KPI == 'VID' or KPI == 'IR') and source == '3P' :
                return df_bm[(sb1) & (sb2) & (df_bm['source'] == 'DFP')]['BM (%)'].values[0]
            else:
                return df_bm[(sb1) & (sb2) & (sb3)]['BM (%)'].values[0]
        except:
            return 0

    def metric_calcs(self, df):
        column_renaming = {
            "DFP Creative ID Impressions":"DFP server imps",
            "DFP Creative ID Clicks":"DFP clicks",
            "Normalized 3P Impressions":"3P imps",
            "Normalized 3P Clicks": "3P clicks",
            "Ad server Active View viewable impressions":"DFP Viewable imps",
            "Ad server downloaded impressions":"DFP downloaded imps"
        }

        metric_dict = {
            'DFP CTR %': ('DFP clicks', 'DFP server imps'),
            '3P CTR %': ('3P clicks', '3P imps'),
            'DFP view %': ('DFP Viewable imps', 'DFP server imps'),
            'DFP VSR %': ('result_5', 'DFP server imps'),
            '3P VSR %': ('result_5', '3P imps'),
            'VCR 75 %': ('result_75', 'result_5'),
            'DFP IR %': ('int sessions', 'DFP clicks', 'DFP server imps'),
            '3P IR %': ('int sessions', '3P clicks', '3P imps')
        }

        df = df.rename(index=str, columns=column_renaming)

        for key in self.instructions[self.tab]['display']:
            if "BM" in key:
                continue
            elif 'IR' in key:
                md = metric_dict[key]
                df[key] = ((df[md[0]] + df[md[1]]) / df[md[2]]) * 100
            else:
                md = metric_dict[key]
                df[key] = (df[md[0]] / df[md[1]]) * 100

        return df

    def apply_benchmarks(self, df):
        """
        """
        BM_dict = {
            'CTR' : ('DFP CTR BM', '3P CTR BM'),
            'VID' : ('DFP VSR BM'),
            'IR'  : ('DFP IR BM')
        }

        BM_CTR_DFP, BM_CTR_3P = [], []
        BM_VSR_DFP, BM_IR_DFP = [], []
        for i in range(len(df)):
            placement = df.iloc[i]['placement']
            BM_CTR_DFP.append(self.return_benchmark('CTR', placement, 'DFP'))
            BM_CTR_3P.append(self.return_benchmark('CTR', placement, '3P'))
            BM_VSR_DFP.append(self.return_benchmark('VID', placement, 'DFP'))
            BM_IR_DFP.append(self.return_benchmark('IR', placement, 'DFP'))
        df[BM_dict['CTR'][0]] = BM_CTR_DFP
        df[BM_dict['CTR'][1]] = BM_CTR_3P
        df['DFP VSR BM'] = BM_VSR_DFP
        df['DFP IR BM'] = BM_IR_DFP
        return df

    def trim_columns(self, df):
        """
        """
        col1 = self.instructions[self.tab]['groupby']
        col2 = self.instructions[self.tab]['display']
        df = df[list(col1) + ["DFP server imps", "3P imps"] + list(col2)]
        df = df.sort_values('DFP server imps', ascending=False)
        return df

    def create_sheet(self):
        """
        """
        pd.DataFrame().to_excel(self.writer, index=False,
                                sheet_name=self.tab, startrow=1, startcol=1)
        self.worksheet = self.writer.sheets[self.tab]

        self.worksheet.write(0,1, 'Week to date')
        self.worksheet.write(1,1, str(self.d2_7) + ' -> ' + str(self.d2))

        self.worksheet.write(0,15, 'Cumulative to date')
        self.worksheet.write(1,15, self.ctd_min_date + ' -> ' + str(self.d2))

        self.worksheet.set_column(14, 14, 1, self.divider)

    def write_excel(self, df, row, col):
        """
        """
        def number_formatting(x):
            if isinstance(x, float) and x>5:
                return format(int(x), ',')
            elif isinstance(x, float):
                return round(x, 2)
            else:
                return x

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        df = df.applymap(number_formatting)
        df.to_excel(self.writer, index=False, sheet_name=self.tab,
            startrow=row, startcol=col)

    def conditional_format(self, df, row, col):
        """
        """
        def paint_it(col, KPI, source):
            """
            applies the format by looping over placement
            """
            for i in range(len(df)):
                placement = df.iloc[i]['placement']
                self.worksheet.conditional_format(row+1+i, col, row+1+i, col,
                    {'type': 'cell',
                    'criteria': 'less than',
                    'value': self.return_benchmark(KPI, placement, source) * 0.75,
                    'format': self.f_red_dark})

                self.worksheet.conditional_format(row+1+i, col, row+1+i, col,
                    {'type': 'cell',
                    'criteria': 'between',
                    'minimum': self.return_benchmark(KPI, placement, source) * 0.75,
                    'maximum': self.return_benchmark(KPI, placement, source),
                    'format': self.f_red_light})

                self.worksheet.conditional_format(row+1+i, col, row+1+i, col,
                    {'type': 'cell',
                    'criteria': 'between',
                    'minimum': self.return_benchmark(KPI, placement, source),
                    'maximum': self.return_benchmark(KPI, placement, source) * 1.25,
                    'format': self.f_green_light})

                self.worksheet.conditional_format(row+1+i, col, row+1+i, col,
                    {'type': 'cell',
                    'criteria': 'greater than',
                    'value': self.return_benchmark(KPI, placement, source) * 1.25,
                    'format': self.f_green_dark})

        CTR_DFP = 5 + col
        CTR_3P = 6 + col
        VSR_DFP = 7 + col
        IR_DFP = 8 + col

        formatting_order = [
            (CTR_DFP, 'CTR', 'DFP'),
            (CTR_3P, 'CTR', '3P'),
            (VSR_DFP, 'VID', 'DFP'),
            (IR_DFP, 'IR', 'DFP')
        ]

        for order in formatting_order:
            paint_it(order[0], order[1], order[2])

    def execute_export(self, directory='/users/csaunders/Desktop/excel_export_test/'):
        """
        export -
        self.instructions used to instruct how to groupby data, as well as
        inform following methods on how to operate:
            + metric_calcs
            + apply_benchmarks
            + trim_columns
        """
        self.instructions = {
            'producer': {
                'groupby': ['site', 'creative.type', 'placement'],
                'display': ['DFP CTR %', '3P CTR %', 'DFP VSR %', 'DFP IR %',
                            'DFP view %', 'DFP CTR BM', '3P CTR BM',
                            'DFP VSR BM', 'DFP IR BM']
            },
            'Custom': {
                'groupby': ['none'],
                'display': ['DFP CTR %', '3P CTR %', 'DFP VSR %', 'DFP IR %',
                            'DFP view %', 'DFP CTR BM', '3P CTR BM',
                            'DFP VSR BM', 'DFP IR BM']
            },
            'creative': {
                'groupby': ['site', 'placement', 'Creative', 'creative.type'],
                'display': ['DFP CTR %', '3P CTR %', 'DFP VSR %', '3P VSR %',
                            'VCR 75 %', 'DFP IR %', '3P IR %', 'DFP view %']
            },
            'line item': {
                'groupby': ['site', 'Line item', 'Creative', 'creative.type'],
                'display': ['DFP CTR %', '3P CTR %', 'DFP VSR %', '3P VSR %',
                            'VCR 75 %', 'DFP IR %', '3P IR %', 'DFP view %']
            }
        }


        # initialize xlsxwriter and set book formats
        advert = set(self.df['Advertiser']).pop()
        order = set(self.df['Order']).pop()
        file_name = advert + ' ' + order + '.xlsx'
        self.writer = pd.ExcelWriter(directory + file_name, engine='xlsxwriter')

        self.f_red_dark = self.writer.book.add_format({'bg_color': '#f79494',
            'font_color': '#9C0006'})
        self.f_red_light= self.writer.book.add_format({'bg_color': '#FFC7CE',
            'font_color': '#9C0006'})

        self.f_green_light = self.writer.book.add_format({'bg_color': '#C6EFCE',
            'font_color': '#019117'})
        self.f_green_dark = self.writer.book.add_format({'bg_color': '#4bd164',
            'font_color': '#019117'})


        self.divider = self.writer.book.add_format({'bg_color': '#000000'})

        # create week to date (wtd) and cumulative to date (ctd)
        self.wtd = self.df[self.df['Date'] >= self.d2_7]
        self.ctd = self.df[self.df['Date'] >= self.d1]
        self.ctd_min_date = self.df['Date'].min().strftime("%Y-%m-%d")

        ################# Producer tab #########################################
        self.tab = 'producer'
        self.create_sheet()
        row = 3
        colw = 0
        colc = 15

        for creative_type in self.creative_types:
            self.creative_type = creative_type
            groupbys = self.instructions[self.tab]['groupby']

            self.wtd_prod = self.wtd[self.wtd['creative.type'] == self.creative_type].groupby(
                groupbys, as_index=False).sum()
            if len(self.wtd_prod) > 0:
                self.wtd_prod = self.metric_calcs(self.wtd_prod)
                self.wtd_prod = self.apply_benchmarks(self.wtd_prod)
                self.wtd_prod = self.trim_columns(self.wtd_prod)
                self.write_excel(self.wtd_prod, row, colw)
                self.conditional_format(self.wtd_prod, row, colw)

            self.ctd_prod = self.ctd[self.ctd['creative.type'] == self.creative_type].groupby(
                groupbys, as_index=False).sum()
            if len(self.ctd_prod) > 0:
                self.ctd_prod = self.metric_calcs(self.ctd_prod)
                self.ctd_prod = self.apply_benchmarks(self.ctd_prod)
                self.ctd_prod = self.trim_columns(self.ctd_prod)
                self.write_excel(self.ctd_prod, row, colc)
                self.conditional_format(self.ctd_prod, row, colc)

            if len(self.wtd_prod) > 0 or len(self.ctd_prod) > 0:
                row += max(len(self.wtd_prod), len(self.ctd_prod)) + 2

        ################# Custom tab #########################################
        if self.custom_tab is not None:
            self.tab = 'Custom'
            self.create_sheet()
            row = 3
            colw = 0
            colc = 15

            groupbys = [i for i in self.custom_tab.columns if 'Header' not in i]
            headers = [i for i in self.custom_tab.columns if 'Header' in i]
            custom_columns = (headers +
                             ["DFP server imps", "3P imps"] +
                             ['DFP CTR %', '3P CTR %', 'DFP VSR %',
                              'DFP IR %', 'DFP view %']
            )

            self.wtd_prod = pd.merge(self.wtd, self.custom_tab, on=groupbys, how='left')
            self.wtd_prod = self.wtd_prod.groupby(headers, as_index=False).sum()
            self.wtd_prod = self.metric_calcs(self.wtd_prod)
            self.wtd_prod = self.wtd_prod[custom_columns]
            self.write_excel(self.wtd_prod, row, colw)

            self.ctd_prod = pd.merge(self.ctd, self.custom_tab, on=groupbys, how='left')
            self.ctd_prod = self.ctd_prod.groupby(headers, as_index=False).sum()
            self.ctd_prod = self.metric_calcs(self.ctd_prod)
            self.ctd_prod = self.ctd_prod[custom_columns]
            self.write_excel(self.ctd_prod, row, colc)


        ################# Creative tab #########################################
        self.tab = 'creative'
        self.create_sheet()
        row = 3
        colw = 0
        colc = 15

        for creative_type in self.creative_types:
            self.creative_type = creative_type
            groupbys = self.instructions[self.tab]['groupby']

            self.wtd_prod = self.wtd[self.wtd['creative.type'] == self.creative_type].groupby(
                groupbys, as_index=False).sum()
            if len(self.wtd_prod) > 0:
                self.wtd_prod = self.metric_calcs(self.wtd_prod)
                # self.wtd_prod = self.apply_benchmarks(self.wtd_prod)
                self.wtd_prod = self.trim_columns(self.wtd_prod)
                self.write_excel(self.wtd_prod, row, colw)
                # self.conditional_format(self.wtd_prod, row, colw)

            self.ctd_prod = self.ctd[self.ctd['creative.type'] == self.creative_type].groupby(
                groupbys, as_index=False).sum()
            if len(self.ctd_prod) > 0:
                self.ctd_prod = self.metric_calcs(self.ctd_prod)
                # self.ctd_prod = self.apply_benchmarks(self.ctd_prod)
                self.ctd_prod = self.trim_columns(self.ctd_prod)
                self.write_excel(self.ctd_prod, row, colc)
                # self.conditional_format(self.ctd_prod, row, colc)

            if len(self.wtd_prod) > 0 or len(self.ctd_prod) > 0:
                row += max(len(self.wtd_prod), len(self.ctd_prod)) + 2


        ################ Line item tab ########################################
        self.tab = 'line item'
        self.create_sheet()
        row = 3
        colw = 0
        colc = 15

        for creative_type in self.creative_types:
            self.creative_type = creative_type
            groupbys = self.instructions[self.tab]['groupby']

            self.wtd_prod = self.wtd[self.wtd['creative.type'] == self.creative_type].groupby(
                groupbys, as_index=False).sum()
            if len(self.wtd_prod) > 0:
                self.wtd_prod = self.metric_calcs(self.wtd_prod)
                # self.wtd_prod = self.apply_benchmarks(self.wtd_prod)
                self.wtd_prod = self.trim_columns(self.wtd_prod)
                self.write_excel(self.wtd_prod, row, colw)
                # self.conditional_format(self.wtd_prod, row, colw)

            self.ctd_prod = self.ctd[self.ctd['creative.type'] == self.creative_type].groupby(
                groupbys, as_index=False).sum()
            if len(self.ctd_prod) > 0:
                self.ctd_prod = self.metric_calcs(self.ctd_prod)
                # self.ctd_prod = self.apply_benchmarks(self.ctd_prod)
                self.ctd_prod = self.trim_columns(self.ctd_prod)
                self.write_excel(self.ctd_prod, row, colc)
                # self.conditional_format(self.ctd_prod, row, colc)

            if len(self.wtd_prod) > 0 or len(self.ctd_prod) > 0:
                row += max(len(self.wtd_prod), len(self.ctd_prod)) + 2

        ################# data tab and save #####################################
        self.df.sort_values('Date', ascending=False).to_excel(
            self.writer, index=False, sheet_name='data')
        self.writer.save()

################################## DF reducer ##################################
class dataframe_reducer():
    """
    class that does everything for the dashboard-lite
    """
    def __init__(self, df):
        self.df = df

    def summarize_advert_orders(self):
        df_storage = {}
        for advert in set(self.df['Advertiser']):
            sb1 = self.df['Advertiser'] == advert
            for order in set(self.df[sb1]['Order']):
                sb2 = self.df['Order'] == order
                dmax = self.df[(sb1) & (sb2)]['Date'].max()
                imp_sum = self.df[(sb1) & (sb2)]['DFP Creative ID Impressions'].sum()

                df_storage.setdefault('Advertiser', []).append(advert)
                df_storage.setdefault('Order', []).append(order)
                df_storage.setdefault('Max date', []).append(dmax)
                df_storage.setdefault('total DFP imp', []).append(imp_sum)

        df = pd.DataFrame(df_storage)
        df['Max date'] = pd.to_datetime(df['Max date'])
        self.summarized = df

    def reduce(self, min_date, min_impressions):
        """
        min_date - date from which inactivity is ignore
        min_impressions - any order with less than this amount is ignored
        """
        column_keep = [
            'Date', 'Advertiser', 'Order',
            'Line item ID', 'Line item',
            'Creative', 'Creative ID',
            'Normalized 3P Impressions', 'Normalized 3P Clicks',
            'DFP Creative ID Impressions', 'DFP Creative ID Clicks',
            'Ad server Active View viewable impressions',
            'placement',
            'result_5', 'result_75', 'int sessions', 'interactions',
            'creative.type', 'adunit', 'site',
            'device']

        in_range = self.summarized[(self.summarized['Max date'] > min_date) &
                                   (self.summarized['total DFP imp'] > min_impressions)].copy()
        in_range['keep'] = 'keep'

        dft = pd.merge(self.df, in_range, on=('Advertiser', 'Order'), how='outer')
        dft = dft[dft['keep'].notnull()]
        dft = dft[column_keep]
        self.reduced = dft

################################## Dashboard ###################################
class dashboard_control():
    """
    class that does everything for the dashboard-lite
    """

    def __init__(self, df):
        """
        + initialize class with dataframe
        + initiatlive all ipywidgets
        """
        self.df = df
        self.Advertisers = sorted(list(set(self.df['Advertiser'])))

        self.advert_dropdown = ipywidgets.Dropdown(
            options=self.Advertisers,
            value=self.Advertisers[0],
            disabled=False)

        self.order_dropdown = ipywidgets.Dropdown(
            options=[],
            value=None,
            disabled=False)

        self.PL_dropdown = ipywidgets.Dropdown(
            options=['placement', 'Line item'],
            value='placement',
            disabled=False)

        self.creative_dropdown = ipywidgets.Dropdown(
            options=["No creatives", "All creatives", "Engage creatives", "Marquee creatives", "Inline creatives"],
            value="No creatives",
            disabled=False)

        self.display_dropdown = ipywidgets.Dropdown(
            options=["producer", 'raw values', 'metrics DFP', 'metrics 3P', 'metrics all'],
            value='producer',
            disabled=False)

        self.d1_DatePicker = ipywidgets.DatePicker(disabled=False)
        self.d2_DatePicker = ipywidgets.DatePicker(disabled=False)

    def execute_dashboard(self):
        """
        + initialize class with dataframe
        + initiatlive all ipywidgets
        """

        def update_order_dropdown(change):
            """
            sb - Series Boolean variable
            """
            sb1 = self.df['Advertiser'] == self.advert_dropdown.value
            x1 = self.df[sb1]['Order']
            self.order_dropdown.options = list(set(x1))
            self.order_dropdown.value = self.order_dropdown.options[0]

        def update_dates(change):
            """
            sb - Series Boolean variable
            """
            sb1 = self.df['Advertiser'] == self.advert_dropdown.value
            sb2 = self.df['Order'] == self.order_dropdown.value
            self.d1_DatePicker.value = (self.df[(sb1) & (sb2)]['Date'].min()).date()
            self.d2_DatePicker.value = (self.df[(sb1) & (sb2)]['Date'].max()).date()

            # def update_selection(self, change):
            #     x1 = self.df['Advertiser'] == self.advert_dropdown.value
            #     x2 = self.df['Order'] == self.order_dropdown.value
            #     x3 = self.df['Date'] >= self.d1_DatePicker.value
            #     x4 = self.df['Date'] <= self.d2_DatePicker.value
            #     dfx = self.df[(x1) & (x2) & (x3) & (x4)]
            #     self.selection_list.options = sorted(list(set(dfx['placement']))) + ['all', 'all & creatives', 'engage only', 'marquee only', 'inline only']
            #     self.selection_list.value = 'all'


        self.advert_dropdown.observe(update_order_dropdown, names='value')
        self.order_dropdown.observe(update_dates, names='value')
        # self.order_dropdown.observe(self.update_selection, names='value')

        master_interact = ipywidgets.interactive(self.display_dataframe,
                advert=self.advert_dropdown, order=self.order_dropdown,
                d1_date = self.d1_DatePicker, d2_date = self.d2_DatePicker,
                PL_selection = self.PL_dropdown,
                creative = self.creative_dropdown,
                display_cols = self.display_dropdown)
        return master_interact

    def display_dataframe(self, **kwargs):
        def metric_calculations(df):

            column_renaming = {
                "DFP Creative ID Impressions":"DFP server imps",
                "DFP Creative ID Clicks":"DFP clicks",
                "Normalized 3P Impressions":"3P imps",
                "Normalized 3P Clicks": "3P clicks",
                "Ad server Active View viewable impressions":"DFP Viewable imps",
                "Ad server downloaded impressions":"DFP downloaded imps",
                "creative.type":"creative.type"
            }

            df = df.rename(index=str, columns=column_renaming)

            metric_dict = {
                'DFP CTR %': ('DFP clicks', 'DFP server imps'),
                '3P CTR %': ('3P clicks', '3P imps'),
                'DFP view %': ('DFP Viewable imps', 'DFP server imps'),
                'DFP VSR %': ('result_5', 'DFP server imps'),
                '3P VSR %': ('result_5', '3P imps'),
                'VCR 75 %': ('result_75', 'result_5'),
                'DFP IR %': ('int sessions', 'DFP clicks', 'DFP server imps'),
                '3P IR %': ('int sessions', '3P clicks', '3P imps')
            }


            for key in metric_dict.keys():
                if 'IR' in key:
                    md = metric_dict[key]
                    df[key] = ((df[md[0]] + df[md[1]]) / df[md[2]]) * 100
                else:
                    md = metric_dict[key]
                    df[key] = (df[md[0]] / df[md[1]]) * 100
            return df

        def final_formatting(dfx, display_options):
            dfx = dfx.groupby(groupings).sum()

            ## add in benchmarks here



            columns = base_cols + display_options[display_cols]
            dfx = dfx.replace([np.inf, -np.inf], np.nan)
            dfx = dfx[columns].fillna('-')

            def number_formatting(x):
                if isinstance(x, float) and x>1:
                    return format(int(x), ',')
                elif isinstance(x, float):
                    return round(x, 3)
                elif isinstance(x, int) and x>999:
                    return format(int(x), ',')
                else:
                    return x
            dfx = dfx.sort_values(base_cols[0], ascending=False)
            display(dfx.applymap(number_formatting))

        display_options = {
            'producer':
                ['DFP CTR %', '3P CTR %', 'DFP VSR %', 'DFP IR %', 'DFP view %'],
            'raw values':
                ['DFP clicks', '3P clicks', 'DFP Viewable imps',
                 'result_5', 'result_75', 'int sessions', 'interactions'],
            'metrics DFP':
                ['DFP CTR %', 'DFP VSR %', 'DFP IR %', 'DFP view %'],
            'metrics 3P':
                ['3P CTR %', '3P VSR %', '3P IR %'],
            'metrics all':
                ['DFP CTR %', '3P CTR %', 'DFP VSR %', '3P VSR %', 'VCR 75 %',
                 'DFP IR %', '3P IR %','DFP view %']
        }

        base_cols = ['DFP server imps', '3P imps']

        advert = self.advert_dropdown.value
        order = self.order_dropdown.value
        d1_date = self.d1_DatePicker.value
        d2_date = self.d2_DatePicker.value

        PL_selection = self.PL_dropdown.value
        creative = self.creative_dropdown.value
        display_cols = self.display_dropdown.value

        x1 = self.df['Advertiser'] == advert
        x2 = self.df['Order'] == order
        x3 = self.df['Date'] >= d1_date
        x4 = self.df['Date'] <= d2_date
        dfx = self.df[(x1) & (x2) & (x3) & (x4)]

        if len(dfx) < 1:
            print('no data')
            return

        if d1_date < dfx['Date'].min().date():
            print('\033[1m'+'WARNING - data does not exist before: '+'\033[0m',
                str(dfx['Date'].min().date()))



        #define groupbys
        creative_dict_lookup = {"All creatives": ['engage', 'marquee', 'inline'],
                                "Engage creatives": ['engage'],
                                "Marquee creatives":['marquee'],
                                "Inline creatives": ['inline']}

        if creative == "No creatives":
            groupings = ['site', 'creative.type', PL_selection]

        else:
            sb1 = dfx['adunit'].apply(lambda x: x in creative_dict_lookup[creative])
            dfx = dfx[sb1]
            groupings = ['site', 'creative.type', PL_selection, 'Creative']

        dfx = dfx.groupby(groupings, as_index=False).sum()
        dfx = metric_calculations(dfx)

        creative_types = ('traffic driver',
                          'interactive non video',
                          'branded driver',
                          'video',
                          'interactive video',
                          'no match')

        for creative_type in creative_types:
            dfxx = dfx[dfx['creative.type'] == creative_type]
            if len(dfxx) > 1:
                final_formatting(dfxx, display_options)

################################## Metric explorer #############################
class metric_explorer():
    """
    class for exploring metrics within the Dashboard_lite
    """
    def __init__(self, df):
        """
        + initialize class with dataframe
        + initiatlive all ipywidgets
        """
        self.df = df
        self.split_param = ' - * - '

        self.df['advert_order'] = self.df['Advertiser'] + self.split_param + self.df['Order']

        self.creative_types = ('no match', 'traffic driver',
            'interactive non video', 'branded driver', 'video',
            'interactive video'
        )

        self.AO_multiple = ipywidgets.SelectMultiple(
            options=sorted(set(self.df['advert_order'])),
            value=[],
            rows=3,
            description='Advert-order',
            disabled=False,
            layout=ipywidgets.Layout(width='50%', height='280px')
        )

        self.site_dropdown = ipywidgets.Dropdown(
            options = ['qz', 'wrk', 'zty'],
            value='qz',
            disabled=False
        )

        self.placement_dropdown = ipywidgets.Dropdown(
            options = ['engage mobile', 'engage desktop',
                       'marquee mobile', 'marquee desktop',
                       'inline mobile', 'inline desktop'],
            value='engage mobile',
            disabled=False
        )

        self.creative_type_dropdown = ipywidgets.Dropdown(
            options=self.creative_types,
            value=self.creative_types[0],
            disabled=False
        )

        self.metric_measurement = ipywidgets.Dropdown(
            options=['DFP CTR', '3P CTR', 'Viewability', 'VSR', 'IR'],
            value='DFP CTR',
            disabled=False
        )

        self.d1_DatePicker = ipywidgets.DatePicker(disabled=False)
        self.d2_DatePicker = ipywidgets.DatePicker(disabled=False)

        self.aggregate_checkbox = ipywidgets.Checkbox(
            value=False,
            description='Display aggregate',
            disabled=False
        )

        self.button = ipywidgets.Button(description="CHART IT !",
            layout=ipywidgets.Layout(width='100%', height='55px')
        )

        self.left_box = ipywidgets.VBox(
            [self.site_dropdown, self.placement_dropdown,
             self.creative_type_dropdown, self.metric_measurement,
             self.d1_DatePicker, self.d2_DatePicker,
             self.aggregate_checkbox, self.button]
        )

        self.display = ipywidgets.HBox([self.left_box, self.AO_multiple])

    def update_AO_multiple(self, change):
        """
        update advert_order multiple select widget
        """
        sb1 = self.df['site'] == self.site_dropdown.value
        sb2 = self.df['placement'] == self.placement_dropdown.value
        sb3 = self.df['creative.type'] == self.creative_type_dropdown.value
        #sb4 = self.df['site'] == self.site_dropdown.value

        x1 = self.df[(sb1) & (sb2) & (sb3)]
        self.AO_multiple.options = list(set(x1['advert_order']))

    def display_dashboard(self):
        """
        display the metric explorer dashboard
        """

        self.site_dropdown.observe(self.update_AO_multiple, names='value')
        self.placement_dropdown.observe(self.update_AO_multiple, names='value')
        self.creative_type_dropdown.observe(self.update_AO_multiple, names='value')
        #self.metric_measurement.observe(update_AO_multiple, names='value')

        self.button.on_click(self.print_button)

        return self.display

    def print_button(self, change):
        self.create_chart_dataset()
        self.graph_metrics()
        print('you pressed the button!')


    def create_chart_dataset(self):
        metric_lookup = {
            'DFP CTR': ('DFP Creative ID Clicks', 'DFP Creative ID Impressions'),
            '3P CTR': ('3P Creative ID Clicks', '3P Creative ID Impressions'),
            'Viewability': ('Ad server Active View viewable impressions', 'DFP Creative ID Impressions'),
            'VSR': ('result_5', 'DFP Creative ID Impressions'),
            'IR': ('int sessions','DFP Creative ID Impressions')
        }

        split_param = ' - * - '
        AO = self.AO_multiple.value
        creative_type = self.creative_type_dropdown.value
        placement = self.placement_dropdown.value
        metric = self.metric_measurement.value

        rolling = 4
        df = pd.DataFrame()

        agg = self.df[(self.df['creative.type'] == creative_type) &
                      (self.df['placement'] == placement)].groupby('Date').sum()
        num = metric_lookup[metric][0]
        dem = metric_lookup[metric][1]
        agg = agg[[num, dem]]
        agg = agg.rolling(rolling).sum()

        agg[metric] = (agg[num] / agg[dem]) * 100
        agg = agg[[metric]]
        agg.columns = ['site']

        for i in AO:
            advert = i.split(split_param)[0]
            order = i.split(split_param)[1]
            sb1 = self.df['Advertiser'] == advert
            sb2 = self.df['Order'] == order
            sb3 = self.df['creative.type'] == creative_type
            sb4 = self.df['placement'] == placement

            dfx = self.df[(sb1) & (sb2) & (sb3) & ()]
            dfx = dfx.groupby('Date').sum()
            num = metric_lookup[metric][0]
            dem = metric_lookup[metric][1]
            dfx = dfx[[num, dem]]
            dfx = dfx.rolling(rolling).sum()

            dfx[metric] = (dfx[num] / dfx[dem]) * 100
            dfx = dfx[[metric]]
            dfx.columns = [advert]
            if len(df) == 0:
                df = dfx
            else:
                df = pd.merge(df, dfx, left_index=True, right_index=True, how='outer')

        df = pd.merge(df, agg, left_index=True, right_index=True, how='outer')


        dff = pd.DataFrame()
        df = df.reset_index()

        for col in df.columns:
            if col != 'Date':
                dfx = df[['Date', col]]
                dfx['client'] = col
                dfx.columns = ['Date', 'value', 'client']
                dff = dff.append(dfx)
        self.chart_dataset = dff


    def graph_metrics(self):
        from bokeh.plotting import figure, output_file, show
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.io import output_notebook, push_notebook, show

        dff['Date'] = pd.to_datetime(dff['Date'])

        output_notebook()


        hover = HoverTool(names=['circle'], tooltips=[
            ("Date", "@Date"),
            ("value", "$y"),
            ("client", "@client"),
        ])
        TOOLS = [hover]

        source = ColumnDataSource(dff)

        p = figure(width=1500, height=700, x_axis_type="datetime", y_range=(0,round(int(dff['value'].max() * 1.10))), tools=TOOLS)
        p.circle('Date', 'value', source=source, name='circle')

        for client in set(dff['client']):
            x1 = dff[dff['client'] == client]
            my_plot = p.line(x1['Date'], x1['value'])


        show(p)


