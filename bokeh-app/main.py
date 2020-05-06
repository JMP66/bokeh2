#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np

from os.path import dirname, join
from bokeh.io import show, output_notebook,output_file, show, save, curdoc, output_notebook, export_png
from bokeh.plotting import figure, output_file, show,save
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column, row, widgetbox
from bokeh.models import  HoverTool, LinearColorMapper,TextInput,Label,LabelSet,Title,CustomJS,DataTable,        Slider, Div,RangeSlider, Button,RadioGroup,LinearAxis, Range1d, ColumnDataSource, Paragraph,Select, TableColumn

from bokeh.tile_providers import CARTODBPOSITRON, get_provider,OSM, STAMEN_TERRAIN

#colors for each borough
colors=['#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']

#Load data
mid_dyn = pd.read_csv(join(dirname(__file__), "data/MiddleSchools_2006-2018_clean.csv"))
mid_stat = pd.read_csv(join(dirname(__file__), "data/MiddleSchools_2018_clean.csv"))


nums =['female_rate', 'male_rate', 'asian_rate', 'black_rate', 'hispanic_rate',
       'other_rate', 'white_rate', 'disabilities_rate', 'ell_rate',
       'poverty_rate', 'total_schooldays', 'presence_rate', 'absense_rate',
       'release_rate', 'mean_score_math', 'mean_score_ela', 'diversity_index','crime_rate','avg_rent_per_sqft']

for num in nums:
    mid_stat[num] = round(mid_stat[num],1)
    if num not in ['crime_rate','avg_rent_per_sqft']:
        mid_dyn[num] = round(mid_dyn[num],1)
        
    


# In[8]:


def geographic_to_web_mercator(x_lon, y_lat):     
    if abs(x_lon) <= 180 and abs(y_lat) < 90:          
        num = x_lon * 0.017453292519943295         
        x = 6378137.0 * num         
        a = y_lat * 0.017453292519943295          
        x_mercator = x         
        y_mercator = 3189068.5 * np.log((1.0 + np.sin(a)) / (1.0 - np.sin(a)))         
        
        return x_mercator, y_mercator   


# In[9]:


#Get data from csv to lists
def get_data(school):
    school_data = mid_dyn[mid_dyn['dbn']==school]
    source = ColumnDataSource(school_data)
    
    return source, school_data


# In[10]:


def create_slider(plot, startYear, endYear):
    callback = CustomJS(args=dict(plot=plot), code="""
    var a = cb_obj.value;
    plot.x_range.start = a[0];
    plot.x_range.end = a[1];
    """)

    range_slider = RangeSlider(start=startYear, end=endYear,value=(startYear, endYear), step=1, width= 500, title="Year Range")
    range_slider.js_on_change('value', callback)

    layout = column(plot,column(range_slider))
    return layout


# In[14]:


def create_plot():
    
    colors=['#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']
    radio_idx = radio_group.active
    school = text_input.value
    
    variables = ['Etnicities','Gender','Mean Score']

    text = mid_stat[mid_stat['dbn']==school]['overview'].iloc[0]
    data =  mid_stat[mid_stat['dbn']==school]         
    src, school_data = get_data(school)
          
    if radio_idx == 0:     
        plot = figure(plot_width = 500, plot_height = 400, 
        toolbar_location=None,
        x_axis_label = 'Year', y_axis_label = '% Etnicity')

        races = ['asian_rate', 'black_rate', 'hispanic_rate', 'white_rate']
        race_title =['Asian', 'Black', 'Hispanic',  'White']
        colors1 = colors[1:]
        
        for (race,tit,color) in zip(races,race_title,colors1):
            line=plot.line('year', race, line_width=2, line_color=color, source=src,legend_label=tit)
            plot.circle('year', race, fill_color=color, line_color=color, size=8, source=src)
            hover = HoverTool(renderers=[line])
            hover.tooltips=[
            ('Year', '@year'),
            (tit, '@'+race+'{1.1} %')
            ]
            plot.add_tools(hover)
            
        plot.legend.location ='top_left' 
       #plot.add_layout(Title(text= '{} School \n'.format(level), text_font_style="italic",text_font_size="14pt", align='center'), 'above')
        plot.add_layout(Title(text=school_data['school_name'].unique()[0], text_font_size="16pt",align='center'), 'above',)
        
        #plot.title.align ='center'
        #plot.title.text_font_size = "18px"
        
    elif radio_idx == 1:
        
        plot = figure(plot_width = 500, plot_height = 400, 
        toolbar_location=None,
        x_axis_label = 'Year', y_axis_label = '% Gender')

        genders = ['female_rate','male_rate']
        gender_title =['% Female','% Male']
        colors2 = [colors[2]]+[colors[4]]
        for (gender,tit,color) in zip(genders,gender_title,colors2):
            line=plot.line('year', gender, line_width=2, line_color=color, source=src,legend_label=tit)
            plot.circle('year', gender, fill_color=color, line_color=color, size=8, source=src)
            hover = HoverTool(renderers=[line])
            hover.tooltips=[
            ('Year', '@year'),
            (tit, '@'+gender+'{1.1} %')
            ]
            plot.add_tools(hover)
       
        plot.legend.location ='top_left' 
       # plot.add_layout(Title(text= '{} School \n'.format(level), text_font_style="italic",text_font_size="14pt", align='center'), 'above')
        plot.add_layout(Title(text=school_data['school_name'].unique()[0], text_font_size="16pt",align='center'), 'above',)
        
        
    elif radio_idx == 2:
        
        plot = figure(plot_width = 500, plot_height = 400, 
       toolbar_location=None, 
        x_axis_label = 'Year', y_axis_label = 'Mean Score')
        cols = ['mean_score_math', 'mean_score_ela']
        cols_tit =  ['Mean Math Score', 'Mean ELA Score']
        colors3 = [colors[2]]+[colors[4]]

        for (col,tit,color) in zip(cols,cols_tit,colors3):
            line=plot.line('year', col, line_width=2, line_color=color, source=src,legend_label=tit)
            plot.circle('year', col, fill_color=color, line_color=color, size=8, source=src)
            hover = HoverTool(renderers=[line])
            hover.tooltips=[
                ('Year', '@year'),
                (tit, '@'+col+'{1.1}')
            ]
            plot.add_tools(hover)
        
        plot.legend.location ='top_left' 
       #plot.add_layout(Title(text= '{} School \n'.format(level), text_font_style="italic",text_font_size="14pt", align='center'), 'above')
        plot.add_layout(Title(text=school_data['school_name'].unique()[0], text_font_size="16pt",align='center'), 'above',)
    
   
    #Add overview paragraph
    para = Div(text=text,
    width=400, height=400)
    
    cols=[ 'school_name',
     'category',
     'open_year',
     'borough',
     'neighborhood',
     'district',
     'address',
     'website',
     'total_enrollment',
     'female_rate',
     'male_rate',
     'diversity_index',
     'asian_rate',
     'black_rate',
     'hispanic_rate',
     'white_rate',
     'ell_rate',
     'poverty_rate',
     'total_schooldays',
     'presence_rate',
     'absense_rate',
     'mean_score_math',
     'mean_score_ela',
     'schoolday_duration',
     'uniform',
     'extendedday',
     'summersession',
     'weekendprogram',
     'electives',
     'activities',
     'sports',
     'pupil_teacher_ratio',
     'student_trust_score',
     'crime_rate',
     'avg_rent_per_sqft']


    col_name=[ 'Name',
     'Categpry',
     'Open year',
     'Borough',
     'Neighborhood',
     'District',
     'Address',
     'Website',
     'Enrollment',
     '% Female',
     '% Male',
     'Diversity index',
     '% Asian',
     '% Black',
     '% Hispanic',
     '% White',
     '% ELL',
     '% Supported',
     'Schooldays',
     '% Presence',
     '% Absense',
     'Mean math score',
     'Mean ELA score',
     'Schoolday',
     'Uniform',
     'Extended day',
     'Summer session',
     'Weekend program',
     'Electives',
     'Activities',
     'Sports',
     'Class size',
     'Satisfaction',
     'Crime rate',
     'Rent per sqft $']


    data_dict ={'columns': col_name, 'data': list(data[cols].iloc[0].values)}
    source = ColumnDataSource(data_dict) 
    
    columns = [
            TableColumn(field="columns", title='DBN: '+data['dbn'].iloc[0],width=100),
            TableColumn(field="data", title="",width=1000),
       
        ]
    table = DataTable(source=source, columns=columns, width=220, height=450, fit_columns=False,index_position=None) 
    
    #Get map
    x,y = geographic_to_web_mercator(data['lon'].iloc[0],data['lat'].iloc[0])
    tile_provider = get_provider(CARTODBPOSITRON)
    # range bounds supplied in web mercator coordinates
    m = figure(x_range=(x-500, x+500), y_range=(y-500, y+500),height=300,width=260, 
               x_axis_location=None, y_axis_location=None,toolbar_location='below',tools="pan,wheel_zoom,reset",active_scroll='auto')
    m.add_tile(tile_provider)
    square=m.circle(x=x,y=y,size=12, fill_color=colors[4], fill_alpha=1)
    tooltips = [('Name', data['school_name'].iloc[0]),('Address', data['address'].iloc[0])]
    m.add_tools(HoverTool(renderers=[square],tooltips=tooltips))


    return plot, para, m, table


# In[31]:


def update1(attr, old, new):
    plot,para,m,table = create_plot()
    layout.children[1].children[1]= create_slider(plot, 2006, 2018)
    
def update2(attr, old, new):
    plot,para,m,table = create_plot()
    layout.children[2].children[1] =  table
    layout.children[3].children[1] =  para
    layout.children[0].children[5] =  m
    
    
text_input = TextInput(value='01M034')
text_input.on_change('value',update1,update2)

div1 = Div(text="<b> Write School DBN </b>")

variables = ['Etnicities','Gender','Mean Score']
div2 = Div(text="<b> Choose variable </b>")              
radio_group = RadioGroup(labels=variables, active=1)
radio_group.on_change('active',update1,update2)

div3 = Div(text="<b> Location of School </b>")

div4 = Div(text="<b> Overview </b>")
plot,para,m,table = create_plot()
layout = create_slider(plot, 2006, 2018)

div5 = Div(text="<b> </b>")
div6 = Div(text="<b> </b>")
#Combine all controls to get in column
col1= column(div1,text_input,div2,radio_group,div3,m, width=260)
col2 = column(div6, layout, width=510)
col3 = column(div5,table, width=230)
col4 = column(div4, para, width=230)
#Layout
layout = row(col1,col2,col3,col4)

curdoc().add_root(layout)
curdoc().title = "NYC_map"

#output_file("details.html")
#save(layout)

show(layout)

