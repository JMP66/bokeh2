#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os.path import dirname, join
import pandas as pd
import numpy as np
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, output_file, show, save
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import OpenURL,ColumnDataSource, HoverTool, TextInput,Label,LabelSet,Title,TableColumn, DataTable
from bokeh.layouts import column, row, widgetbox
from bokeh.models import Slider, Div,RangeSlider, CustomJS,Button, RadioButtonGroup,CheckboxButtonGroup,CheckboxGroup,RadioGroup, ColumnDataSource, Paragraph,Select

from bokeh.plotting import figure, output_file, show
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
from bokeh.io import output_file, show, save, curdoc, output_notebook, export_png

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
        
    


# In[2]:


def geographic_to_web_mercator(x_lon, y_lat):     
    if abs(x_lon) <= 180 and abs(y_lat) < 90:          
        num = x_lon * 0.017453292519943295         
        x = 6378137.0 * num         
        a = y_lat * 0.017453292519943295          
        x_mercator = x         
        y_mercator = 3189068.5 * np.log((1.0 + np.sin(a)) / (1.0 - np.sin(a)))         
        
        return x_mercator, y_mercator  

#Ger mercator coordinates
mercx = []
mercy = []

for i in range(len(mid_stat)):
    x,y =geographic_to_web_mercator(mid_stat['lon'].iloc[i],mid_stat['lat'].iloc[i])
    mercx.append(x)
    mercy.append(y)
mid_stat['mercx'] = mercx
mid_stat['mercy'] = mercy


# In[3]:


def create_plot():        
   
    #Data
    data =  mid_stat.copy()
    
    #Borough 
    boro_val = select1.value
    boros = ['No Preference','Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']
       
    if boro_val != boros[0]:
        data = data[data['borough']==boro_val]
        
    #Mean Scores    
    math_range = slider21.value
    ela_range = slider22.value   

    data = data[(data['mean_score_math']>=math_range[0]) & (data['mean_score_math']<=math_range[1])]
    data =data[(data['mean_score_ela']>=ela_range[0]) & (data['mean_score_ela']<=ela_range[1])]   
        
    #Total Enrollment
    enroll_range = slider3.value  
    data =data[(data['total_enrollment']>=enroll_range[0]) & (data['total_enrollment']<=enroll_range[1])]   
    
    
    #Class size
    class_range = slider4.value  
    data =data[(data['pupil_teacher_ratio']>=class_range[0]) & (data['pupil_teacher_ratio']<=class_range[1])]   
    
    #Sports
    s = select51.value
    sports= ['No Preference','Dance and Fitness','Outdoor', 'Water Sports ','Martial Arts ','Racquet Sports','Ball Teamsports']

    if s == sports[1]:
        data = data[data['sports'].str.lower().str.contains('salsa|zumba|step|weight|yoga|step|dance|fitness|cheer|gym')]
    elif s == sports[2]:
        data = data[data['sports'].str.lower().str.contains('track|run|board|climb')]
    elif s == sports[3]:
        data = data[data['sports'].str.lower().str.contains('row|swim')]
    elif s == sports[4]:
        data = data[data['sports'].str.lower().str.contains('martial|karate|wrest|judo')]
    elif s == sports[5]:
        data = data[data['sports'].str.lower().str.contains('tennis|squash|ping')]
    elif s == sports[6]:
        data = data[data['sports'].str.lower().str.contains('lacrosse|ball|rugby|soccer|frisbee')]
        
    #Electives
    e = select52.value
    elect = ['No Preference','Technology','Creativity','Society']  
    
    if e == elect[1]:
        data = data[data['electives'].str.lower().str.contains('tech|coding|computer|web|movie|program|stem')]
    elif e == elect[2]:
        data = data[data['electives'].str.lower().str.contains('guitar|band|vocal|music|drama|instrument|danc|art|theat|writ|choir')]
    elif e == elect[3]:
        data = data[data['electives'].str.lower().str.contains('language|cult|social|leader|spanish')]


    #Cross off
    offer_idx = checkbox52.active
    offer = ['uniform', 'extendedday',
       'summersession', 'weekendprogram']
    offer_choice = [offer[idx] for idx in offer_idx]
    
    for off in offer_choice:
        data = data[data[off]==True]
    
    #Gender  
    male_range = slider61.value
    female_range = slider62.value   
    data = data[(data['male_rate']>=male_range[0]) & (data['male_rate']<=male_range[1])]
    data =data[(data['female_rate']>=female_range[0]) & (data['female_rate']<=female_range[1])]   
    
   #Etnicities
    div_range = slider7.value
    data = data[(data['diversity_index']>=div_range[0]) & (data['diversity_index']<=div_range[1])]
    
    #Rental
    rent_idx = radio_button81.active
   
    if rent_idx == 1:
        data = data[data['rent_level']<=1]
    elif rent_idx == 2:
        data = data[data['rent_level']<=2]
    elif rent_idx == 3:
        data = data[data['rent_level']>=3] 
        
    #Crime
    crime_idx = radio_button82.active
   
    if crime_idx == 1:
        data = data[data['crime_rate']<30]
    elif crime_idx == 2:
        data = data[(data['crime_rate']>=30) & (data['crime_rate']<60)]
    elif crime_idx == 3:
        data = data[data['crime_rate']>60]    
    
    
    #Make data dict and columnsource
    data_dict = data.to_dict('list')
    source = ColumnDataSource(data_dict) 
    
    w=80

    div0= Div(text="<b> Schools matching Preferences: {} </b>".format(len(data)),style={'font-size': '150%'})
    
    columns = [
            TableColumn(field="dbn", title="School DBN",width=w),
            TableColumn(field="mean_score_math", title="Math Score",width=w),
            TableColumn(field="mean_score_ela", title="ELA Score",width=w),
            TableColumn(field="total_enrollment", title="Enrollment",width=w),
            TableColumn(field="pupil_teacher_ratio", title="Class size",width=w),
            TableColumn(field="male_rate", title="% Male",width=w),
            TableColumn(field="female_rate", title="% Female",width=w),
            TableColumn(field="black_rate", title="% Black",width=w),
            TableColumn(field="white_rate", title="% White",width=w),
            TableColumn(field="asian_rate", title="% Asian",width=w),
            TableColumn(field="hispanic_rate", title="% Hispanic",width=w),
            TableColumn(field="avg_rent_per_sqft", title="Rent per sqft",width=w),
            TableColumn(field="crime_rate", title="Crime rate",width=w),
            TableColumn(field="sports", title="Sports",width=800),
            TableColumn(field="electives", title="Electives",width=800)
       
        ]
    data_table = DataTable(source=source, columns=columns, width=800, height=180, fit_columns=False)  
    
       
    #Get map
    NY = [-73.935242,40.730610,]
    x1,y1 = geographic_to_web_mercator(NY[0],NY[1])
    tile_provider = get_provider(CARTODBPOSITRON)
    
    # range bounds supplied in web mercator coordinates
    w = 40000
    m = figure(x_range=(x1-w, x1+w), y_range=(y1-w, y1+w),height=500,width=500, 
               x_axis_location=None, y_axis_location=None,toolbar_location='below',tools="pan,wheel_zoom,reset",active_scroll='auto')
    m.add_tile(tile_provider)
    circles = m.circle(x="mercx",y="mercy",size=8, source=source, fill_color="midnightblue", fill_alpha=1)
    tooltips = [('DBN','@dbn'),('Name', "@school_name"),('Address', "@address")]
    m.add_tools(HoverTool(renderers=[circles],tooltips=tooltips))
    
       
    return data_table, m,div0
    
    
    


# # SPORTS 
# 
# mid_stat['electives'] =mid_stat['electives'].astype(str)
# sports = []
# 
# for i in range(len(mid_stat)):
#     col = mid_stat.electives[i].split(', ')
#     
#     for i in range(len(col)):
#         sports.append(col[i])
#         
# sport,count = np.sort(np.unique(np.array(sports),return_counts=True))
# sports = pd.DataFrame()
# sports['sport'] = sport
# sports['sport'] = sports['sport'].str.lower()
# sports['count'] = count.astype(int)
# sports = sports.sort_values(by='count',ascending=False)

# In[6]:


def update(attr, old, new):
    table,m, div0 = create_plot()
    layout.children[2].children[0]= table
    layout.children[2].children[1]= m
    layout.children[0]= div0


div00= Div(text="<i> Sort and Select</i>")

# 1) Choose borough
div1 = Div(text="<b> Choose Borough</b>")
boros = ['No Preference']+list(mid_stat.borough.unique())
select1 = Select(options=boros, value=boros[0])
select1.on_change('value',update)

# 2) Mean Scores
div2 = Div(text="<b> Test Score Intervals </b>")
slider21 = RangeSlider(start=150, end=300, value=(150,300), step=1, title="Math Score")
slider21.on_change('value',update)
slider22 = RangeSlider(start=150, end=300, value=(150,300), step=1, title="ELA Score")
slider22.on_change('value',update)

# 3) Enrollment
div3 = Div(text="<b> Enrolled Students </b>")
slider3 = RangeSlider(start=94, end=2251, value=(94,2251), step=1, title="Enrolled Students")
slider3.on_change('value',update)

# 4) Class size
div4 = Div(text="<b> Class size </b>")
slider4= RangeSlider(start=3, end=26, value=(3,26), step=1, title="Class size")
slider4.on_change('value',update)


# 5) Choose Offers
div5 = Div(text="<b> Student Offers</b>")

sports_choice= ['No Preference','Dance and Fitness','Outdoor', 'Water Sports ','Martial Arts ','Racquet Sports','Ball Teamsports']
div51=Div(text="<i> Choose sport </i>")
select51 = Select(options=sports_choice,value=sports_choice[0])
select51.on_change('value',update)

elect_choice=  ['No Preference','Technology','Creativity','Society'] 
div52=Div(text="<i> Choose elective </i>")
select52 = Select(options=elect_choice,value=elect_choice[0])
select52.on_change('value',update)


div53 = Div(text="<i> Cross off</i>")
offer = ['uniform', 'extendedday',
       'summersession', 'weekendprogram']
checkbox52 = CheckboxGroup(labels=offer, active=[])
checkbox52.on_change('active',update)


# 6) Gender
div6 = Div(text="<b> Gender rate </b>")
slider61 = RangeSlider(start=0, end=100, value=(0,100), step=1, title="% Male")
slider61.on_change('value',update)
slider62 = RangeSlider(start=0, end=100, value=(0,100), step=1, title="% Female")
slider62.on_change('value',update)

# 7) Etnicity
div7 = Div(text="<b> Diverty Index </b>")
slider7 = RangeSlider(start=0, end=100, value=(0,100), step=1, title="Diversity Index")
slider7.on_change('value',update)

# 8) 
#Rent
div81 =Div(text="<b> Housing rent per sqft </b>")
radio_button81= RadioButtonGroup(
        labels=["No Preference","< 20 $", "20-30$", ">30 $"], active=0)
radio_button81.on_change('active',update)

#Crime
div82 =Div(text="<b>Felonies per 100.000 </b>")
radio_button82= RadioButtonGroup(
        labels=["No Preference","< 30", "30-60", "> 60"], active=0)
radio_button82.on_change('active',update)

table,m, div0= create_plot()

#Combine all controls to get in column

col1 = row(table,m,height=200,width=1000)
col2 = column(div1,select1, slider21, slider22,slider3, slider4, width = 250)
col3 = column(div5, div51, select51, div52, select52, div53,  checkbox52,width=250)
col4 = column(slider61,slider62,slider7, div81, radio_button81, div82, radio_button82, width=250)


#Layout
layout = column(div0,div00,col1,row(col2,col3,col4))


curdoc().add_root(layout)
curdoc().title = "hej"

output_notebook()
show(layout)

