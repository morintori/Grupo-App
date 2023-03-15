from dash import Dash, html, dcc, ctx
import plotly.express as px
import pandas as pd
import requests
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np
import json


np.set_printoptions(threshold=np.inf)

gdf = pd.read_csv('Mexico State Centroids.csv')
df = pd.read_csv('data_agg.csv')
df = df.rename(columns={'Demanda_uni_equil': 'Units Sold', 'Semana': 'Week', 'NombreProducto': 'Product Name'})
repo_url = 'https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json'
mx_regions_geo = requests.get(repo_url).json()
mapboxtoken = 'pk.eyJ1IjoibW9yaW50b3JpIiwiYSI6ImNsZjJmeHlmMTBoeXE0MnBqcGZlcWIycmMifQ.m-Dc6TVBBiolSNXX8aNsBA'
mapboxstyle = 'mapbox://styles/morintori/clf2nnygd001h01kdt0qr5hq8'

product_names = df['Product Name'].unique()
semana_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

sum_fig_df = df[['Week', 'Revenue']].groupby(['Week']).sum().reset_index()
summary_fig = px.scatter(sum_fig_df, x='Week', y='Revenue')
summary_fig.update_traces(mode='lines+markers')
summary_fig.update_layout(title='Total Revenue by Week', title_x=0.5)
max_rev_week = df[['Week', 'Revenue', 'State']].groupby(['Week', 'State']).sum().max().values[0]
max_dem_week = df[['Week', 'Units Sold', 'State']].groupby(['Week', 'State']).sum().max().values[0]
max_rev_total = df[['Revenue', 'State']].groupby(['State']).sum().max().values[0]
max_dem_total = df[['Units Sold', 'State']].groupby(['State']).sum().max().values[0]
max_color = {'Revenue': max_rev_week, 'Units Sold': max_dem_week}
max_color_total = {'Revenue': max_rev_total, 'Units Sold': max_dem_total}
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)
app.title = 'Grupo Bimbo Inventory Demand'
app._favicon = ('favicon.ico')
server = app.server
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    html.H2(children='Grupo Bimbo Demand'),
    html.H6('Grupo Bimbo is a Mexican bakery chain that sells their baked goods country-wide,'
            ' here we have the inventory demand for the bakery organized by State and Product Cluster.'
            ' The inventory demand for weeks 10 and 11 were predicted using a XGBoost model, weeks 3-9 is the train'
            ' set.'
            ' Product Clusters were created by using a Natural Language Processing Model using Spherical K-Means to'
            ' seperate the products into 30 clusters. Revenue is based on Revenue/week.'
            ' Product Clusters/States in the bar chart can be clicked to display its distribution in the pie chart'
            ', the products in the pie chart can be clicked to display its demand over time. The dropdown menu '
            'below for product names can be chosen to display its time series. Below that, product clusters can be '
            'chosen to see its component products.'),
    html.Div([

        html.Div([
            html.H6('Choose Product or State View'),
            dcc.Dropdown(
                ['Product', 'State'],
                'State',
                id='state_or_product_dd',
            ),
            dcc.RadioItems(
                ['Revenue', 'Units Sold'],
                'Revenue',
                id='rev_or_units_radio',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            ),
            html.Div(id='state_or_product_graph'),
            # dcc.Graph(
            #     id='state_or_product_graph',
            # ),
            dcc.Slider(
                3,
                12,
                step=None,
                id='crossfilter-Semana-slider',
                value=8,
                marks={str(Semana): (str(Semana) if Semana != 12 else 'All') for Semana in semana_list},
            ),
            html.Center('Week')

        ],
            style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

        html.Div([
            dcc.Store(id = 'state_name'),
            dcc.Graph(id='cluster-pie',
                      clickData={'points': [{'label': '0'}]},
                      hoverData={'points': [{'label': '0'}]}
                      ),

            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-yaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            ),

            dcc.Graph(id='time-series'),
            html.H6('Search Product Name to View Demand Over Time Here'),
            dcc.Dropdown(product_names, id='time-series-select', placeholder='Search Product Name Here'),
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

        html.Div(className='row', children=[
            html.Div([
                dcc.Graph(
                    id='summary-bar',
                    figure=summary_fig
                ),

                # html.H6('Product Cluster',id='display_type'),
                #
                # dcc.Dropdown(clusters_prod,id='product_cluster'),
                # html.Hr(),
                # html.Pre(id = 'display_product_names',style = styles['pre'])
            ], className='twelve columns'),
            dcc.Clipboard(target_id="structure"),
            html.Pre(
                id='structure',
                style={
                    'border': 'thin lightgrey solid',
                    'overflowY': 'scroll',
                    'height': '275px'
                })
        ])

    ])
    # html.Div([
    #
    #
    # ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})

])


# html.Div([
#     # dcc.Graph(
#     #     id='state_or_product_graph',
#     #     clickData={'points': [{'label': 'México'}]}
#     # ),
#     # dcc.Slider(
#     #
#     #     3,
#     #     12,
#     #     step=None,
#     #     id='crossfilter-Semana-slider',
#     #     value=11,
#     #     marks={str(Semana): (str(Semana) if Semana !=12 else 'All') for Semana in semana_list},
#     # ),
#
# ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),


# @app.callback(
#     Output('display_product_names','children'),
#     Input('product_cluster','value'),
# )
# def show_cluster(value):
#     prodInClust = product_list[product_list['cluster']==value]['NombreProducto'].unique()
#
#     return '{}'.format(prodInClust)

@app.callback(
    Output('state_or_product_graph', 'children'),
    # Input('state_or_product_graph','figure'),
    Input('crossfilter-Semana-slider', 'value'),
    Input('state_or_product_dd', 'value'),
    Input('rev_or_units_radio', 'value'),
)
def update_graph(semana, x_type, y_type):
    if x_type == 'Product':
        x_type = 'Product Name'
    if semana == 12:
        dff = df[[x_type, y_type]].groupby(x_type) \
            .sum().reset_index()
        zmax_ = max_color_total[y_type]
    else:
        dff = df[df['Week'] == semana][[x_type, y_type]].groupby(x_type) \
            .sum().reset_index()
        zmax_ = max_color[y_type]

    if x_type == 'State':
        gdff=gdf.copy()
        gdff = gdff.merge(dff,how='left', left_on ='name',right_on='State')
        gdff['Revenue']='$' + (gdff['Revenue'].astype(float)/1000000).round(2).astype(str) + 'MM'
        # print(gdff.head())
        data = [go.Choroplethmapbox(geojson=mx_regions_geo, ids=dff['State'], z=dff[y_type],
                                    locations=dff['State'], featureidkey='properties.name', colorscale='reds',
                                    marker=dict(line=dict(color='black'), opacity=0.6),
                                    zmin = 0, zmax = zmax_
                                    ),
                go.Scattermapbox(
                                # hoverinfo=gdff['Revenue'],
                                lat = gdff['lat'],lon = gdff['lon'],
                                text = gdff['Revenue'],
                                textfont= {'color':'black','size':15,'family':'Courier New'},
                                mode = 'text',
                                name = 'Revenue',

                                ),
                ]
        layout = go.Layout(
            autosize=True,
            mapbox={
            'accesstoken': mapboxtoken,
            'style': mapboxstyle,
            'zoom': 4,
            'center': {'lat': 25, 'lon': -99}},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=700,

            )

    return dcc.Graph(id='graph',figure={'data': data, 'layout': layout}, )

@app.callback(
    Output('state_name','data'),
    Input('graph', 'clickData'),
    Input('state_or_product_dd', 'value'),
)
def store_state_name(clickData, x_type):
    if x_type =='State':
        number_ = clickData['points'][0]['pointNumber']
        name_ = gdf.iloc[number_,:]['name']
    return name_

@app.callback(
    Output("structure", "children"),
    Input("cluster-pie", "clickData"))
def display_structure(fig_json):
    return json.dumps(fig_json, indent=2)


def create_pie(dff,y_type, title):

    fig = px.pie(dff,
                 names = 'Product Name',
                 values=y_type,
                 height = 700,
                 )

    fig.update_traces(textposition='inside',
                      hovertemplate =
                      '<b>%{label}</b><br>'+
                      '<i>Revenue:</i> $%{value}'
                      )
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide',margin=dict(t=50, l=25, r=25, b=25),\
                      title ='Top 20 Products by ' + y_type + ' in ' + title,)

    return fig
#
def create_time_series(dff,y_type,title,axis_type,sp_name):

    fig = px.scatter(dff,x='Week',y=y_type)
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False,tickmode='linear',dtick=1)
    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 30}, title = 'Product '+ y_type + \
                                                                        ' Per Week in ' + sp_name, title_x = 0.5)
    return fig
# #
# @app.callback(
#     Output('product name','children'),
#     Input('cluster-pie','hoverData'),
# )
# def display_product_name(hoverData):
#
#     product_ID = hoverData['points'][0]['label']
#
#     product_name = product_list[product_list['Producto_ID']==int(product_ID)]['NombreProducto'].item()
#     return str(product_name)
#
#
#
@app.callback(
    Output('time-series','figure'),
    Input('cluster-pie','clickData'),
    Input('state_name','data'),
    Input('time-series-select','value'),
    Input('rev_or_units_radio', 'value'),
    Input('crossfilter-yaxis-type','value')
)
def update_ts(clickData,sp_name,p_Select,y_type,axis_type):
    trig_id = ctx.triggered_id if not None else 1
    if trig_id == 'time-series-select':
        product_nm = ctx.triggered[0]['value']

    else:
        product_nm = clickData['points'][0]['label']

    dff =df[df['Product Name']==product_nm]
    dff = dff[dff['State'] == sp_name]

    dff = dff[[ 'Week', y_type]].groupby(['Week']).sum().reset_index()

    # product_name= product_list[product_list['Producto_ID']==int(product_id)]['NombreProducto'].item()
    return create_time_series(dff,y_type,product_nm,axis_type,sp_name)

# #
@app.callback(
    Output('cluster-pie', 'figure'),
    Input('graph', 'clickData'),
    Input('crossfilter-Semana-slider','value'),
    Input('state_name','data'),
    Input('rev_or_units_radio', 'value'),
    Input('state_or_product_dd', 'value'),
)
def update_pie(clickData,semana,sp_name,y_type,x_type):
    # print(clickData['points'][0]['pointNumber'])
    # if x_type =='State':
    #     number_ = clickData['points'][0]['pointNumber']
    #     name_ = gdf.iloc[number_,:]['name']
    # print(sp_name)
    if semana == 12:
        dff = df[df[x_type]== sp_name]
        dff = dff[['Product Name', y_type]].groupby('Product Name') \
            .sum().reset_index()

    else:
        dff = df[df['Week'] == semana]
        dff = dff[dff[x_type]== sp_name]
        dff = dff[['Product Name', y_type]].groupby('Product Name') \
            .sum().reset_index()

    if x_type =='State':
        dff= dff.sort_values(by='Revenue', ascending=False)
        dff = dff.iloc[:20,:]

    title = '<b>{}</b><br>week:{}'.format(sp_name, semana)

    return create_pie(dff,y_type, title)

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")

# next steps add map to replace main bar graph, search bar to group like items and display

# if x_type == 'State':
#     print(dff.head())
#     data = [go.Choroplethmapbox(geojson=mx_regions_geo, ids=dff['State'], z=dff[y_type],
#                                         locations=dff['State'], featureidkey='properties.name', colorscale='reds',
#                                         marker=dict(line=dict(color='black'), opacity=0.6))]
#     # fig.add_trace(go.Choroplethmapbox(name='Mexico', geojson=mx_regions_geo, ids=dff['State'], z=dff[y_type],
#     #                                     locations=dff['State'], featureidkey='properties.name', colorscale='reds',
#     #                                     marker=dict(line=dict(color='black'), opacity=0.6)))
#     layout = go.Layout(mapbox = {
#                             'style':'open-street-map',
#                             'zoom':4,
#                             'center':{'lat': 25, 'lon': -99}},
#                        margin={"r": 0, "t": 0, "l": 0, "b": 0})
#     # fig.update_layout(mapbox_style='open-street-map',
#     #                   mapbox_zoom=4,
#     #                   mapbox_center={'lat': 25, 'lon': -99}
#     #                   )
#     # fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
#     # fig = px.choropleth(data_frame=dff,
#     #                     geojson=mx_regions_geo,
#     #                     locations='State',  # nombre de la columna del Dataframe
#     #                     featureidkey='properties.name',
#     #                     # ruta al campo del archivo GeoJSON con el que se hará la relación (nombre de los estados)
#     #                     color=y_type,  # El color depende de las cantidades
#     #                     color_continuous_scale="burg",
#     #                     scope="north america"
#     #                     )
#     #
#     # fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")
# elif x_type =='Product Name':
#     data=[go.Scatter(x = dff[x_type],y=df[y_type])]
#     layout = go.Layout()
