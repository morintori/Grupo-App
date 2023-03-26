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

repo_url = 'mexicoHigh.json'
with open(repo_url,'r') as f:
    mx_regions_geo = json.load(f)
# mx_regions_geo = requests.get(repo_url).json()
mapboxtoken = 'pk.eyJ1IjoibW9yaW50b3JpIiwiYSI6ImNsZmE3a2I1aTJteHAzeGxoZXhpZXB6OHIifQ.yxdD2S1f7B3M1PsP7JVGug'
mapboxstyle = 'mapbox://styles/morintori/clf2nnygd001h01kdt0qr5hq8'

product_names = df['Product Name'].unique()
semana_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
type_list = df['Type'].unique()

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

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.config['suppress_callback_exceptions'] = True
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
            ' here we have the inventory demand for the bakery organized by State and Product.'
            ' The inventory demand for weeks 10 and 11 were predicted using a XGBoost model.'
            ' The user can select Revenue/Units Sold for an understanding of how much demand there was for the '
            ' Product. The Graphs are interactive, clicking on different sectors will allow different segments of data'
            ' to be displayed. A dropdown is provided to search for the Revenue/Units Sold over time for any product.'
            ' The purpose of this dashboard is to inform seasonal changes in the demand of certain products, as well as'
            ' the popularity of different items.'
            ),
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
            dcc.Store(id='sp_name', data='Ciudad de México'),
            html.Div(id='pie_type_or_product'),
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
            dcc.RadioItems(
                ['State', 'Country'],
                'State',
                id='crossfilter-state-country',
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

])


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

    if x_type == 'State':
        if semana == 12:
            dff = df[[x_type, y_type]].groupby(x_type).sum().reset_index()
            zmax_ = max_color_total[y_type]
        else:
            dff = df[df['Week'] == semana][[x_type, y_type]].groupby(x_type).sum().reset_index()
            zmax_ = max_color[y_type]

        gdff = gdf.copy()
        gdff = gdff.merge(dff, how='left', left_on='name', right_on='State')
        if y_type == 'Revenue':
            gdff['Revenue'] = '$' + (gdff['Revenue'].astype(float) / 1000000).round(2).astype(str) + 'MM'
        elif y_type == 'Units Sold':
            gdff['Units Sold'] = (gdff['Units Sold'].astype(float) / 1000000).round(2).astype(str) + 'MM'
        # np.stack((df['height'], df['thickness_1'], df['thickness_2']), axis=-1)

        dff['Y MM'] = (dff[y_type].astype(float) / 1000000)
        if y_type == 'Revenue':
            pre = '$'
        else:
            pre = ''
        if (semana <= 11) & (semana > 3):

            dfff = df[df['Week'] == (semana - 1)][[x_type, y_type]].groupby(x_type).sum().reset_index()
            dff = dff.merge(dfff, how='left', on='State', suffixes=[None, '_y'])
            dff['Percent Change'] = ((dff[y_type] - dff[y_type + '_y']) / dff[y_type] * 100).round(1)
            cmb = go.Choroplethmapbox(geojson=mx_regions_geo, ids=dff['State'], z=dff[y_type],
                                      locations=dff['State'], featureidkey='properties.name', colorscale='reds',
                                      marker=dict(line=dict(color='black'), opacity=0.6),
                                      zmin=0, zmax=zmax_, name=y_type,
                                      customdata=np.stack((dff['Percent Change'],dff['Y MM']),axis=-1),
                                      hovertemplate=
                                      '<b>%{id}</b><br>' +
                                      pre +
                                      '%{customdata[1]:.2f} MM<br>' +
                                      '%{customdata[0]}%'
                                      )
        else:

            cmb = go.Choroplethmapbox(geojson=mx_regions_geo, ids=dff['State'], z=dff[y_type],
                                      locations=dff['State'], featureidkey='properties.name', colorscale='reds',
                                      marker=dict(line=dict(color='black'), opacity=0.6),
                                      zmin=0, zmax=zmax_, name=y_type, customdata=(dff['Y MM']),
                                      hovertemplate=
                                      '<b>%{id}</b><br>' +
                                      pre +
                                      '%{customdata:.2f} MM'
                                      )
        data = [cmb,
                go.Scattermapbox(
                    # hoverinfo=gdff['Revenue'],
                    lat=gdff['lat'], lon=gdff['lon'],
                    text=gdff[y_type],
                    textfont={'color': 'black', 'size': 10, 'family': 'Courier New'},
                    mode='text',
                    name=y_type, hoverinfo='skip'

                ),
                ]
        layout = go.Layout(
            autosize=True,
            mapbox={
                'accesstoken': mapboxtoken,
                'style': mapboxstyle,
                # 'style': 'open-street-map',
                'zoom': 4,
                'center': {'lat': 25, 'lon': -99}},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=700,

        )
    elif x_type == 'Product Name':
        if semana == 12:
            dff = df
        else:
            dff = df[df['Week'] == semana]
        dff = dff[['Product Name',y_type,'Type']].groupby(['Product Name','Type']).sum().reset_index()
        dff = dff.sort_values(by=y_type, ascending=False)
        dff = dff.iloc[:500, :]
        fig = px.treemap(dff, path=[px.Constant('all'), 'Type', 'Product Name'], values=y_type, height=700)
        fig.data[0].textinfo = 'value + percent parent'
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        data = fig['data']
        layout = fig['layout']

    return dcc.Graph(id='graph', figure={'data': data, 'layout': layout}, \
                     clickData={"points":[{"id":"Ciudad de México","label":"Nito 1p 62g Central BIM 2425"}]} )


@app.callback(
    Output('pie_type_or_product','children'),
    Input('state_or_product_dd', 'value'),
)
def display_productortype(sp):
    if sp == 'State':
        buttons = dcc.RadioItems(
            ['Top 20 Products', 'Product Types'],
            'Top 20 Products',
            id='types_or_top20',
            labelStyle={'display': 'inline-block', 'marginTop': '5px'})
    elif sp == 'Product':
        buttons = dcc.RadioItems(
            id='types_or_top20',
            labelStyle={'display': 'inline-block', 'marginTop': '5px'})
    return buttons


@app.callback(
    Output('sp_name', 'data'),
    Input('graph', 'clickData'),
    Input('state_or_product_dd', 'value'), prevent_initial_call=True
)
def store_sp_name(clickData, x_type):
    if x_type == 'State':
        name_ = clickData['points'][0]['id']

    elif x_type == 'Product':
        name_ = clickData['points'][0]['label']
    return name_


@app.callback(
    Output("structure", "children"),
    Input('graph ', 'clickData'))
def display_structure(fig_json):
    return json.dumps(fig_json, indent=2)


def create_pie(dff, y_type, names_, title):

    fig = px.pie(dff,
                 names=names_,
                 values=y_type,
                 height=700,
                 )

    if y_type == 'Revenue':
        fig.update_traces(textposition='inside',
                          hovertemplate=
                          '<b>%{label}</b><br>' +
                          '<i>Revenue:</i> $%{value}'
                          )
    elif y_type == 'Units Sold':
        fig.update_traces(textposition='inside',
                          hovertemplate=
                          '<b>%{label}</b><br>' +
                          '<i>Units Sold:</i> $%{value}'
                          )

    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', margin=dict(t=50, l=25, r=25, b=25), \
                      title=title, )

    return fig


#
def create_time_series(dff, y_type, title, axis_type, sp_name):
    fig = px.scatter(dff, x='Week', y=y_type)
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False, tickmode='linear', dtick=1)
    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 30}, title='Product ' + y_type + \
                                                                                     ' Per Week in ' + sp_name,
                      title_x=0.5)
    return fig


@app.callback(
    Output('time-series', 'figure'),
    Input('cluster-pie', 'clickData'),
    Input('sp_name', 'data'),
    Input('time-series-select', 'value'),
    Input('rev_or_units_radio', 'value'),
    Input('state_or_product_dd', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-state-country', 'value'),
    prevent_initial_call=True
)
def update_ts(clickData, sp_name, p_Select, y_type, x_type, axis_type,st_cn):
    trig_id = ctx.triggered_id if not None else 1
    if trig_id == 'time-series-select':
        product_nm = ctx.triggered[0]['value']
        if x_type == 'Product':
            sp_name = clickData['points'][0]['label']

    else:
        product_nm = clickData['points'][0]['label']
        if x_type == 'Product':
            sp_name_ = sp_name
            sp_name = product_nm
            product_nm = sp_name_

    if product_nm in type_list:
        dff = df[df['Type'] == product_nm]
    else:
        dff = df[df['Product Name'] == product_nm]

    if st_cn == 'State':
        dff = dff[dff['State'] == sp_name]
    elif st_cn =='Country':
        sp_name = 'Country-Wide'


    dff = dff[['Week', y_type]].groupby(['Week']).sum().reset_index()

    return create_time_series(dff, y_type, product_nm, axis_type, sp_name)



@app.callback(
    Output('cluster-pie', 'figure'),
    Input('graph', 'clickData'),
    Input('crossfilter-Semana-slider', 'value'),
    Input('sp_name', 'data'),
    Input('rev_or_units_radio', 'value'),
    Input('state_or_product_dd', 'value'),
    Input('time-series-select', 'value'),
    Input('types_or_top20','value'),
    Input('crossfilter-state-country','value'),
    prevent_initial_call=True
)
def update_pie(clickData, semana, sp_name, y_type, x_type,pd_choice,typetop20,st_cn):

    trig_id = ctx.triggered_id if not None else 1

    if trig_id == 'time-series-select':
        if x_type == 'Product':
            sp_name = ctx.triggered[0]['value']

    if x_type == 'State':
        if typetop20 == 'Product Types':
            labs = 'Type'
        elif typetop20 == 'Top 20 Products':
            labs = 'Product Name'

        if semana == 12:
            if st_cn == 'State':
                dff = df[df[x_type] == sp_name]
            elif st_cn== 'Country':
                sp_name = 'Country of Mexico'
            dff = dff[[labs, y_type]].groupby(labs) \
                .sum().reset_index()
            title = '<b>{}</b><br>'.format(sp_name)


        else:
            dff = df[df['Week'] == semana]
            if st_cn == 'State':
                dff = df[df[x_type] == sp_name]
            elif st_cn == 'Country':
                sp_name = 'Country of Mexico'
            dff = dff[[labs, y_type]].groupby(labs) \
                .sum().reset_index()
            title = '<b>{}</b><br>week:{}'.format(sp_name, semana)

        title = typetop20 + ' by ' + y_type + ' in ' + title

        if typetop20 == 'Top 20 Products':
            dff = dff.sort_values(by=y_type, ascending=False)
            dff = dff.iloc[:20, :]
            names_ = 'Product Name'
        elif typetop20 == 'Product Types':
            names_ = 'Type'


    elif x_type == 'Product':
        x_type = 'Product Name'
        if semana == 12:
            if sp_name in type_list:
                dff = df[df['Type'] == sp_name]
            else:
                dff = df[df[x_type] == sp_name]
            dff = dff[['State', y_type]].groupby('State').sum().reset_index()
            title = '<b>{}</b><br>'.format(sp_name)
        else:
            dff = df[df['Week'] == semana]
            if sp_name in type_list:
                dff = dff[dff['Type']==sp_name]
            else:
                dff = dff[dff[x_type] == sp_name]
            dff = dff[['State', y_type]].groupby('State').sum().reset_index()
            title = '<b>{}</b><br>week:{}'.format(sp_name, semana)
        title = 'Distribution of ' + title + ' ' + y_type + ' by State'
        names_ = 'State'

    return create_pie(dff, y_type, names_, title)


if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0")
