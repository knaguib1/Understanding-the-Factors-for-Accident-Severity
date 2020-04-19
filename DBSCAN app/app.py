import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly

# dashboard layout
app = dash.Dash()

app.layout = html.Div(
    children=[
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='four columns div-user-controls',
                    children=[  # header
                        html.H2("DBSCAN Implementation on US Car Accidents"),

                        # Direction for users
                        html.P(
                            """ Select different datasets and filters
                             to visualize the DBSCAN alogrithm on US car accidents.
                            """
                        ),

                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                dcc.Dropdown(
                                    id='state',
                                    options=[
                                        {'label': 'New York', 'value': 'NY'},
                                        {'label': 'Georgia', 'value': 'GA'},
                                        {'label': 'New Jersey', 'value': 'NJ'},
                                        {'label': 'Texas', 'value': 'TX'},
                                        {'label': 'California', 'value': 'CA'}],
                                    value='NY'
                                )
                            ]
                        ),

                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(
                                            id='eps',
                                            options=[
                                                {'label': '50 m radius epsilon', 'value': '_eps_005'},
                                                {'label': '100 m radius epsilon', 'value': '_eps_01'}],
                                            value='_eps_005'
                                        )
                                    ]
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(
                                            id='n_clusters',
                                            options=[{'label': 'Top ' + str(i) + ' clusters',
                                                      'value': i}
                                                     for i in range(1, 21)],
                                            value=20
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(
                                            id='sev',
                                            options=[{'label': 'Severity ' + \
                                                      str(i), 'value': i} for i in range(1, 5)],
                                            value=[1, 2, 3, 4],
                                            multi=True
                                        )
                                    ]
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(
                                            id='cluster_id',
                                            options=[{'label': 'Cluster ' + str(i), 'value': 'Cluster ' + str(i)}
                                                     for i in range(1, 21)],
                                            multi=True,
                                            placeholder="Select cluster id(s)"
                                        )
                                    ]
                                ),
                                dcc.Markdown(
                                    id='txt',
                                    children="""
                                             """
                                ),
                                html.Div(
                                    children=[
                                        dcc.Graph(id='heatmap')
                                    ]
                                ),
                            ]
                        )

                    ]

                ),
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id="map-graph"),
                        dcc.Graph(id='histogram')
                    ]
                )
            ]
        )
    ]
)


@app.callback(Output('txt', 'children'),
              [Input('state', 'value'),
               Input('eps', 'value'),
               Input('map-graph', 'hoverData')])
def update_txt(state, eps, info):
    if info:
        df = pd.read_pickle('data/' + state + eps + '.pkl')

        data = info['points'][0]
        lon, lat = float(data['lon']), float(data['lat'])
        loc = str(data['hovertext'])

        datapoint = df[(df['Location'] == loc) & (
            df['Start_Lng'] == lon) & (df['Start_Lat'] == lat)]

        date = datapoint['Start_Time'].dt.strftime("%m/%d/%Y, %H:%M:%S").values[0]
        weekend = datapoint['Weekend'].values[0]
        if weekend != 0:
            weekend = 'Weekday'
        else:
            weekend = 'Weekend'

        weather = datapoint['Weather_Condition'].values[0]
        temp = datapoint['Temperature(F)'].values[0]
        precip = datapoint['Precipitation(in)'].values[0]
        if precip != 1:
            precip = 0

        visibility = datapoint['Visibility(mi)'].values[0]

        accident_info = """
                        **Time of accident:** {}, *{}*.
                        **Weather conditions:** {}.
                        **Temperature:** {} degr F.
                        **Precipitation:** {} in.
                        **Visibility:** {} mi.
                        """.format(date, weekend, weather, temp,
                                   precip, visibility)

        return accident_info


@app.callback(Output('map-graph', 'figure'),
              [Input('state', 'value'),
               Input('eps', 'value'),
               Input('n_clusters', 'value'),
               Input('cluster_id', 'value'),
               Input('sev', 'value'),
               Input('heatmap', 'clickData')])
def update_map(state, eps, n_clusters, cluster_id, sev, clickData):
    df = pd.read_pickle('data/' + state + eps + '.pkl')

    df['cluster'] = df['new_cluster'].apply(lambda x: x.split()[1]).astype(int)
    clusters = df[(df['Severity'].isin(sev)) & (df['cluster'] < (n_clusters + 1))]

    map_zoom = 8
    map_color = 'new_cluster'
    font_color = '#d8d8d8'

    if cluster_id:
        clusters = clusters[clusters['new_cluster'].isin(cluster_id)]
        if len(cluster_id) == 1:
            map_zoom = 15
        map_color = 'Severity'
        font_color = '#1E1E1E'

    # create graphics
    fig = px.scatter_mapbox(clusters, lat="Start_Lat", lon="Start_Lng",
                            hover_name='Location',
                            hover_data=['Severity', 'cluster'],
                            color=map_color,
                            size='Severity',
                            size_max=10,
                            zoom=map_zoom,
                            color_discrete_sequence=px.colors.qualitative.T10,
                            opacity=0.7)

    fig.update_layout(mapbox_style="carto-positron",
                      font={'family': 'Open Sans', 'size': 10, 'color': font_color},
                      margin={"r": 10, "t": 20, "l": 0, "b": 0},
                      paper_bgcolor='#1E1E1E',
                      plot_bgcolor='#1E1E1E',
                      hovermode='closest',
                      legend={'yanchor': 'auto'})

    return fig


@app.callback(Output('histogram', 'figure'),
              [Input('state', 'value'),
               Input('eps', 'value'),
               Input('n_clusters', 'value'),
               Input('sev', 'value')])
def update_barchart(state, eps, n_clusters, sev):
    clusters = pd.read_pickle('data/' + state + eps + '.pkl')

    df = clusters.groupby(by=['new_cluster', 'Severity'])['ID'].count().reset_index()

    df['cluster'] = df['new_cluster'].apply(lambda x: x.split()[1]).astype(int)
    df = df[(df['Severity'].isin(sev)) & (df['cluster'] < (n_clusters + 1))]

    colors = [plotly.colors.sequential.YlOrRd[i] for i in [1, 3, 5, 7]]

    x = df['new_cluster'].unique()

    layout = go.Layout(
        bargap=0.01,
        bargroupgap=0,
        barmode="stack",
        margin=go.layout.Margin(l=10, r=10, t=30, b=0),
        dragmode="select",
        yaxis={'categoryorder': 'total ascending'},
        xaxis={'title': 'Accident Occurrences'},
        font={'family': 'Open Sans', 'size': 10, 'color': '#d8d8d8'},
        paper_bgcolor='#1E1E1E',
        plot_bgcolor='#1E1E1E',
        hovermode='closest'
    )

    fig = go.Figure(data=[go.Bar(y=x, x=df[df['Severity'] == 1]['ID'],
                                 name='Sev 1', hoverinfo='x',
                                 marker_color=colors[0], orientation='h'),

                          go.Bar(y=x, x=df[df['Severity'] == 2]['ID'],
                                 name='Sev 2', hoverinfo='x',
                                 marker_color=colors[1], orientation='h'),

                          go.Bar(y=x, x=df[df['Severity'] == 3]['ID'],
                                 name='Sev 3', hoverinfo='x',
                                 marker_color=colors[2], orientation='h'),

                          go.Bar(y=x, x=df[df['Severity'] == 4]['ID'],
                                 name='Sev 4', hoverinfo='x',
                                 marker_color=colors[3], orientation='h')],
                    layout=layout)

    return fig


@app.callback(Output('heatmap', 'figure'),
              [Input('state', 'value'),
               Input('eps', 'value'),
               Input('n_clusters', 'value'),
               Input('cluster_id', 'value'),
               Input('sev', 'value')])
def update_heatmap(state, eps, n_clusters, cluster_id, sev):
    df = pd.read_pickle('data/' + state + eps + '.pkl')

    df['cluster'] = df['new_cluster'].apply(lambda x: x.split()[1]).astype(int)
    clusters = df[(df['Severity'].isin(sev)) & (df['cluster'] < (n_clusters + 1))]

    if cluster_id:
        clusters = clusters[clusters['new_cluster'].isin(cluster_id)]

    cluster_piv = pd.pivot_table(clusters, values='Severity',
                                 index='Start_Hour', columns='Weekend')

    layout = go.Layout(
        yaxis={'title': 'Hour of Day'},
        font={'family': 'Open Sans', 'size': 10, 'color': '#d8d8d8'},
        paper_bgcolor='#1E1E1E',
        plot_bgcolor='#1E1E1E',
        annotations=[
            dict(
                x=1.28,
                y=1.05,
                align="left",
                valign="top",
                text='Average Severity',
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="auto",
                yanchor="auto"
            )
        ],
        margin=go.layout.Margin(l=50, r=0, t=30, b=10),
        height=429
    )

    fig = go.Figure(data=go.Heatmap(
        z=cluster_piv,
        y=[str(i) + ':00' for i in range(24)],
        x=['Weekday', 'Weekend'],
        colorscale='YlOrRd'),
        layout=layout)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
