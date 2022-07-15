from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from dash_bootstrap_components._components.Container import Container
import pokebase as pb
from sklearn import metrics

 
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, '/assert/style.css'])
 
PLOTLY_LOGO = "https://i.gifer.com/Vnni.gif"

train = pd.read_csv('PokemonGOData.csv')
train['Max CP'] = [int(i.replace(',', '')) for i in train['Max CP']]
data = train.copy()

data['Type 1'] = data['Type 1'].astype('category')
data['ประเภทที่1'] = data['Type 1'].cat.codes

data['Type 2'] = data['Type 2'].astype('category')
data['ประเภทที่2'] = data['Type 2'].cat.codes

data['Is Legendary'] = data['Is Legendary'].astype('category')
data['เป็นตำนานหรือไม่'] = data['Is Legendary'].cat.codes

data['CP สูงสุด'] = data['Max CP']

data['HP สูงสุด'] = data['Max HP']

data['พลังโจมตี'] = data['Attack']

data['พลังป้องกัน'] = data['Defense']

data['ค่าความแข็งแกร่ง'] = data['Stamina']

data['ค่าพลังโดยรวม'] = data['Total Stats']

data['Pokemon'] = data.Pokemon.astype('category')
data['ชื่อโปเกม่อน'] = data.Pokemon.cat.codes

columns = ['ประเภทที่1', 'ประเภทที่2', 'เป็นตำนานหรือไม่', 'CP สูงสุด', 'ค่าความแข็งแกร่ง']  
x_train = data[ columns ].values
y_train = data['Pokemon'].values

tree = DecisionTreeClassifier().fit(x_train, y_train)

ypred = tree.predict(x_train)
model_acc = metrics.accuracy_score(ypred, y_train) * 100

poke_type = len(set(data['Type 1']))

def dx(x):
    return [ code(data['Type 1'], x[0]), code(data['Type 2'], x[1]), code(data['Is Legendary'], x[2]), x[3], x[4]  ]

def NamePokemon_predict(x):
    d = dx(x)
    p = tree.predict([ d ])
    return p[0]


def code(column, text):
    return [ i for i in range(len(column.cat.categories)) if column.cat.categories[i] == text ][0]
 
app.layout = html.Div([
   dbc.Navbar(
           dbc.Container(
               [
                   html.A(
                       # Use row and col to control vertical alignment of logo / brand
                       dbc.Row(
                           [
                               dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                               dbc.Col(dbc.NavbarBrand("Dash PokemonGO Stats Dataset Prediction", className="ms-2")),
                           ],
                           align="center",
                           className="g-0",
                       ),
                       href="https://www.kaggle.com/datasets/bretthammit/pokemongo-stats-dataset",
                       style={"textDecoration": "none"},
                   ),
               ]
           ),
           color="dark",
           dark=True,
       )
   ,
   html.Div([
    html.Div([html.H5(f'{poke_type}'), html.P('Number of Pokemon Types')], style={'padding': 20, 'flex': 1,'background-color':'#f7f7f7', 'margin-left':20, 'margin-right':20, 'margin-top':20, 'text-align':'center', 'color':'#292b2c'}),
    html.Div([html.H5('{:.2f}'.format(model_acc)), html.P('Model  Accuracy')], style={'padding': 20, 'flex': 1, 'background-color':'#292b2c', 'margin-left':20, 'margin-right':20, 'margin-top':20, 'text-align':'center', 'color':'white'}),
    html.Div([html.H5(f'{len(data)}'), html.P('Number of rows in the data set')], style={'padding': 20, 'flex': 1, 'background-color':'#0275d8', 'margin-left':20, 'margin-right':20, 'margin-top':20, 'text-align':'center', 'color':'white'})


   ] , style={'display': 'flex', 'flex-direction': 'row'}),
   
   html.Div([
       html.Div([
 
       html.H5('Pokemon Type 1',style={'display':'inline-block','margin-right':20, 'font-weight':'bold'}),
       dcc.Dropdown(['normal', 'psychic', 'ground', 'water', 'dragon', 'steel', 'fire', 'rock', 'fairy', 'dark', 'bug', 'electric', 'ghost', 'flying', 'fighting', 'ice', 'grass', 'poison'], 'dragon', id="demo-dropdown-type1"),
       html.Br(),
 
       html.H5('Pokemon Type 2',style={'display':'inline-block','margin-right':20, 'font-weight':'bold'}),
       dcc.Dropdown(['normal', 'psychic', 'ground', 'water', 'dragon', 'steel', 'fire', 'rock', 'fairy', 'dark', 'bug', 'electric', 'ghost', 'flying', 'fighting', 'ice', 'grass', 'poison', 'none'], 'none', id="demo-dropdown-type2"),
       html.Br(),

       html.H5('Is Legendary',style={'display':'inline-block','margin-right':20, 'font-weight':'bold'}),
       dcc.Dropdown(['True', 'False'], 'False', id="demo-dropdown_TF"),
       html.Br(),

       html.H5('Max CP',style={'display':'inline-block','margin-right':20, 'font-weight':'bold'}),
           dbc.Input(placeholder="Input Max CP...", type="text", id="input-on-submit"),
           dbc.FormText(" Type number Max CP in the box above"),
           html.Br(),html.Br(),
 
        html.H5('Stamina',style={'display':'inline-block','margin-right':20, 'font-weight':'bold'}),
           dbc.Input(placeholder="Input Stamina...", type="text", id="input-on-submit2"),
           dbc.FormText(" Type number Stamina in the box above"),
           html.Br(),html.Br(),
        
        html.Div([
           dbc.Button("Submit", color="primary", id='submit-val', n_clicks=0),
   ], className="d-grid gap-2 col-6 mx-auto"),
 
   ], style={'padding': 20, 'flex': 1}),
 
       html.Div([

           html.Div(id='container-button-basic', style={'text-align':'center'},
            children=[])
 
       ], style={'padding': 20, 'flex': 1}),
      
      
   ], style={'display': 'flex', 'flex-direction': 'row'}),
 
   
])
 
 
@app.callback(
   Output('container-button-basic', 'children'),
   Input('submit-val', 'n_clicks'),
   State('demo-dropdown-type1', 'value'),
   State('demo-dropdown-type2', 'value'),
   State('demo-dropdown_TF', 'value'),
   State('input-on-submit', 'value'),
   State('input-on-submit2', 'value'),
   
)
  
def update_output(n_clicks, value, value2, value3, value4, value5):
    if value4 is not None:
        if ',' in value4:
            value4 = int(value4.replace(',', ''))
    if value5 is not None:
        if ',' in value5:
            value5 = int(value5.replace(',', ''))
        
    value3 = True if value3 == 'True' else False
    value_db = [value, value2, value3, value4, value5]

    if not None in value_db:
        result = str(NamePokemon_predict(value_db)).lower()
        pokemon = pb.pokemon(result)
        try:
            pokemon_id = pokemon.id
            return html.Div([html.H5(f'ผลการทำนายคือ {result}', style={'font-weight':'bold'}),html.Br(),html.Img(src=f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/home/{pokemon_id}.png" ,alt="", width=400, height=400)])
        except:
            return html.Div([html.H5(f'ผลการทำนายคือ {result}', style={'font-weight':'bold'}),html.Br(),html.H6('ขออภัยในขณะนี้เรายังไม่มีรูปภาพของโปเกม่อนตัวนี้', style={'font-weight':'bold','color':'red'})])
            
    else:
        return ''
 
if __name__ == '__main__':
   app.run_server(host='127.0.0.1', port='8080', debug=True)
