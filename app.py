from flask import Flask, render_template,session, url_for,redirect, request
import numpy as np
from wtforms import TextField, SubmitField
from flask_wtf import FlaskForm
import pandas as pd
import joblib
import folium

### Fonction qui renvoie au résultat de la prédiction 
def return_prediction(model,scaler,sample_json):
    capp_ = sample_json['Cappuccino']
    cine_ = sample_json['Cinema']
    wine_ = sample_json["Wine"]
    gaso_ = sample_json['Gasoline']
    rent_ = sample_json['Avg.Rent']
    inc_ = sample_json['Avg.Disposable.Income']
    
    place = [[capp_,cine_,wine_,gaso_,rent_,inc_]]
    
    place = scaler.transform(place)
    
    classes = model.predict(place)

    data = pd.read_csv('data.csv')
    
    villes = data['City'][data['label']==int(classes)]
    
    return list(villes)


### Fonction qui renvoie au résultat de la prédiction de cluster sur la map
def return_prediction_card(model,scaler,sample_json):
    capp_ = sample_json['Cappuccino']
    cine_ = sample_json['Cinema']
    wine_ = sample_json["Wine"]
    gaso_ = sample_json['Gasoline']
    rent_ = sample_json['Avg.Rent']
    inc_ = sample_json['Avg.Disposable.Income']
    
    place = [[capp_,cine_,wine_,gaso_,rent_,inc_]]
    
    place = scaler.transform(place)
    
    classes = model.predict(place)
    
    data = pd.read_csv('data.csv')
    
    villes = data['City'][data['label']==int(classes)]
    
    data_merged = pd.read_csv('data_merged.csv')
    
    location = [[i,j] for i,j in (data_merged[['lat','lng']]).items()]
    
    colors = {0 : 'red', 1 : 'blue', 2 : 'green',3 : 'yellow', 4 : 'black', 5 : 'orange'}

    m = folium.Map(location = [27.901402, 10.903920], zoom_start= 2.2,tiles="Stamen Terrain")

    data_merged[data_merged["label"]==int(classes)].apply(lambda row:folium.Marker(location=[row["lat"], row["lng"]], 
                                                  radius=8,icon=folium.Icon(color=colors[row['label']]), popup=row['City_x'])
                                                 .add_to(m), axis=1)
    m
    
    return m
#_repr_html_()

app = Flask(__name__)
app.config['SECRET_KEY']  = 'mysecretkey'

class FlowerForm(FlaskForm):
    
    capp_price = TextField("Prix moyen d'un cafée")
    cine_price = TextField("Prix moyen d'un cinéma")
    wine_price = TextField("Prix moyen d'une bouteille de vin")
    gaso_price = TextField("Prix moyen d'un litre de gasoil")
    rent_price = TextField("Cout moyen du loyer")
    inc_price = TextField("Revenu moyen")

    submit = SubmitField('Analyser')
    submit2 = SubmitField('cliquer ici')

@app.route("/", methods=['GET','POST'])
def index():
   form = FlowerForm()
   
   if form.validate_on_submit():
       
     session['capp_price'] = form.capp_price.data 
     session['cine_price'] = form.cine_price.data
     session['wine_price'] = form.wine_price.data
     session['gaso_price'] = form.gaso_price.data
     session['rent_price'] = form.rent_price.data
     session['inc_price'] = form.inc_price.data
     
     return redirect(url_for("prediction"))
 
   return render_template('home.html', form=form)
     
     
smart_c_model = joblib.load("smart_city_model.sav")
smart_c_scaler = joblib.load('smart_city_scaler.pkl')

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        if request.form.get('action1') == 'cliquez ici':
            return redirect(url_for("/prediction/villes"))
    content = {}
        
    content['Cappuccino'] =float(session['capp_price'])
    content['Cinema'] =float(session['cine_price'])
    content['Wine'] =float(session['wine_price'])
    content['Gasoline'] =float(session['gaso_price'])
    content['Avg.Rent'] =float(session['rent_price'])
    content['Avg.Disposable.Income'] =float(session['inc_price'])
        
    results = return_prediction(smart_c_model,smart_c_scaler,content)
    result2 = return_prediction_card(smart_c_model,smart_c_scaler,content)
        

    
    return render_template('prediction.html', results=results) 



@app.route('/prediction/villes', methods=['POST'])
def prediction_villes():
    
    content = {}
        
    content['Cappuccino'] =float(session['capp_price'])
    content['Cinema'] =float(session['cine_price'])
    content['Wine'] =float(session['wine_price'])
    content['Gasoline'] =float(session['gaso_price'])
    content['Avg.Rent'] =float(session['rent_price'])
    content['Avg.Disposable.Income'] =float(session['inc_price'])
        
    result2 = return_prediction_card(smart_c_model,smart_c_scaler,content)
    result2.save('templates/index2.html')
        

    return result2._repr_html_()
    



if __name__=='__main__':
    app.run(debug=True)