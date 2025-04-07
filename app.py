from flask import Flask, render_template, request
from datetime import datetime
import re 
import os
import LinearRegression

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, flask!"

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()

    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! Hour: " + str(now)
    return content

@app.route("/pagina")
def pagina():
    return render_template("pagina.html")

@app.route("/LinearRegression", methods= ["GET", "POST"])
def calculateGrade():
    calculateResult = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        calculateResult = LinearRegression.calculateGrade(hours)
    return render_template("LinearRegressionGrades.html", result = calculateResult)

@app.route("/logistica")
def logistica():
    return render_template("logistica.html")

@app.route("/menu_modelos")
def menu_modelos():
    return render_template("menu_modelos.html")

@app.route("/modelos/<modelo>")
def mostrar_modelo(modelo):
    try:
        return render_template(f"modelos/{modelo}.html")
    except:
        return "Modelo no encontrado", 404