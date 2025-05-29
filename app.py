from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def pagina_inicio(): 
    return render_template('inicio.html') 

if __name__ == '__main__':
    app.run(debug=True)