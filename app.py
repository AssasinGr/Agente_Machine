from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from agente_rl.agente import AgenteQLearning

app = Flask(__name__)

historial_recompensas_entrenamiento = []
tabla_valores_q = None
informacion_entorno = {} 

@app.route('/')
def pagina_inicio(): 
    return render_template('inicio.html') 

@app.route('/que-es-rl') 
def explicacion_aprendizaje_refuerzo(): 
    return render_template('explicacion_rl.html') 

@app.route('/descripcion-entorno') 
def describir_entorno(): 
    return render_template('entorno.html', env_info=informacion_entorno) 

@app.route('/entrenamiento', methods=['POST']) 
def entrenar_agente_rl(): 
    global historial_recompensas_entrenamiento, tabla_valores_q, informacion_entorno

    tasa_aprendizaje = float(request.form.get('tasa_aprendizaje', 0.1))
    factor_descuento = float(request.form.get('factor_descuento', 0.99))
    tasa_decaimiento_epsilon = float(request.form.get('tasa_decaimiento_epsilon', 0.001))
    numero_episodios = int(request.form.get('numero_episodios', 10000))

    agente_de_refuerzo = AgenteQLearning(
        nombre_entorno='FrozenLake-v1', 
        tasa_aprendizaje=tasa_aprendizaje,
        factor_descuento=factor_descuento,
        tasa_decaimiento_epsilon=tasa_decaimiento_epsilon,
        numero_episodios=numero_episodios
    )
    historial_recompensas_entrenamiento, tabla_valores_q = agente_de_refuerzo.entrenar(n_episodes=numero_episodios)
    informacion_entorno = agente_de_refuerzo.obtener_info_entorno() 

    plt.figure(figsize=(10, 6))
    plt.plot(historial_recompensas_entrenamiento)
    plt.title('Recompensa acumulada por episodio durante el entrenamiento')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Total')
    plt.grid(True)
    img_buffer = io.BytesIO() 
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    url_grafico_base64 = base64.b64encode(img_buffer.getvalue()).decode() 
    plt.close()

    tabla_q_para_mostrar = tabla_valores_q.tolist() 

    return jsonify(
        url_grafico=url_grafico_base64, 
        tabla_q=tabla_q_para_mostrar, 
        episodios_entrenados=len(historial_recompensas_entrenamiento), 
        recompensa_promedio_ultimos_100 = np.mean(historial_recompensas_entrenamiento[-100:]).round(2) 
    )

@app.route('/resultados-entrenamiento') 
def mostrar_resultados_entrenamiento(): 
    return render_template('resultados.html') 

@app.route('/simular') 
def simular_comportamiento_agente(): 
    if tabla_valores_q is None:
        return "Por favor, entrena al agente primero para poder simularlo.", 400

    agente_simulacion = AgenteQLearning(nombre_entorno='FrozenLake-v1') 
    agente_simulacion.tabla_q = tabla_valores_q 
    pasos_simulacion = agente_simulacion.simular_politica(num_pasos=20) 

    return render_template('simulacion.html', pasos_simulacion=pasos_simulacion) 

if __name__ == '_main_':
    app.run(debug=True)