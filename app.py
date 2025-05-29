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

agente_entrenado_instancia = None 

def setup_environment_info():
    global informacion_entorno
    temp_agent = AgenteQLearning(nombre_entorno='FrozenLake-v1')
    informacion_entorno = temp_agent.obtener_info_entorno()
setup_environment_info() 

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
    global historial_recompensas_entrenamiento, tabla_valores_q, informacion_entorno, agente_entrenado_instancia

    data = request.json
    tasa_aprendizaje = float(data.get('tasa_aprendizaje', 0.1))
    factor_descuento = float(data.get('factor_descuento', 0.99))
    tasa_decaimiento_epsilon = float(data.get('tasa_decaimiento_epsilon', 0.001))
    numero_episodios = int(data.get('numero_episodios', 10000))

    agente_entrenado_instancia = AgenteQLearning(
        nombre_entorno='FrozenLake-v1', 
        tasa_aprendizaje=tasa_aprendizaje,
        factor_descuento=factor_descuento,
        tasa_decaimiento_epsilon=tasa_decaimiento_epsilon
    )
    historial_recompensas_entrenamiento, tabla_valores_q = agente_entrenado_instancia.entrenar(num_episodios=numero_episodios)
    
    plt.figure(figsize=(10, 6))
    plt.plot(historial_recompensas_entrenamiento)
    plt.title('Recompensa acumulada por episodio durante el entrenamiento')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Total')
    plt.grid(True)
    plt.tight_layout() 

    img_buffer = io.BytesIO() 
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    url_grafico_base64 = base64.b64encode(img_buffer.getvalue()).decode() 
    plt.close()

    tabla_q_para_mostrar = tabla_valores_q.tolist() 

    if len(historial_recompensas_entrenamiento) >= 100:
        recompensa_promedio_ultimos_100 = np.mean(historial_recompensas_entrenamiento[-100:]).round(2)
    else:
        recompensa_promedio_ultimos_100 = np.mean(historial_recompensas_entrenamiento).round(2)

    return jsonify(
        url_grafico=url_grafico_base64, 
        tabla_q=tabla_q_para_mostrar, 
        episodios_entrenados=len(historial_recompensas_entrenamiento), 
        recompensa_promedio_ultimos_100 = recompensa_promedio_ultimos_100 
    )

@app.route('/resultados-entrenamiento') 
def mostrar_resultados_entrenamiento(): 
    return render_template('resultados.html') 

@app.route('/simular', methods=['GET']) 
def simular_comportamiento_agente(): 
    global tabla_valores_q, informacion_entorno, agente_entrenado_instancia

    agente_entrenado = (tabla_valores_q is not None) and (np.any(tabla_valores_q != 0))

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        if not agente_entrenado:
            return jsonify({'error': 'Agente no entrenado. Por favor, entrena un agente primero.'}), 400

        if agente_entrenado_instancia is not None and agente_entrenado_instancia.tabla_q is not None:
            agente_simulacion = agente_entrenado_instancia
        else:
            agente_simulacion = AgenteQLearning(nombre_entorno='FrozenLake-v1')
            agente_simulacion.tabla_q = tabla_valores_q 

        pasos_simulacion, recompensa_final_simulacion = agente_simulacion.simular_politica(num_pasos=100) 

        return jsonify(
            path=pasos_simulacion, 
            final_reward=recompensa_final_simulacion
        )
    return render_template('simulacion.html', 
                           agente_entrenado=agente_entrenado, 
                           env_info=informacion_entorno) 

if __name__ == '_main_':
    app.run(debug=True)