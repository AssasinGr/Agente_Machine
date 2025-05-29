import gym
import numpy as np

class AgenteQLearning: 
    def _init_(self, nombre_entorno, tasa_aprendizaje=0.1, factor_descuento=0.99, epsilon=1.0, tasa_decaimiento_epsilon=0.001, epsilon_minimo=0.01):
        self.entorno = gym.make(nombre_entorno, is_slippery=False) 
        self.tasa_aprendizaje = tasa_aprendizaje 
        self.factor_descuento = factor_descuento 
        self.epsilon = epsilon
        self.tasa_decaimiento_epsilon = tasa_decaimiento_epsilon 
        self.epsilon_minimo = epsilon_minimo 

        self.numero_estados = self.entorno.observation_space.n 
        self.numero_acciones = self.entorno.action_space.n 
        self.tabla_q = np.zeros((self.numero_estados, self.numero_acciones)) 

        self.nombre_entorno = nombre_entorno 

    def elegir_accion(self, estado): 
        if np.random.uniform(0, 1) < self.epsilon:
            return self.entorno.action_space.sample() 
        else:
            return np.argmax(self.tabla_q[estado, :]) 

    def aprender(self, estado, accion, recompensa, siguiente_estado, terminado): 
        mejor_siguiente_accion = np.argmax(self.tabla_q[siguiente_estado, :]) 
        objetivo_td = recompensa + self.factor_descuento * self.tabla_q[siguiente_estado, mejor_siguiente_accion] * (1 - int(terminado)) 
        error_td = objetivo_td - self.tabla_q[estado, accion] 
        self.tabla_q[estado, accion] += self.tasa_aprendizaje * error_td

    def entrenar(self, num_episodios=10000, max_pasos_por_episodio=100): 
        recompensas_por_episodio = [] 

        for episodio in range(num_episodios): 
            estado, info = self.entorno.reset()
            terminado = False 
            truncado = False 
            recompensa_episodio_actual = 0 

            for paso in range(max_pasos_por_episodio): 
                accion = self.elegir_accion(estado)
                siguiente_estado, recompensa, terminado, truncado, info = self.entorno.step(accion)
                self.aprender(estado, accion, recompensa, siguiente_estado, terminado)
                estado = siguiente_estado
                recompensa_episodio_actual += recompensa

                if terminado or truncado:
                    break
            
            recompensas_por_episodio.append(recompensa_episodio_actual)

            self.epsilon = max(self.epsilon_minimo, self.epsilon - self.tasa_decaimiento_epsilon)
            
            if (episodio + 1) % 1000 == 0:
                print(f"Episodio {episodio+1}/{num_episodios}, Epsilon: {self.epsilon:.2f}, Recompensa Promedio (últimos 100): {np.mean(recompensas_por_episodio[-100:]):.2f}")

        return recompensas_por_episodio, self.tabla_q

    def obtener_info_entorno(self): 
        if self.nombre_entorno == 'FrozenLake-v1':
            descripcion = self.entorno.desc.tolist() 
            return {
                "name": "FrozenLake-v1",
                "description": "El agente debe navegar por un lago congelado sin caer en agujeros para llegar a la meta.",
                "states": self.numero_estados,
                "actions": ["Izquierda", "Abajo", "Derecha", "Arriba"], 
                "reward_system": "Recompensa de +1 al llegar a la meta, 0 en otro caso.",
                "map": [[caracter.decode('utf-8') for caracter in fila] for fila in descripcion] 
            }
        return {"name": self.nombre_entorno, "description": "Descripción no disponible.", "states": self.numero_estados, "actions": self.numero_acciones}


    def simular_politica(self, num_pasos=20): 
        if self.tabla_q is None:
            raise ValueError("El agente no ha sido entrenado. Por favor, entrena al agente primero.")

        estado, info = self.entorno.reset()
        terminado = False
        truncado = False
        ruta_simulacion = [] 

        for paso in range(num_pasos):
            caracter_celda_actual = self.entorno.desc[estado // int(np.sqrt(self.numero_estados))][estado % int(np.sqrt(self.numero_estados))].decode('utf-8') # 'current_cell_char'
            
            ruta_simulacion.append({
                'paso': paso, 
                'estado': estado,
                'caracter_celda_actual': caracter_celda_actual, 
                'accion_tomada': None, 
                'recompensa': None,
                'siguiente_estado': None, 
                'terminado': terminado 
            })

            accion = np.argmax(self.tabla_q[estado, :])
            
            ruta_simulacion[-1]['accion_tomada'] = self.obtener_nombre_accion(accion) 

            siguiente_estado, recompensa, terminado, truncado, info = self.entorno.step(accion)

            ruta_simulacion[-1]['recompensa'] = recompensa
            ruta_simulacion[-1]['siguiente_estado'] = siguiente_estado
            ruta_simulacion[-1]['terminado'] = terminado

            estado = siguiente_estado
            
            if terminado or truncado:
                caracter_celda_final = self.entorno.desc[estado // int(np.sqrt(self.numero_estados))][estado % int(np.sqrt(self.numero_estados))].decode('utf-8') 
                ruta_simulacion.append({
                    'paso': paso + 1,
                    'estado': estado,
                    'caracter_celda_actual': caracter_celda_final,
                    'accion_tomada': "FIN",
                    'recompensa': recompensa,
                    'siguiente_estado': None,
                    'terminado': terminado
                })
                break
        
        return ruta_simulacion
    
    def obtener_nombre_accion(self, indice_accion): 
        if self.nombre_entorno == 'FrozenLake-v1':
            acciones = ["Izquierda", "Abajo", "Derecha", "Arriba"]
            return acciones[indice_accion]
        return f"Acción {indice_accion}"