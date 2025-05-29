import gymnasium as gym
import numpy as np

class AgenteQLearning:
    def __init__(self, nombre_entorno, tasa_aprendizaje=0.1, factor_descuento=0.99, epsilon=1.0, tasa_decaimiento_epsilon=0.001, epsilon_minimo=0.01):
        self.entorno = gym.make(nombre_entorno, is_slippery=False)
        self.entorno_base = self.entorno.unwrapped
        self.tasa_aprendizaje = tasa_aprendizaje
        self.factor_descuento = factor_descuento
        self.epsilon = epsilon
        self.tasa_decaimiento_epsilon = tasa_decaimiento_epsilon
        self.epsilon_minimo = epsilon_minimo
        self.numero_estados = self.entorno.observation_space.n
        self.numero_acciones = self.entorno.action_space.n
        self.tabla_q = np.zeros((self.numero_estados, self.numero_acciones))
        self.nombre_entorno = nombre_entorno

        self.action_names_map = {
            0: "Izquierda",
            1: "Abajo",
            2: "Derecha",
            3: "Arriba"
        }
    def elegir_accion(self, estado):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.entorno.action_space.sample()
        else:
            return np.argmax(self.tabla_q[estado, :])

    def aprender(self, estado, accion, recompensa, siguiente_estado, terminado):
        mejor_siguiente_accion_q = np.max(self.tabla_q[siguiente_estado, :])
        objetivo_td = recompensa + self.factor_descuento * mejor_siguiente_accion_q * (1 - int(terminado))
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
                self.aprender(estado, accion, recompensa, siguiente_estado, terminado or truncado)
                estado = siguiente_estado
                recompensa_episodio_actual += recompensa

                if terminado or truncado:
                    break

            recompensas_por_episodio.append(recompensa_episodio_actual)
            self.epsilon = max(self.epsilon_minimo, self.epsilon - self.tasa_decaimiento_epsilon)

            if (episodio + 1) % 1000 == 0:
                avg_reward_last_100 = np.mean(recompensas_por_episodio[-100:]) if len(recompensas_por_episodio) >= 100 else np.mean(recompensas_por_episodio)
                print(f"Episodio {episodio+1}/{num_episodios}, Epsilon: {self.epsilon:.2f}, Recompensa Promedio (últimos 100): {avg_reward_last_100:.2f}")

        return recompensas_por_episodio, self.tabla_q

    def obtener_info_entorno(self):
        if self.nombre_entorno == 'FrozenLake-v1':
            descripcion_bytes = self.entorno_base.desc
            map_chars = [[char.decode('utf-8') for char in fila] for fila in descripcion_bytes]

            return {
                "name": "FrozenLake-v1",
                "description": "El agente debe navegar por un lago congelado sin caer en agujeros para llegar a la meta. S=Inicio, F=Congelado, H=Agujero, G=Meta.",
                "states": int(self.numero_estados),
                "actions": list(self.action_names_map.values()),
                "num_actions": int(self.numero_acciones),
                "actions_map": self.action_names_map,
                "reward_system": "Recompensa de +1 al llegar a la meta (G), 0 en baldosas congeladas (F), y 0 en agujeros (H), pero el episodio termina.",
                "map": map_chars
            }
        return {
            "name": self.nombre_entorno,
            "description": "Descripción no disponible.",
            "states": int(self.numero_estados),
            "actions": [],
            "num_actions": int(self.numero_acciones),
            "actions_map": {},
            "reward_system": "No especificado.",
            "map": []
        }
    def obtener_nombre_accion(self, indice_accion):
        return self.action_names_map.get(indice_accion, f"Acción {indice_accion}")

    def simular_politica(self, num_pasos=100):
        if self.tabla_q is None or not np.any(self.tabla_q != 0):
            raise ValueError("El agente no ha sido entrenado o la tabla Q está vacía.")

        estado, info = self.entorno.reset()
        terminado = False
        truncado = False
        ruta_simulacion = []
        recompensa_total = 0

        for paso in range(num_pasos):
            accion = np.argmax(self.tabla_q[estado, :])

            siguiente_estado, recompensa, terminado, truncado, info = self.entorno.step(accion)
            recompensa_total += recompensa

            ruta_simulacion.append({
                'state': int(estado), 
                'action': self.obtener_nombre_accion(int(accion)), 
                'action_idx': int(accion), 
                'reward': float(recompensa)
            })
            estado = siguiente_estado

            if terminado or truncado:
                ruta_simulacion.append({
                    'state': int(estado), 
                    'action': "FIN",
                    'action_idx': int(-1), 
                    'reward': 0.0
                })
                break
        return ruta_simulacion, float(recompensa_total)