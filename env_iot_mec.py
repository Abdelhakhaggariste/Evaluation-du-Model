import numpy as np
import gymnasium as gym
from gymnasium import spaces

class IoTMECEnv(gym.Env):
    def __init__(self, n_iot=10):
        super(IoTMECEnv, self).__init__()
        
        self.n_iot = n_iot
        self.state_dim = 6  # Nombre de caractéristiques par IoT
        
        # Espace d'observation: état de tous les IoT
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(n_iot * self.state_dim,),  # État aplati: (n_iot * state_dim,)
            dtype=np.float32
        )
        
        # Espace d'action: action pour chaque IoT
        self.action_space = spaces.MultiDiscrete([2] * n_iot)  # 0=local, 1=MEC
        
        # Paramètres
        self.max_steps = 20
        self.current_step = 0
        
        # Initialisation des caractéristiques des IoT
        self._init_device_characteristics()
        
    def _init_device_characteristics(self):
        """Initialise les caractéristiques des dispositifs IoT"""
        self.device_characteristics = []
        for i in range(self.n_iot):
            device = {
                'id': i,
                'cpu_capacity': np.random.uniform(0.5, 2.0),
                'battery_level': np.random.uniform(0.3, 1.0),
                'task_size': np.random.uniform(0.5, 5.0),
                'cpu_required': np.random.uniform(0.1, 0.5),
                'deadline': np.random.uniform(30, 150),
                'channel_quality': np.random.uniform(0.5, 1.0)
            }
            self.device_characteristics.append(device)
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self._init_device_characteristics()
        
        # Initialiser l'état
        state = self._get_state()
        
        info = {}
        return state, info
    
    def _get_state(self):
        """Construit le vecteur d'état"""
        state_vector = []
        
        for device in self.device_characteristics:
            # Normaliser chaque caractéristique entre 0 et 1
            state_vector.extend([
                device['cpu_capacity'] / 2.0,          # 0-1
                device['battery_level'],              # 0-1
                device['task_size'] / 5.0,            # 0-1
                device['cpu_required'],               # 0-1 déjà
                device['deadline'] / 150.0,           # 0-1
                device['channel_quality']             # 0-1 déjà
            ])
        
        return np.array(state_vector, dtype=np.float32)
    
    def step(self, action):
        """Exécute une étape"""
        self.current_step += 1
        
        # Calculer les métriques
        total_energy = 0.0
        total_latency = 0.0
        successful_tasks = 0
        offloaded_tasks = 0
        
        # Mettre à jour les caractéristiques des dispositifs
        for i, device in enumerate(self.device_characteristics):
            action_taken = action[i] if i < len(action) else 0
            
            if action_taken == 0:  # Local
                energy = device['cpu_required'] * 0.5 * (1.0 - device['battery_level'])
                latency = (device['task_size'] * 8) / (device['cpu_capacity'] * 1000)
                offloaded_tasks += 0
            else:  # MEC
                energy = 0.05 + device['task_size'] * 0.01
                latency = (device['task_size'] * 8) / (0.8 * 1000)  # MEC CPU = 0.8 GHz
                offloaded_tasks += 1
            
            # Vérifier le délai
            if latency <= device['deadline']:
                successful_tasks += 1
            
            total_energy += energy
            total_latency += latency
            
            # Mettre à jour la batterie
            device['battery_level'] = max(0.1, device['battery_level'] - 0.02)
        
        # Calculer les moyennes
        avg_energy = total_energy / self.n_iot
        avg_latency = total_latency / self.n_iot
        success_rate = successful_tasks / self.n_iot
        offload_rate = offloaded_tasks / self.n_iot
        
        # Calculer la récompense
        reward = self._calculate_reward(avg_energy, avg_latency, success_rate, offload_rate)
        
        # Vérifier si l'épisode est terminé
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Générer le nouvel état
        next_state = self._get_state()
        
        # Informations
        info = {
            'energy': avg_energy,
            'latency': avg_latency,
            'success_rate': success_rate,
            'offload_rate': offload_rate,
            'total_energy': total_energy,
            'step': self.current_step
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _calculate_reward(self, energy, latency, success_rate, offload_rate):
        """Calcule la récompense multi-objectif"""
        # Normaliser
        energy_norm = 1.0 - min(energy / 0.5, 1.0)  # Énergie max ~0.5 J
        latency_norm = 1.0 - min(latency / 100.0, 1.0)  # Latence max ~100ms
        
        # Récompense pondérée
        reward = (
            0.35 * energy_norm +
            0.35 * latency_norm +
            0.20 * success_rate +
            0.10 * offload_rate  # Bonus pour équilibre de charge
        )
        
        return reward
    
    def render(self):
        """Affiche l'état courant"""
        print(f"Step: {self.current_step}")
        print(f"État dimension: {self.observation_space.shape}")
        print(f"Action dimension: {self.action_space.nvec}")
    
    def close(self):
        """Ferme l'environnement"""
        pass