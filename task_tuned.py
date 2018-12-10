import numpy as np
from physics_sim import PhysicsSim
from task import Task

class MyTask(Task):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        current_position = self.sim.pose[:3]
        distance = np.sqrt(
            (current_position[0]-self.target_pos[0])**2 +
            (current_position[1]-self.target_pos[1])**2 +
            (current_position[2]-self.target_pos[2])**2
        )

        if distance < 10:
            return 100 - distance
        else:
            return -1 * distance
