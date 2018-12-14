import numpy as np
from physics_sim import PhysicsSim
from task import Task

class MyTask(Task):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Starting distance
        self.original_distance = np.sqrt(
            (target_pos[0]-init_pose[0])**2 +
            (target_pos[1]-init_pose[1])**2 +
            (target_pos[2]-init_pose[2])**2
        )

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        current_position = self.sim.pose[:3]
        distance = np.sqrt(
            (current_position[0]-self.target_pos[0])**2 +
            (current_position[1]-self.target_pos[1])**2 +
            (current_position[2]-self.target_pos[2])**2
        )
        return self.original_distance - distance
