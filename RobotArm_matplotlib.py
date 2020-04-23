"""
Created on Thu Apr 23 12:43:29 2020

@author: Emily Brady
"""
%matplotlib inline
from math import cos, sin, sqrt
import numpy as np
import matplotlib.pyplot as plt

#creates three joint arm
class ThreeLinkArm:
    def __init__(self, joint_angles=[0, 0, 0]):
        self.shoulder = np.array([0, 0])
        self.link_lengths = [1, 1, 1]
        self.update_joints(joint_angles)

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.forward_kinematics()

    def forward_kinematics(self):
        theta0 = self.joint_angles[0]
        theta1 = self.joint_angles[1]
        theta2 = self.joint_angles[2]
        l0 = self.link_lengths[0]
        l1 = self.link_lengths[1]
        l2 = self.link_lengths[2]
        self.elbow = self.shoulder + np.array([l0*cos(theta0), l0*sin(theta0)])
        self.wrist = self.elbow + np.array([l1*cos(theta0 + theta1), l1*sin(theta0 + theta1)])
        self.finger = self.wrist + np.array([l2*cos(theta0 + theta1 + theta2), l2*sin(theta0 + theta1 + theta2)])

    def plot(self):
        plt.plot([self.shoulder[0], self.elbow[0],],
                 [self.shoulder[1], self.elbow[1]],
                 'r-')
        plt.plot([self.elbow[0], self.wrist[0]],
                 [self.elbow[1], self.wrist[1]],
                 'r-')
        plt.plot([self.wrist[0], self.finger[0]], 
                 [self.wrist[1], self.finger[1]],
                 'r-')
        
        plt.plot(self.shoulder[0], self.shoulder[1], 'ko')
        plt.plot(self.elbow[0], self.elbow[1], 'ko')
        plt.plot(self.wrist[0], self.wrist[1], 'ko')
        plt.plot(self.finger[0], self.finger[1], 'ko')

'''functions to help draw the angles
did not modify these in any way'''
def transform_points(points, theta, origin):
    T = np.array([[cos(theta), -sin(theta), origin[0]],
                  [sin(theta), cos(theta), origin[1]],
                  [0, 0, 1]])
    return np.matmul(T, np.array(points))

def draw_angle(angle, offset=0, origin=[0, 0], r=0.5, n_points=100):
        x_start = r*cos(angle)
        x_end = r
        dx = (x_end - x_start)/(n_points-1)
        coords = [[0 for _ in range(n_points)] for _ in range(3)]
        x = x_start
        for i in range(n_points-1):
            y = sqrt(r**2 - x**2)
            coords[0][i] = x
            coords[1][i] = y
            coords[2][i] = 1
            x += dx
        coords[0][-1] = r
        coords[2][-1] = 1
        coords = transform_points(coords, offset, origin)
        plt.plot(coords[0], coords[1], 'k-')

'''create arm object '''
arm = ThreeLinkArm()

theta0 = 0.5
theta1 = 1
theta2 = 1

arm.update_joints([theta0, theta1, theta2])
arm.plot()

def label_diagram():
    plt.plot([0, 0.5], [0, 0], 'k--')
    plt.plot([arm.elbow[0], arm.elbow[0]+0.5*cos(theta0)],
             [arm.elbow[1], arm.elbow[1]+0.5*sin(theta0)],
             'k--')
    
    #        [arm.wrist[1], arm.wrist[1]+0.5*sin(theta2)]

    draw_angle(theta0, r=0.25)
    draw_angle(theta1, offset=theta0, origin=[arm.elbow[0], arm.elbow[1]], r=0.25)
    draw_angle(theta2, offset=theta1+theta0, origin=[arm.wrist[0], arm.wrist[1]], r=0.25)

    plt.annotate("$l_0$", xy=(0.5, 0.4), size=15, color="r")
    plt.annotate("$l_1$", xy=(0.8, 1), size=15, color="r")
    plt.annotate("$l_2$", xy=(0.7, 1.75), size=15, color="r")

    plt.annotate(r"$\theta_0$", xy=(0.35, 0.05), size=15)
    plt.annotate(r"$\theta_1$", xy=(1, 0.8), size=15)
    plt.annotate(r"$\theta_2$", xy=(1, 1.75), size=15)

label_diagram()

plt.annotate("Shoulder", xy=(arm.shoulder[0], arm.shoulder[1]), xytext=(0.15, 0.5),
    arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate("Elbow", xy=(arm.elbow[0], arm.elbow[1]), xytext=(1.25, 0.25),
    arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate("Wrist", xy=(arm.wrist[0], arm.wrist[1]), xytext=(1.5, 1.5),
    arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate("Finger", xy=(arm.finger[0], arm.finger[1]), xytext=(0, 1.5),
    arrowprops=dict(facecolor='black', shrink=0.05))


plt.axis("equal")

plt.show()















