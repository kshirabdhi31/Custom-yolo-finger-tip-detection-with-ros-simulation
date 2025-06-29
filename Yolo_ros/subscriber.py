#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point, Twist
from turtlesim.msg import Pose  
import math
from turtlesim.srv import Spawn, Kill  
import cv2  

class TurtleController:
    def __init__(self):
        rospy.init_node('turtle_controller', anonymous=True)

        rospy.Subscriber('/fingertip_coordinates', Point, self.coordinate_callback)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=35) 
        self.is_first_goal = True
        self.goal_list = []  
        self.Kp = 1.8
        self.Kd = 0.8 
        self.derivative_term = 0.0
        self.previous_error = 0.0
        self.turtle_pose = None
        rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)
        self.goal_counter = 0 
        self.spawn_service = rospy.ServiceProxy('/spawn', Spawn)  
        self.kill_service = rospy.ServiceProxy('/kill', Kill) 

    def pose_callback(self, pose_msg):
        self.turtle_pose = pose_msg

    def coordinate_callback(self, msg):
        self.goal_list.append((msg.x, msg.y))
        print("Received Coordinate:", msg.x, msg.y) 


    def move_turtle(self):
        if self.goal_list and self.turtle_pose is not None:
            self.goal_counter = (self.goal_counter + 1) % 2 

            if self.goal_counter == 0 and self.is_first_goal:  
                current_goal = self.goal_list[0] 
                self.goal_x = current_goal[0]
                self.goal_y = current_goal[1]

# Code for spawning the turtle to the first publsihedcoordinate
                try:
                    self.kill_service('turtle1')  
                except rospy.ServiceException as e:
                    rospy.logerr("Kill service call failed: %s", e)

                try:
                    self.spawn_service(self.goal_x,self.goal_y, 0,'turtle1')
                    self.is_first_goal = False
                except rospy.ServiceException as e:
                    rospy.logerr("Spawn service call failed: %s", e)
                print("Spawn Logic Executed")
                self.is_first_goal = False
                print("is_first_goal set to:", self.is_first_goal)
            else:
                
                self.goal_x = self.goal_list[0][0]  
                self.goal_y = self.goal_list[0][1]  
                distance_to_goal = math.sqrt((self.goal_x - self.turtle_pose.x)**2 + (self.goal_y - self.turtle_pose.y)**2)
                angle_to_goal = math.atan2(self.goal_y - self.turtle_pose.y, self.goal_x - self.turtle_pose.x)

                # angle difference to be within -pi to pi range
                angle_diff = angle_to_goal - self.turtle_pose.theta 
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi 
                elif angle_diff < -math.pi:
                    angle_diff += 2 * math.pi 

                # PD calculation for angular speed
                error = angle_diff
                self.derivative_term = error - self.previous_error
                angular_speed = self.Kp * error + self.Kd * self.derivative_term
                self.previous_error = error
                linear_speed = 1.6 * distance_to_goal 
                twist_msg = Twist()
                twist_msg.linear.x = linear_speed
                twist_msg.angular.z = angular_speed  
                self.velocity_publisher.publish(twist_msg)

                if distance_to_goal < 0.1: 
                    self.goal_list.pop(0)  
    def run(self):
        rate = rospy.Rate(25)  
        while not rospy.is_shutdown():
            self.move_turtle()
            key = cv2.waitKey(1)  
            if key == 27: 
                break
            rate.sleep()

if __name__ == '__main__':
    controller = TurtleController()
    controller.run()