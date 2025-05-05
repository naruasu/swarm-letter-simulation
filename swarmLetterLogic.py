import pygame
import numpy as np
import random
import sys
import pickle
import os
import imageio
from PIL import Image, ImageDraw, ImageFont
import cv2

# CLI inputs
letter = sys.argv[1]
numRobots = int(sys.argv[2])
robotSpeed = float(sys.argv[3])
letterIndex = sys.argv[4]

# Constants
screenWidth, screenHeight = 800, 600
robotRadius = 4
simulationSteps = 700
minDist = 20
safetyRadius = robotRadius + minDist
letterBox = (screenWidth // 2 - 150, screenHeight // 2 - 150, 300, 300)
pathDataFile = "robotPaths.pkl"
gifOutput = f"letter_{letter}.gif"
robotColor = (0, 255, 0)

class Robot:
    def __init__(self, x, y):
        self.position = np.array([x, y], dtype="float64")
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.target = None
        self.path = [self.position.copy()]

    def update(self, others):
        if self.target is None:
            self.path.append(self.position.copy())
            return

        self.acceleration = np.zeros(2)
        toTarget = self.target - self.position
        distanceToTarget = np.linalg.norm(toTarget)
        if distanceToTarget > 1:
            desiredVelocity = (toTarget / distanceToTarget) * robotSpeed
            steeringForce = desiredVelocity - self.velocity
            self.acceleration += steeringForce * 0.05

        for other in others:
            if other is not self and other.target is not None:
                away = self.position - other.position
                d = np.linalg.norm(away)
                if 0 < d < minDist:
                    self.acceleration += (away / d) * (1 / d) * 2

        self.velocity += self.acceleration
        self.position += self.velocity
        self.velocity *= 0.9
        self.path.append(self.position.copy())

    def draw(self, screen):
        pygame.draw.circle(screen, robotColor, self.position.astype(int), robotRadius)

def generateContourLetterTargets(letter, fontPath="arial.ttf", numRobots=25):
    font = ImageFont.truetype(fontPath, 220)
    img = Image.new("L", (300, 300), 0)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), letter, font=font, fill=255)

    img_cv = np.array(img)
    contours, _ = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]

    total_len = len(contour)
    idxs = np.linspace(0, total_len - 1, numRobots).astype(int)
    sampled = contour[idxs]

    x0, y0, w, h = letterBox
    sim_points = []
    for x, y in sampled:
        px = x0 + (x / 300) * w
        py = y0 + (y / 300) * h
        sim_points.append(np.array([px, py], dtype='float64'))

    return np.array(sim_points)

def randomSpawnOutsideBox(num, box):
    x0, y0, w, h = box
    robots = []
    while len(robots) < num:
        x = random.randint(0, screenWidth)
        y = random.randint(0, screenHeight)
        if not (x0 <= x <= x0 + w and y0 <= y <= y0 + h):
            robots.append([x, y])
    return np.array(robots)

pygame.init()
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("Swarm Letter Simulation")
clock = pygame.time.Clock()

targetPoints = generateContourLetterTargets(letter, numRobots=numRobots)
robotPositions = randomSpawnOutsideBox(numRobots, letterBox)
robots = [Robot(x, y) for x, y in robotPositions]

unassignedRobots = set(range(numRobots))
for tIdx in range(len(targetPoints)):
    bestDist = float("inf")
    bestRobotIdx = None
    for rIdx in unassignedRobots:
        dist = np.linalg.norm(robotPositions[rIdx] - targetPoints[tIdx])
        if dist < bestDist:
            bestDist = dist
            bestRobotIdx = rIdx
    if bestRobotIdx is not None:
        robots[bestRobotIdx].target = targetPoints[tIdx]
        unassignedRobots.remove(bestRobotIdx)

center = np.array([screenWidth / 2, screenHeight / 2])
angleStep = 2 * np.pi / max(1, len(unassignedRobots))
radius = 180
for i, rIdx in enumerate(unassignedRobots):
    angle = i * angleStep
    offset = np.array([np.cos(angle), np.sin(angle)]) * radius
    robots[rIdx].target = center + offset

frames = []
threshold = 5
for _ in range(simulationSteps):
    screen.fill((30, 30, 30))
    for robot in robots:
        robot.update(robots)
        robot.draw(screen)
    pygame.display.flip()
    clock.tick(60)
    
    # Save frame
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))
    frames.append(frame.copy())

    # Check convergence
    all_reached = all(
        np.linalg.norm(robot.position - robot.target) < threshold
        for robot in robots if robot.target is not None
    )
    if all_reached:
        print("✅ All robots reached their targets. Ending simulation.")
        break

pygame.quit()

paths = [robot.path for robot in robots]
if os.path.exists(pathDataFile):
    with open(pathDataFile, "rb") as f:
        data = pickle.load(f)
else:
    data = {}
data[letter] = paths
with open(pathDataFile, "wb") as f:
    pickle.dump(data, f)

try:
    imageio.mimsave(gifOutput, frames, fps=30)
except Exception as e:
    print(f"GIF generation error: {e}")