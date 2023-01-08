from pathlib import Path
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
