import numpy as np
from abc import ABC, abstractmethod
from collections import deque

class Stabilizer(ABC):
    @abstractmethod
    def stabilize(self, gaze_x, gaze_y):
        """
        Stabilize the gaze data.
        :param gaze_x: Gaze x-coordinates
        :param gaze_y: Gaze y-coordinates
        :return: Stabilized gaze x and y coordinates
        """
        pass
    
class DisplaceGmmStabilizer(Stabilizer):
    """
    This stabilizer classifies the displacement distribution into fixation and saccade. 
    It maintains a history of gaze coordinates and computes the displacement between 
    consecutive time points. A Gaussian Mixture Model (GMM) is then fitted to the 
    displacement distribution. Once the model is trained, it can be used to classify 
    incoming gaze data as either fixation or saccade based on the learned distribution.
    """
    def __init__(self, n_components=2, n_history=500, refit_interval=20):
        self.n_components = n_components
        self.n_history = n_history
        self.history = deque(maxlen=n_history)
        self.refit_interval = refit_interval
        self.sample_count = 0
        self.gmm = None
        self.fitted = False
        self.dispXY = np.zeros((1, 2))

    def fit(self):
        """
        Fit the GMM to the displacement data.
        """
        from sklearn.mixture import GaussianMixture

        # Calculate displacements
        displacements = np.diff(np.array(self.history), axis=0)
        
        # Fit GMM
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=42)
        self.gmm.fit(displacements)
        self.fitted = True  
    
    def predict(self, gaze_x, gaze_y):
        """
        Predict the class of the incoming gaze data.
        :param gaze_x: Gaze x-coordinates
        :param gaze_y: Gaze y-coordinates
        :return: Class label (fixation or saccade)
        """
        if not self.fitted:
            raise RuntimeError("GMM not fitted yet")
        
        # Calculate displacement
        displacement = np.array([gaze_x, gaze_y]) - self.history[-1]
        
        # Predict using GMM
        probs = self.gmm.predict_proba(displacement.reshape(1, -1))
        return 'fixation' if probs[0][0] > probs[0][1] else 'saccade'
    
    def stabilize(self, gaze_x, gaze_y):
        """
        Append gaze to history, refit every refit_interval samples, then classify.
        """
        self.sample_count += 1
        self.history.append((gaze_x, gaze_y))
        # initial or periodic refit
        if (not self.fitted or (self.sample_count % self.refit_interval == 0)) and self.sample_count >= self.refit_interval:
            self.fit()

        if not self.fitted:
            self.dispXY = np.array([gaze_x, gaze_y])
            return gaze_x, gaze_y
        
        label = self.predict(gaze_x, gaze_y)
        if label == 'fixation':
            return self.dispXY[0], self.dispXY[1]
        else:
            self.dispXY = np.array([gaze_x, gaze_y])
            return gaze_x, gaze_y


class MovingAverageStabilizer(Stabilizer):
    """
    This stabilizer uses a moving average filter to smooth the gaze data.
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def stabilize(self, gaze_x, gaze_y):
        """
        Stabilize the gaze data using a moving average filter.
        :param gaze_x: Gaze x-coordinates
        :param gaze_y: Gaze y-coordinates
        :return: Stabilized gaze x and y coordinates
        """
        self.history.append((gaze_x, gaze_y))
        if len(self.history) < self.window_size:
            return gaze_x, gaze_y
        
        avg_x = np.mean([pos[0] for pos in self.history])
        avg_y = np.mean([pos[1] for pos in self.history])
        
        return avg_x, avg_y
