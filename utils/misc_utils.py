from datetime import datetime

class date_n_time():
    """
    Class to get current date and time in the format: dd_mm_yyyy_hh_mm_ss
    Note: really could just use time.asctime().replace(" ", "_"), it will look better
    """
    def __init__(self):
        self.time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    def __str__(self):
        return self.time_now

class Timer:
    """
    A class to time things.

    Methods:
      start(): starts the timer
      stop(): stops the timer
      lap(lap_name): adds a lap to the timer. If no name is provided, a default name is used
      get_laps(): returns a dictionary of the lap times
      average_time(): returns the average time of all laps on the timer
      total_time(): returns the total time of all laps on the timer
      remove_last(): removes the last lap from the timer
      clear(): clears all laps from the timer
      view(): plots the times
      save(filename): saves all times to a csv file

    Attributes:
      start_time: the time when the timer was started. Restarts for each lap
      times_dict: a dictionary of the times
      default_lap_count: the count of the default lap name
      default_lap_name: the default lap name
    """
    def __init__(self):
        import time
        self.time = time

        import pandas as pd
        self.pd = pd

        import seaborn as sns
        self.sns = sns

        import matplotlib.pyplot as plt
        self.plt = plt

        self.start_time = 0
        self.times_dict = {'lap_name': [], 'lap_time': []}
        self.default_lap_count = 0
        self.default_lap_name = f'unnamed_lap_{self.default_lap_count}'

    def start(self):
        """
        Starts the timer.
        """
        self.start_time = self.time.time()

    def stop(self):
        """
        Stops the timer.
        """
        self.start_time = 0

    def lap(self, lap_name=None):
        """
        Adds a lap to the timer.

        Parameters:
          lap_name (str): the name of the lap. If None, a default name is used
        """
        if lap_name is None:
            lap_name = self.default_lap_name
            self.default_lap_count += 1
            self.default_lap_name = f'unnamed_lap_{self.default_lap_count}'

        self.times_dict['lap_name'].append(str(lap_name))

        lap_time = self.time.time() - self.start_time
        self.times_dict['lap_time'].append(lap_time)
        self.start_time = self.time.time()
        print(f"{lap_name}: {lap_time} seconds")

    def get_laps(self):
        """
        Returns a dictionary of the times.
        """
        return self.times_dict
    
    def get_time(self):
        """
        Returns the time since the timer was started.
        """
        if self.start_time == 0:
            print("Timer not started")
            return
        return self.time.time() - self.start_time
    
    def average_time(self):
        """
        Returns the average time of all laps on the timer.
        """
        return sum(self.times_dict['lap_time']) / len(self.times_dict['lap_time'])
    
    def total_time(self):
        """
        Returns the total time of all laps on the timer.
        """
        return sum(self.times_dict['lap_time'])

    def remove_last(self):
        """
        Removes the last lap from the timer.
        """
        removed_name = self.times_dict['lap_name'].pop()
        removed_lap = self.times_dict['lap_time'].pop()
        print(f"Removed {removed_name}: {removed_lap} seconds")
        del removed_name, removed_lap

    def clear(self):
        """
        Clears all data from the timer.
        """
        if not self.times_dict['lap_name']:
            print("No laps to clear")
            return
        
        self.times_dict['lap_name'] = []
        self.times_dict['lap_time'] = []
        self.default_lap_count = 0
        self.default_lap_name = f'unnamed_lap_{self.default_lap_count}'
        self.start_time = 0
        print("Cleared all laps")

    def view(self):
        """
        Plots the times.
        """
        if not self.times_dict['lap_name']:
            print("No times to view")
            return
        
        fig = self.plt.figure(figsize=(10, 6))
        self.sns.barplot(x='lap_name', y='lap_time', data=self.times_dict, palette='pastel', hue='lap_name')
        fig.show()

    def save(self, filename):
        """
        Saves the times to a csv file.

        Parameters:
          filename (str): the name of the file to save to
        """
        if not self.times_dict['lap_name']:
            print("No times to save")
            return
        
        df = self.pd.DataFrame(self.times_dict)
        df.to_csv(filename, index=False)
        print(f"Saved times to {filename}")
