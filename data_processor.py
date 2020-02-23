import ast
import csv
import os
from datetime import datetime
import math

import torch


class DataProcessor:
    def __init__(self, time_period, lat_grid_num, lon_grid_num):
        self.file_path = os.getenv("FILE_PATH", "data/crimes.csv")
        self.time_period = time_period
        self.lat_grid_num = lat_grid_num
        self.lon_grid_num = lon_grid_num
        self.min_lat, self.max_lat = 36.619446395, 42.022910333
        self.min_long, self.max_long = -91.686565684, -87.524529378
        self.lat_grid_range = float(0)
        self.lon_grid_range = float(0)
        self.grid_values = None
        self.time_period_range = 0
        self.start_time = 978339600
        self.end_time = 1581713400
        self.data = []
        self.flat_data = []

    def read_data_from_csv(self):
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    if row[19] and row[20]:
                        lat, long = self.row_to_grid(row)
                        if lat and long:
                            period = self.row_to_period(row)
                        self.data[period - 1][lat - 1][long - 1][-1] += 1
                line_count += 1

    def read_to_tensor(self):
        self.divide_grid_descriptions()
        self.divide_time_periods()
        print("Data divided to time periods.")
        print("Data structure created.")
        self.read_data_from_csv()
        print("Data distributed into data structure.")
        self.flat_read_data()
        print("Data flattened.")
        return self.data_to_tensor()

    def read_to_file(self, file_name):
        self.divide_grid_descriptions()
        self.divide_time_periods()
        print("Data divided to time periods.")
        print("Data structure created.")
        self.read_data_from_csv()
        print("Data distributed into data structure.")
        self.flat_read_data()
        print("Data flattened.")
        with open(file_name, 'w') as csv_file:
            csv_file.writelines("%s\n" % crime_grid for crime_grid in self.flat_data)
        print("Data written into " + str(file_name) + ".")

    def data_to_tensor(self):
        tensor = torch.zeros(self.time_period_range, self.lat_grid_num, self.lon_grid_num, 1)
        for row in self.flat_data:
            tensor[row[0], row[1], row[2], :] = row[-1]
        return tensor

    def read_file_to_tensor(self, file_name):
        with open(file_name, mode='r') as file:
            lines = file.readlines()
            self.time_period_range = ast.literal_eval(lines[-1])[0] + 1
            tensor = torch.zeros(self.time_period_range, self.lat_grid_num, self.lon_grid_num, 1)
            for line in lines:
                line = ast.literal_eval(line)
                tensor[line[0], line[1], line[2], :] = line[-1]
        print("Data transformed to torch.tensor")
        return tensor

    def flat_read_data(self):
        for period in self.data:
            for lat in period:
                for lon in lat:
                    self.flat_data.append(lon)
        del self.data

    def divide_time_periods(self):
        date_range = self.end_time - self.start_time
        date_range_hour = int(self.second_to_hour(date_range))
        self.time_period_range = int(date_range_hour / self.time_period)
        for period in range(self.time_period_range):
            self.data.append(self.get_grid_matrix(period))

    def divide_grid_descriptions(self):
        self.lat_grid_range = float((self.max_lat - self.min_lat) / self.lat_grid_num)
        self.lon_grid_range = float((self.min_long - self.max_long) / self.lon_grid_num) * -1
        self.grid_values = [[dict() for i in range(self.lon_grid_num)] for j in range(self.lat_grid_num)]

        for lat in range(self.lat_grid_num):
            for long in range(self.lon_grid_num):
                self.grid_values[lat][long]["start_lat"] = self.min_lat + (lat * self.lat_grid_range)
                self.grid_values[lat][long]["end_lat"] = self.min_lat + ((lat + 1) * self.lat_grid_range)

                self.grid_values[lat][long]["start_long"] = self.min_long + (long * self.lon_grid_range)
                self.grid_values[lat][long]["end_long"] = self.min_long + ((long + 1) * self.lon_grid_range)

    def find_lat_long_and_time_interval(self):
        self.min_lat, self.max_lat = 0.0, 0.0
        self.min_long, self.max_long = 0.0, -10000.0
        self.start_time = 0
        self.end_time = 0
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    if row[19]:
                        if self.min_lat > float(row[19]):
                            self.min_lat = float(row[19])
                        if self.max_lat <= float(row[19]):
                            self.max_lat = float(row[19])
                    if row[20]:
                        if self.min_long > float(row[20]):
                            self.min_long = float(row[20])
                        if self.max_long < float(row[20]):
                            self.max_long = float(row[20])
                    date = int(self.get_time(row))
                    if line_count == 1:
                        self.start_time = date
                    else:
                        if date > self.end_time:
                            self.end_time = date
                line_count += 1
            print("Coordinates of Data: \n" + "Lat: " + str(self.min_lat) + ", " + str(self.max_lat) + "\nLong: " +
                  str(self.min_long) + ", " + str(self.max_long))
            print("Start Time of Data: " + str(self.start_time) + "\n" + "End Time of Data: " + str(self.end_time))

    def get_grid_matrix(self, period):
        return [[[period, m, n, 0] for n in range(self.lon_grid_num)] for m in range(self.lat_grid_num)]

    def row_to_period(self, row):
        if row[2]:
            time = self.get_time(row)
            time_period = self.second_to_hour(time - self.start_time)
            period = math.floor(time_period / self.time_period)
            return period

    def row_to_grid(self, row):
        diff_lat = float(row[19]) - self.min_lat
        lat_grid = math.floor(diff_lat / self.lat_grid_range)
        diff_long = float(row[20]) - self.min_long
        long_grid = math.floor(diff_long / self.lon_grid_range)
        return lat_grid, long_grid

    @staticmethod
    def second_to_hour(time):
        return time / 60 / 60

    @staticmethod
    def get_time(row):
        return int(datetime.strptime(row[2], "%m/%d/%Y %I:%M:%S %p").timestamp())
