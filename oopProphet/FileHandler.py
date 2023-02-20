import json
import pandas as pd

'''
IOHandler class is responsible for handling input and output operations for the project. 
It includes methods for reading in data from a file, writing data to a file, and 
creating directories for output files. It also includes a method for creating a timestamp
string that can be appended to output files for tracking purposes.

The IOHandler class has the following attributes:

    None
    
The class has the following methods:

    read_csv:      Reads data from a CSV file given a filename and returns a Pandas DataFrame.
    
    write_csv:     Writes data to a CSV file given a filename and a Pandas DataFrame.
    
    write_json:    Writes data to a JSON file given a filename and a dictionary.

To use the class, create an instance of the IOHandler class and call its methods.

    io_handler = IOHandler()
    data = io_handler.read_csv('data.csv')
    io_handler.write_csv('data_out.csv', data)
    io_handler.write_json('data.json', {'key': 'value'})
'''

class IOHandler:
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
    
    def read_json(self):
        with open(self.input_file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def write_json(self, data):
        with open(self.output_file_path, 'w') as f:
            json.dump(data, f)
            
    def read_csv(self):
        data = pd.read_csv(self.input_file_path)
        return data
    
    def write_csv(self, data):
        data.to_csv(self.output_file_path, index=False)


if __name__ == '__main__':
    io_handler = IOHandler('./data/AAPL_2021_02_08.csv', './data/AAPL_2021_02_14.json')
    data = io_handler.read_csv()
    io_handler.write_json(data)
    data = io_handler.read_json()
    io_handler.write_csv(data)