class DataLogger(object):
    def __init__(self):
        self.data_dict = {}
        self.attribute_dict = {}

    def save_data(self, step, type, data):
        self.data_dict[(step, type)] = data

    def save_attribute(self, type, attribute):
        self.attribute_dict[type] = attribute