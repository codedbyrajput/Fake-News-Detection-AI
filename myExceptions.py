class DataFormatException(Exception):
    """We throw this exception when the csv file misses any comma or column"""
    def __init__(self):
        message = "Input type is not in acceptable form."
        super().__init__(message) # like in java we do super(message) and the message goes to the parent exception class, in python we do super.__init__(message) where super takes us to the parent class and init helps to access the parent class constructor 



class ModelTrainingException(Exception):
    """Exception is raised when the model training process fails or any invalid attemp in the model training process eg. training with no data or using untrained model to make predictions."""
    message = "something is wrong with the process of model training"
    def __init__(self):
        super.__init__(message)


class PredictionException(ModelTrainingException):
    """raised if error occurs during prediction
        This exception is a subclass exception of ModelTrainingException. Like in java we do class Cat extends Animal we do it like this in python style"""
    message = "Error occured while predicting"
    def __init__(self):
        super.__init__(message)

class ProcessingDataTypeException(Exception):
    """We use this exception if the data which we are trying to read is not string"""
    message = "The data type of the data being read is not string check please."

    def __init__(self):
        super.__init__(message)
