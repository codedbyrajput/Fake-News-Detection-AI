"""
myExceptions.py

Custom exceptions used across the fake news detection project.
"""


class DataFormatException(Exception):
    """Raised when input data is missing required fields or is in an invalid format."""

    def __init__(self, message: str = "Input data is not in an acceptable format."):
        super().__init__(message)


class ModelTrainingException(Exception):
    """Raised when model training or inference is attempted in an invalid state."""

    def __init__(self, message: str = "Model training/inference error."):
        super().__init__(message)


class PredictionException(ModelTrainingException):
    """Raised when a prediction fails unexpectedly."""

    def __init__(self, message: str = "Error occurred while predicting."):
        super().__init__(message)


class ProcessingDataTypeException(Exception):
    """Raised when data passed to preprocessing is not of the expected type."""

    def __init__(self, message: str = "Expected a string input for text processing."):
        super().__init__(message)
