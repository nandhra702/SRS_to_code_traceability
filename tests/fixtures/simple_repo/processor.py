from helper import HelperClass
from processor import DataProcessor
from main import utility_function

class AdvancedProcessor:
    """
    Summary:
    Manages data processing and result retrieval.

    Description:
    This class manages data processing using `HelperClass` and `DataProcessor`, and retrieves the processed result using `process_result()`.

    Attributes:
    - helper (HelperClass): An instance of `HelperClass` for data processing.
    - data_processor (DataProcessor): An instance of `DataProcessor` for handling internal operations.
    """

    def __init__(self):
        self.helper = HelperClass()
        self.data_processor = DataProcessor()

    def run(self):
        """
        Summary:
        Executes data processing and returns the result.

        Description:
        This method processes data using `self.helper.process_data()`, handles internal operations using `self.data_processor._internal_process()`, and returns the processed result using `self.process_result()`.

        Returns:
        The result of the data processing as returned by `self.process_result()`.
        """
        self.helper.process_data()
        self.data_processor._internal_process()
        return self.process_result()

    def process_result(self):
        """
        Summary:
        Calls the utility function and returns its result.

        Description:
        This method calls the `utility_function()` and returns its result.

        Returns:
        The result of the `utility_function()`.
        """
        return utility_function()