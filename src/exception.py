import sys
import traceback
from src.logger import logging

def error_message_detail(error, error_detail):
    """
    Generate a detailed error message with file name, line number, and error description
    
    Args:
        error (Exception): The original error/exception
        error_detail (sys.exc_info()): Traceback information
    
    Returns:
        str: Formatted error message
    """
    try:
        # Extract traceback information
        _, _, exc_tb = sys.exc_info() if error_detail is None else error_detail
        
        # Get file name and line number
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        # Create detailed error message
        error_message = (
            "Error occurred in python script name [{0}] "
            "line number [{1}] "
            "error message [{2}]"
        ).format(
            file_name,
            line_number,
            str(error)
        )
        
        # Optional: Log the full traceback for more detailed debugging
        logging.error(f"Full Traceback:\n{traceback.format_exc()}")
        
        return error_message
    
    except Exception as e:
        # Fallback error message in case of any issues in error handling
        return f"An error occurred while generating error details: {str(e)}"

class CustomException(Exception):
    """
    Custom exception class to provide more detailed error information
    """
    def __init__(self, error_message, error_detail=None):
        """
        Initialize the custom exception
        
        Args:
            error_message (str or Exception): Error message or exception object
            error_detail (sys.exc_info(), optional): Traceback information
        """
        # Ensure error_message is converted to string
        if not isinstance(error_message, str):
            error_message = str(error_message)
        
        # Call parent class constructor
        super().__init__(error_message)
        
        # Generate detailed error message
        self.error_message = error_message_detail(
            error_message, 
            error_detail or sys.exc_info()
        )
    
    def __str__(self):
        """
        String representation of the exception
        
        Returns:
            str: Detailed error message
        """
        return self.error_message