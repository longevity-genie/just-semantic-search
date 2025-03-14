from typing import Callable, Dict, List, Optional, Type, TypeVar
from pydantic._internal._model_construction import ModelMetaclass

# Define a TypeVar for classes that will use the metaclass
T = TypeVar('T')

def get_param_value(param_names: List[str], args: tuple, kwargs: dict, param_positions=None) -> Optional[str]:
    """
    Utility function to get a parameter value from either kwargs or args.
    
    Args:
        param_names: List of possible parameter names to look for
        args: Tuple of positional arguments
        kwargs: Dictionary of keyword arguments
        param_positions: Optional list of parameter positions to check in the args
    
    Returns:
        The found parameter value or None if not found
    """
    # First check kwargs for any of the parameter names
    for param_name in param_names:
        if param_name in kwargs:
            return str(kwargs[param_name])
    
    # If not found in kwargs and we have positional args info
    if args and param_positions:
        for position in param_positions:
            if len(args) > position:
                return str(args[position])
    
    return None

class IndexMultitonMeta(type):
    """
    A metaclass that keeps track of all instances created for a class using an index_name parameter.
    TODO: decide if it makes sence or we just create a headache for ourselves
    """
    _instances: Dict[str, T] = {}  # Now typed with T instead of object

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        # Try to get index_name from kwargs first
        index_name = kwargs.get('index_name')
        
        if index_name is None and args:
            # Use get_param_value to retrieve index_name from args or kwargs
            index_param_names = ['index_name', 'index']
            index_name = get_param_value(index_param_names, args, kwargs)
            
        if index_name is None:
            raise ValueError("index_name parameter is required for instance creation")
            
        # Convert index_name to string to ensure it can be used as a dictionary key
        index_key = str(index_name)
        
        # Check if an instance with this index_key already exists
        if index_key in cls._instances:
            return cls._instances[index_key]
        
        # If not, create a new instance
        instance = super(IndexMultitonMeta, cls).__call__(*args, **kwargs)
        cls._instances[index_key] = instance
        
        return instance
    

    def get_instances(cls: Type[T]) -> Dict[str, T]:
        """
        Returns all instances created for this class as a dictionary with index keys.
        """
        return cls._instances
    
class PydanticIndexMultitonMeta(ModelMetaclass, IndexMultitonMeta):
    """
    A metaclass that combines Pydantic's ModelMetaclass with IndexMultitonMeta.
    This allows a class to be both a Pydantic model and use the index-based multiton pattern.
    """
    def __call__(cls: Type[T], *args, **kwargs) -> T:
        # Add this line to capture if we're creating a new instance or reusing one
        is_new_instance = str(kwargs.get('index_name', '')) not in cls._instances
        
        # Call the __call__ method of IndexMultitonMeta first to get/create instance
        instance = super(IndexMultitonMeta, cls).__call__(*args, **kwargs)
        
        # Add a flag that model_post_init can check
        instance._is_new_instance = is_new_instance
        
        return instance 
