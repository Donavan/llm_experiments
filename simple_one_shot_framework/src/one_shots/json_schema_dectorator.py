from typing import, Callable, Dict, Union


def json_schema(description: str, params: Dict[str, Dict[str, Union[str, bool]]]) -> Callable:
    """
    A decorator to attach an OpenAI compatible JSON schema to a function. The schema contains
    information about the function's name, description, parameters, and required parameters.


    :param description: A description of the function.
    :param params: A dictionary containing information about the parameters of the function.
    :return: The original function with an attached JSON schema.
    """

    def decorator(func: Callable) -> Callable:
        # Define the structure of parameters
        parameters = {
            'type': 'object',
            'properties': {}
        }

        # Keep track of required parameters
        required = []

        # Populate the parameters and required lists
        for param, info in params.items():
            if info.get('required', False):
                required.append(param)

            # Populate the properties of each parameter
            parameters['properties'][param] = {
                'type': info.get('type', 'string'),
                'description': info.get('description', '')
            }

        # Define the schema with function name, description, and parameters
        schema = {
            'name': func.__name__,
            'description': description,
            'parameters': parameters
        }

        # Add required parameters to the schema if they exist
        if required:
            schema['required'] = required

        # Attach the schema to the original function
        func.schema = schema

        # Return the original function
        return func

    return decorator
