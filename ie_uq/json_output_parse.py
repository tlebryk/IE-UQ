import json
from typing import Dict, List, Optional, TypedDict

# Used for field level perplexity calculations

class MaterialsData(TypedDict):
    basemats: Dict[str, str]
    dopants: Dict[str, str]
    dopants2basemats: Dict[str, List[str]]

def parse_materials_json(json_str: str) -> MaterialsData:
    """
    Parse a JSON string containing materials data with arbitrary numbers of basemats and dopants.
    
    Args:
        json_str: A JSON string containing basemats, dopants, and dopants2basemats mappings
        
    Returns:
        MaterialsData: A dictionary containing the parsed data with typed annotations
        
    Raises:
        json.JSONDecodeError: If the JSON string is invalid
        KeyError: If required top-level keys are missing
    """
    try:
        # Parse the JSON string
        data = json.loads(json_str)
        
        # Validate the structure
        required_keys = {'basemats', 'dopants', 'dopants2basemats'}
        if not all(key in data for key in required_keys):
            missing_keys = required_keys - set(data.keys())
            raise KeyError(f"Missing required keys: {missing_keys}")
        
        # Create the typed dictionary
        materials_data: MaterialsData = {
            'basemats': {},
            'dopants': {},
            'dopants2basemats': {}
        }
        
        # Parse basemats (b0, b1, ...)
        materials_data['basemats'] = {
            k: v for k, v in data['basemats'].items()
            if k.startswith('b') and k[1:].isdigit()
        }
        
        # Parse dopants (d0, d1, ...)
        materials_data['dopants'] = {
            k: v for k, v in data['dopants'].items()
            if k.startswith('d') and k[1:].isdigit()
        }
        
        # Parse dopants2basemats mappings
        materials_data['dopants2basemats'] = {
            k: v for k, v in data['dopants2basemats'].items()
            if k.startswith('d') and k[1:].isdigit() and isinstance(v, list)
        }
        
        return materials_data
    
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON string: {str(e)}", e.doc, e.pos)

def collapse_fields(data: MaterialsData) -> list:
    """
    Collapse the basemats and dopants fields into a single list.
    
    Args:
        data: A dictionary containing basemats and dopants data
    """

    # Grab all the items from the MaterialsData object
    basemat_list = list(data['basemats'].items())
    dopant_list = list(data['dopants'].items())
    dopant2basemat_list = list(data['dopants2basemats'].items())

    all_materials = basemat_list + dopant_list + dopant2basemat_list
    return all_materials
