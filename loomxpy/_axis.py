from enum import Enum
from abc import abstractclassmethod
from typing import MutableMapping

class Axis(Enum):
   OBSERVATIONS = 0
   FEATURES = 1

class AlignedAttribute():

    def __init__(self, key: str, mode_type, axis: Axis, data):
        '''
        '''
        self.mode_type = mode_type
        self.axis = axis
        self.key = key
        self.data = data
    
    def __repr__(self):
        '''
        '''
        return self.data.__repr__()

class ObservationAttribute(AlignedAttribute):

    def __init__(self, key: str, mode_type, data):
        '''
        '''
        super().__init__(key=key, mode_type=mode_type, axis=Axis.OBSERVATIONS, data=data)

class FeatureAttribute(AlignedAttribute):

    def __init__(self, key: str, mode_type, data):
        '''
        '''
        super().__init__(key=key, mode_type=mode_type, axis=Axis.FEATURES, data=data)
    
    def __repr__(self):
        return super().__repr__()

class GlobalAttribute(AlignedAttribute):

    def __init__(self):
        '''
        '''

class AlignedAttributes(MutableMapping[str, AlignedAttribute]):

    def __init__(self, mode_type):
        '''
        '''
        self.mode_type = mode_type
        self._keys = []
    
    def __delitem__(self, key):
        '''
        '''
        self._keys.remove(key)
        delattr(self, key)

    def __getitem__(self, key):
        '''
        '''
        return getattr(self, key)
    
    def __iter__(self):
        '''
        '''
        return iter(AttributesIterator(self))
    
    def __len__(self):
        '''
        '''
        return len(self.keys)
    
   
    def add_item(self, key, value):
        self._keys.append(key)
        setattr(self, key, value)

    @abstractclassmethod
    def __setitem__(self, key, value):
        '''
        '''
        return

class AttributesIterator:

    """Class to implement an iterator
    of powers of two"""

    def __init__(self, attrs: AlignedAttributes):
        self.attrs = attrs

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= len(o=self.attrs):
            self.n += 1
            current_key = self.attrs._keys[self.n]
            return current_key,self.attrs[current_key]
        else:
            raise StopIteration

class FeatureAttributes(AlignedAttributes):

    def __init__(self, mode_type):
        '''
        '''
        super(FeatureAttributes, self).__init__(mode_type=mode_type)

    @staticmethod
    def is_valid_key(key: str):
        return isinstance(key, str) 
    
    @staticmethod
    def is_valid_value(value):
        return True

    def __setitem__(self, key, value):
        '''
        '''
        # if not isinstance(value, FeatureAttribute):
        #     raise Exception(f"Cannot add attribute of type {type(value).__name__} to {type(self).__name__}. Attribute has to be of type {FeatureAttribute}.")
        if not FeatureAttributes.is_valid_key(key=key):
            raise Exception(f"Cannot add attribute of type ({type(key).__name__}, {type(value).__name__}) to {type(self).__name__}. Not a valid key.")
        if not FeatureAttributes.is_valid_value(value=value):
            raise Exception(f"Cannot add attribute of type {type(value).__name__} to {type(self).__name__}. Not a valid value.")
        value = FeatureAttribute(key, self.mode_type, value)
        self.add_item(key=key, value=value)
    
class ObservationAttributes(MutableMapping[str, ObservationAttribute]):

    def __init__(self, mode_type):
        '''
        '''
        super(ObservationAttributes, self).__init__(mode_type=mode_type)

    def __setattr__(self, key, value):
        if not isinstance(value, ObservationAttribute):
            raise Exception(f"Cannot add attribute of type {type(value).__name__} to {type(self).__name__}.")
        setattr(self, key, value)
