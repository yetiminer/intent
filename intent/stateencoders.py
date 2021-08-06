import numpy as np
from collections import namedtuple

FML_State=namedtuple('FML_State',['fire','exit','ewop','in_queue','q_pos','q_len','q_arrivals','q_departures'])

StateFormat=namedtuple('StateFormat',['fire','exit','ewop','paid','in_queue','q_pos','q_len'])  

class StateEncoder():
    def  __init__(self):
        pass
    
    @staticmethod
    def check_exit(fa_state):
        exit=False
        ewop=False
        paid=False
        
        if fa_state['exits']!=[]:
            for k, li in fa_state['exits'][0].items():
                if 0 in li[0,:]:
                    exit=True
                    if fa_state['exits'][1]=='Paid': paid=True
                    elif fa_state['exits'][1]=='EWOP': ewop=True
                    break

        return np.array((exit,ewop,paid))
    
    @staticmethod
    def get_position(fa_state):
        wi=fa_state['where']   
        wia=np.zeros(3)#####!!!!
        if wi!=(0,0): wia[0]=1
        #remember the first queue is labelled 1
        wia[wi[0]]=wi[1] #returns zeros if given zeros
        return wia
        
    @staticmethod    
    def get_queue_length(fa_state):
        return np.array([len(q) for _,q in fa_state['queues'].items()],dtype=int)
        
    @staticmethod
    def fire(fa_state):
        return np.array(fa_state['fire'],dtype=int)
    
    @staticmethod
    def encode_state(fa_state,**kwargs):
        return np.hstack((StateEncoder.fire(fa_state),StateEncoder.check_exit(fa_state),
        StateEncoder.get_position(fa_state),
        StateEncoder.get_queue_length(fa_state)))
        
class StateEncoderArray():
    def __init__(self):
        pass
    
    @property
    def _fields(self):
        return FML_State._fields
        
    def encode_state(fa_state,flatten=False):
        d=np.hstack([np.expand_dims(a,1) if len(a.shape)==1 else a for a in fa_state]).data
        if flatten:
            return d.astype(int).flatten()
        else:
            return d.astype(int)
