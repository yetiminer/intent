from collections import deque, namedtuple
import numpy as np
from random import random

class TillQueue():
    def __init__(self,name,departure_rate,arrival_rate):
        self.name=name
        self.q=deque()
        self.departure_rate=departure_rate
        self.arrival_rate=arrival_rate #for reference
        
    def __repr__(self):
        return str(self.q)
        
    def __eq__(self,other):
        if not isinstance(other,TillQueue):
            return NotImplementedError
        return all((self.name==other.name,self.q==other.q,
        self.departure_rate==other.departure_rate,self.arrival_rate==other.arrival_rate))
    
    @property
    def len(self):
        return len(self.q)
        
    def add(self,pt):
        person,time=pt
        self.q.append((person,time))
        
        
    def serve(self,num=None):
        x=[]
        if num is None: num=np.random.poisson(self.departure_rate)
        if num>0:
            for i in range(num):
                try:
                    x+=[self.q.popleft()]
                    
                except IndexError:
                    x+=[]
                    
        return x
        
    def purge(self):
        exit_list=list(self.q)
        self.q.clear()        
        return exit_list
    
    def exit_queue(self,person):
        self.q.remove(person)
        
        
    def where_is(self,cust_id):
        for idx,cid in enumerate(self.q):
            if cid==cust_id:
                return idx
        
        return IndexError
    
    def __iter__(self):
        return TillQueueIterator(self.q)
    
class TillQueueIterator():
    def __init__(self, data_sequence):
       self.idx = 0
       self.data = data_sequence
    def __iter__(self):
       return self
    def __next__(self):
       self.idx += 1
       try:
           return self.data[self.idx-1]
       except IndexError:
           self.idx = 0
           raise StopIteration  # Done iterating.