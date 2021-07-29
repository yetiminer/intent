from collections import deque, namedtuple
import numpy as np
from random import random

class TillQueue():
    #class variable
    departure_functions={'poisson':np.random.poisson,
    }


    def __init__(self,name,departure_rate,arrival_rate,departure_func='poisson'):
        self.name=name
        self.q=deque()
        self.entry_times={}
        self.departure_rate=departure_rate
        self.departure_func=self.departure_functions['poisson']
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
        
    def add(self,cust_id,time):        
        self.q.append(cust_id)
        assert cust_id not in self.entry_times
        self.entry_times[cust_id]=time
    
    @property
    def entry_times_list(self):
        #could make use of new python functionality that orders dicts
        return list(self.entry_times.values())
        #return [self.entry_times[cust_id] for cust_id in self.q]
    
    def __entry_time_list(self,list_like):
        #removes the items 
        return [self.entry_times.pop(cust_id) for cust_id in list_like]
        
        
        
    def serve(self,num=None):
        x=[]
        if num is None: num=self.departure_func(self.departure_rate)
        if num>0:
            for i in range(num):
                try:
                    
                    x.append(self.q.popleft())
                    
                    
                except IndexError:
                    x+=[]
        #pop the times from the waiting dictionary
        exit_times=self.__entry_time_list(x)
        
        return x,exit_times
        
    def purge(self):
        exit_list=list(self.q)
        self.q.clear()
        exit_list_times=self.__entry_time_list(exit_list)
        
        return exit_list,exit_list_times
    
    def exit_queue(self,person):
        self.q.remove(person)
        self.entry_times.pop(person)
        
        
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