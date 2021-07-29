import numpy as np
import numpy.ma as ma
from random import random
from collections import namedtuple

FML_state=namedtuple('FML_state',['fire','exit','ewop','in_queue','q_pos','q_len','q_arrivals','q_departures'])



class FishMarketLight():
    
    arrival_func_dic={
    'poisson':np.random.poisson
    }


    def __init__(self,q_num=2,arrival_rate=[1,1],departure_rate=[1,1],arrival_func='poisson',departure_func='poisson',fire_rate=0.1):
        
        if isinstance(arrival_rate,np.ndarray):
            self.__inst_array(arrival_rate=arrival_rate)
        else:
            self.q_num=q_num
            self.instances=1
           
            
            assert len(arrival_rate)==q_num
        self.dims=(self.instances,self.q_num)    
          
        self.arrival_rate=arrival_rate
        self.departure_rate=departure_rate
        self.queues=np.zeros(self.dims)
        self.arrival_func_name=arrival_func
        self.arrival_func=self.arrival_func_dic[arrival_func]
        self.departure_func_name=departure_func
        self.departure_func=self.arrival_func_dic[departure_func]
        self.position=ma.masked_array(np.zeros(self.dims),np.ones(self.dims))
        
        self.arrivals=np.zeros(self.dims)
        self.departures=np.zeros(self.dims)
        
        self.time=0
        self.fire=np.zeros((self.instances,1))
        self.fire_rate=fire_rate
        self.exit=np.zeros((self.instances,1))
        self.EWOP=np.zeros((self.instances,1))
        self.inqueue=np.zeros((self.instances,1))
            
    def __inst_array(self,arrival_rate=None):
        #array form of the market
        self.q_num=arrival_rate.shape[1]
        self.instances=arrival_rate.shape[0]
        
        
    def __repr__(self):
        str1="fire: {}, exit: {}, EWOP: {}, inqueue: {}, position: {}, queues: {},arrivals: {},departures: {}".format(*self.state)
        str2="arrival_rate: {0[arrival_rate]},departure_rate: {0[departure_rate]}, fire_rate {0[fire_rate]}, arrival_func: {0[arrival_func]}, departure_func: {0[departure_func]}".format(self.get_params)
        return str1+str2
        
    
    @property
    def state(self):
        return self.fire.reshape(self.instances,1),self.exit,self.EWOP,self.inqueue, self.position,self.queues,self.arrivals,self.departures
        
    @property
    def pretty_state(self):
        return FML_state(*self.state)
    
    @staticmethod
    def load(pstate,arrival_rate=None,departure_rate=None,fire_rate=None,arrival_func=None,departure_func=None):
        if isinstance(arrival_rate,np.ndarray):
            q_num=arrival_rate.shape[1]
        elif isinstance(arrival_rate,(list,tuple)):
            q_num=len(arrival_rate)
        else:
            print("arrival_rate must be a np array or list or tuple")
            raise error
        FML=FishMarketLight(q_num,arrival_rate=arrival_rate,departure_rate=departure_rate,fire_rate=fire_rate,
        arrival_func=arrival_func,departure_func=departure_func)
        FML.load_state(pstate)
        
        return FML
        
    def copy(self):
        return FishMarketLight.load(self.pretty_state,**self.get_params)
       
    def load_state(self,pstate):
        self.fire=pstate.fire
        self.exit=pstate.exit
        self.ewop=pstate.ewop
        self.inqueue=pstate.in_queue
        self.position=pstate.q_pos
        self.queues=pstate.q_len
        self.arrivals=pstate.q_arrivals
        self.departures=pstate.q_departures
    
    @property
    def get_params(self):
        return dict(
        arrival_rate=self.arrival_rate,
        departure_rate=self.departure_rate,
        fire_rate=self.fire_rate,
        arrival_func=self.arrival_func_name,
        departure_func=self.departure_func_name,
        
        )
        
    @staticmethod
    def _equal(lf1,lf2):
            for a,b in zip(lf1.get_params,lf2.get_params):
                
                try: 
                    assert a.all()==b.all()
                except AssertionError:
                    print(a,b)
                    return False
                except AttributeError:
                    #for the string in the tuple
                    try: assert a==b
                    except AssertionError: return False
                    
            for a,b in zip(lf1.pretty_state,lf2.pretty_state):
                if isinstance(a,np.ma.core.MaskedArray):
                    try: assert np.ma.allequal(a,b)
                    except AssertionError: return False
                else:
                    try: 
                        assert a.all()==b.all()
                    except AssertionError:

                        print(a,b)
                        return False
                    except AttributeError:
                        try: assert a==b
                        except AssertionError: return False
            return True
                    
        
    def decide_fire(self):
        # a process to decide when it is on fire. 
        #fire=np.zeros((self.instances,1))
        fire=np.random.random(self.instances)<self.fire_rate

        return fire
        
    def process_arrivals(self):
        #process other arrivals
        self.arrivals=self.arrival_func(self.arrival_rate,self.dims)
        self.queues+=self.arrivals
        
    def process_departures(self):
        #process departure
        #if  self.fire:
        #when there is a fire everyone departs
        self.departures[self.fire,:]=self.queues[self.fire,:]
        #leaving noone in the queue
            #self.queues[self.fire,:]=np.zeros(self.q_num)
            
        #for the remainder calculute normal departures
        new_departures=self.departure_func(self.departure_rate,self.dims)
        #implicitly using property that departure func is memoryless
        self.departures[~self.fire,:]=new_departures[~self.fire,:]
        
        #for those queues with a fire, noone will be left
        self.queues-=self.departures
        #can only serve the non empty queues
        self.queues=np.clip(self.queues,0,None)
    
    def init_queues(self,siz):
        if isinstance(siz,int):
            self.queues=np.ones(self.dims)*siz
        elif isinstance(siz,np.ndarray):
            assert siz.shape==self.q_num
            self.queues=siz
        else:
            print('input an integer or an array equal to length of queues')
            raise
    
    def step(self,fire=None,fancy=False,**kwargs):
                  
        self.time+=1       
        ##decide if market on fire  
        if fire is None: self.fire=self.decide_fire()
        else: self.fire=fire
        

        #process other arrivals
        self.process_arrivals()

        #process departure
        self.process_departures()

        #calculate position
        self.position-=self.departures
        
        #calculate outcome
        
        # if self.position.any()<=0:
            # if self.fire: self.EWOP=True
            # else: self.Exit=True
            # self.inqueue=False
            
        left_queue=(self.position<0).any(axis=1)
        
        #fill the EWOPs according to fire.
        self.EWOP=np.zeros((self.instances,1))
        self.EWOP[np.logical_and(self.fire,left_queue)]=1
        
        #fill the exits according to not EWOP and left_queue
        self.Exit=np.zeros((self.instances,1))
        self.Exit[np.logical_and(~self.fire,left_queue)]=1
        
        #calculate in queues
        self.inqueue=np.ones((self.instances,1))
        self.inqueue[left_queue]=0
               

        if fancy: return self.pretty_state    
        else: return self.state

    def enter_queue(self,q_num):
        
        #deduct from those instances where in queue
        self.queues-=~self.position.mask
        
        self.position=ma.masked_array(np.zeros(self.dims),np.ones(self.dims))
        #advanced indexing
        self.position.mask[np.arange(self.instances),q_num-1]=0
                                              
        add_pos=np.zeros(self.q_num)
        add_pos[q_num-1]=1
       
        #record new addition to queue
        self.queues+=add_pos
                                   
        #record new position at back of queue
        self.position+=self.queues 
        
        #record that in a queue
        self.inqueue=q_num>0