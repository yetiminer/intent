from intent.fishauction import FishAuction, FA_tests
from intent.fishauctionlight import FishMarketLight
from intent.stateencoders import StateEncoder,StateEncoderArray,StateFormat, FML_State
from intent.TQ import TillQueue
import gym
from gym.spaces import Space, MultiBinary, Discrete, Box
import numpy as np
from collections import namedtuple


class  Customer(gym.Env):
   
    actions={
        0:'wait',
        1:'enter_queue_1',
        2:'enter_queue_2'
    }
    
    FM={'standard':FishAuction,
        'array':FishMarketLight}
        
    SE={'standard':StateEncoder,
        'array':StateEncoderArray}
        
    StateFormat={'standard':StateFormat,'array':FML_State}
    
    def __init__(self,name,departure_rate,arrival_rate,fire_rate,time_limit=100,rw_exit_pay=1,rw_exit_no_pay=10,fm='standard'):
        
        self.name=name
        self.time=0
        self.cust_id=0
        self.time_limit=time_limit
        self.departure_rate=departure_rate
        self.arrival_rate=arrival_rate
        self.fire_rate=fire_rate
        
        #number of queues
        if isinstance(departure_rate,np.ndarray):
            self.q_num=departure_rate.shape[1]
        elif isinstance(departure_rate,(list,tuple)):
            self.q_num=len(departure_rate)
        

        
        #record the rewards
        self.rw_exit_pay=rw_exit_pay
        self.rw_exit_no_pay=rw_exit_no_pay
               

        
        #select the correct state encoder
        self.se=self.SE[fm]
        self.state_format=self.StateFormat[fm]
        
        self.fa_name=fm
        
        self.flatten=False
        if self.instances==1:
            self.flatten=True
            
        #self.observation_space=ObservationSpace(3) ###### not correct
        if self.instances==1: obs_dims=(4+4*self.q_num,)
        else: obs_dims=(self.instances,4+4*self.q_num)
        self.observation_space=ObservationSpace(low=0,high=30,shape=obs_dims,dtype=np.int32)
        self.action_space=MultiDiscrete(3)
        self.current_queue=0 #not used in array form
  
        self.initiate_state()
    
    def _info(self):
        return dict(name=self.name,departure_rate=self.departure_rate,arrival_rate=self.arrival_rate,fire_rate=self.fire_rate,
        time_limit=self.time_limit,rw_exit_pay=self.rw_exit_pay,rw_exit_no_pay=self.rw_exit_no_pay,fm=self.fa_name)
    
    def __repr__(self):
        return str(self._info())
        
    
    
    
    @property  
    def instances(self):
        instances=1
        if self.fa_name=='array':
           dims=self.departure_rate.shape
           instances=dims[0]
        return  instances
        
    @property
    def rewards(self):
        return f'Exit paying: {self.rw_exit_pay} Exit without paying: {self.rw_exit_no_pay}'
        
    def reset(self):
        self.initiate_state()
        return self._return_array()
    
    @property
    def pretty_state(self):
        state=self.state
        return self.make_pretty_state(state)
        # if self.fa_name=='standard':
            # return StateFormat(*state[0:5],state[5:5+self.q_num],state[5+self.q_num:])
        # else:
            # return FML_State(*[np.expand_dims(a,1) if len(a.shape)==1 else a for a in self.fa.state])
            
    def make_pretty_state(self,state):
        if self.fa_name=='standard':
            return StateFormat(*state[0:5],state[5:5+self.q_num],state[5+self.q_num:])
        else:
            return FML_State(*[np.expand_dims(a,1) if len(a.shape)==1 else a for a in self.fa.state])
    
    def initiate_state(self):
        
        self.time=0
        self.fa_name
        
        self.fa=self.FM[self.fa_name](departure_rate=self.departure_rate,arrival_rate=self.arrival_rate,fire_rate=self.fire_rate)
        self.fa.init_queues(10)
        
        self.raw_state=self.fa.state
        self.state=self.se.encode_state(self.raw_state,flatten=self.flatten)
        #self.pretty_state=StateFormat(*self.state)
    
    def _return_array(self):
        if self.fa_name=='standard':
            return np.array(self.state)
        else:
            return self.state
        
        
    def step(self,action):        
        self.time+=1
          
        self.process_action(action)
        
        self.raw_state=self.fa.step(fancy=False,cust_ID=0)
        self.state=self.se.encode_state(self.raw_state,flatten=self.flatten)
        #self.pretty_state=StateFormat(*self.state)
        
        reward=self.get_rewards()
        done=self.done_check()
        info={}
        
        return self.state,reward,done,info
        
    def done_check(self):    
        #check that time isn't up
        done=False
        if  self.time>=self.time_limit:
            done=True
        return done
            
    def process_action(self,action):
        #check action is valid
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        if self.fa_name=='standard':
            if action>0:
                
                if self.pretty_state.in_queue:
                    #print('swapping queue')
                    self.fa.swap_queue(self.cust_id,action)
                else:
                    #print("entering queue")
                    self.fa.new_arrival(action,self.cust_id)
                    self.current_queue=action
        elif self.fa_name=='array':
            #swapping or entering or doing nothing all handled at fa level.
            self.fa.enter_queue(action)
            
        
    def get_rewards(self,state=None):
        if state is None: state=self.pretty_state
        reward=np.zeros(self.instances)
        #who left without paying
        ewops_tf=np.logical_and(state.exit,state.ewop==1).flatten()
        reward[ewops_tf]=self.rw_exit_no_pay
        #who left and paid
        exit_tf=np.logical_and(state.exit,state.ewop==0).flatten()
        reward[exit_tf]=self.rw_exit_pay
        
        #workaround for stablebaselines which needs float as a reward.
        if self.instances==1: reward=reward[0]
        
        return reward
        
    @staticmethod   
    def make_customer_array(cust,instances):
        #takes a single customer instance and makes an array of it
    
        #this is only going to work if it is a customer array
        assert type(cust)==Customer
        assert cust.fa_name=='array'
        
        name=cust.name+' '+str(instances)+' array'
        departure_rate=np.vstack([cust.arrival_rate for i in range(instances)])
        arrival_rate=np.vstack([cust.departure_rate for i in range(instances)])
        fire_rate=np.vstack([cust.fire_rate for i in range(instances)])
        time_limit=cust.time_limit
        rw_exit_pay=cust.rw_exit_pay
        rw_exit_no_pay=cust.rw_exit_no_pay
        fm='array'
        return Customer(name,departure_rate,arrival_rate,fire_rate,time_limit,rw_exit_pay,rw_exit_no_pay,fm=fm)
    
class ObservationSpace(Box):
        def contains(self, x):
            if isinstance(x, list):
                x = np.array(x)  # Promote list to array for contains check
            return (
                x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)
            )       

class MultiDiscrete(Discrete):
       def contains(self,x):       
            if isinstance(x,(int,list,tuple,np.ndarray,np.generic)):
                x=np.array(x)
            else:
                print('action should be list like or array')
                return False
            try:
                assert x.dtype.char in np.typecodes["AllInteger"]
            except AssertionError:
                print("Array should only contain integers")
                return False
            return np.logical_and(x>=0 ,x<self.n).all()

       def sample(self,i=1,instances=1):
            return np.random.randint(0,self.n,i)

#StateFormat=namedtuple('StateFormat',['exit','ewop','paid','in_queue','q_pos','q_len'])        
        

        

        