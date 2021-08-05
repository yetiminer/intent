from intent.fishauction import FishAuction, FA_tests
from intent.fishauctionlight import FishMarketLight
from intent.stateencoders import StateEncoder,StateEncoderArray,StateFormat, FML_State
from intent.TQ import TillQueue
import gym
from gym.spaces import Space, MultiBinary, Discrete
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
        
    
    
    def __init__(self,name,departure_rate,arrival_rate,fire_rate,time_limit=100,rw_exit_pay=1,rw_exit_no_pay=10,fm='standard'):
        self.time=0
        self.cust_id=0
        self.time_limit=time_limit
        self.departure_rate=departure_rate
        self.arrival_rate=arrival_rate
        self.fire_rate=fire_rate
        
        #number of queues
        self.q_num=len(departure_rate)
        

        
        #record the rewards
        self.rw_exit_pay=rw_exit_pay
        self.rw_exit_no_pay=rw_exit_no_pay
               
        self.observation_space=ObservationSpace(3) ###### not correct
        self.action_space=MultiDiscrete(3)
        self.current_queue=0
        
        #select the correct state encoder
        self.se=self.SE[fm]
        
        self.fa_name=fm
  
        self.initiate_state()
    
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
        self.state=self.se.encode_state(self.raw_state)
        #self.pretty_state=StateFormat(*self.state)
    
    def _return_array(self):
        return np.array(self.state)
        
        
    def step(self,action):        
        self.time+=1
          
        self.process_action(action)
        
        self.raw_state=self.fa.step(fancy=False,cust_ID=0)
        self.state=self.se.encode_state(self.raw_state)
        #self.pretty_state=StateFormat(*self.state)
        
        reward=self.get_rewards()
        done=self.done_check()
        info=None
        
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
        
        return reward
        
class ObservationSpace(MultiBinary):
        def contains(self, x):
            if isinstance(x, (list,tuple)):
                x = np.array(x)  # Promote list to array for contains check
            return ((x==0) | (x==1)).all()        

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

       def sample(self,i=1):
            return np.random.randint(0,self.n,i)

#StateFormat=namedtuple('StateFormat',['exit','ewop','paid','in_queue','q_pos','q_len'])        
        

        

        