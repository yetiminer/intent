from fishauction import FishAuction, FA_tests
from TQ import TillQueue
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
    
    def __init__(self,name,departure_rate,arrival_rate,fire_rate,time_limit=100,rw_exit_pay=1,rw_exit_no_pay=10):
        self.time=0
        self.cust_id=0
        self.time_limit=time_limit
        self.departure_rate=departure_rate
        self.arrival_rate=arrival_rate
        self.fire_rate=fire_rate
        
        #record the rewards
        self.rw_exit_pay=rw_exit_pay
        self.rw_exit_no_pay=rw_exit_no_pay
               
        self.observation_space=ObservationSpace(3) ###### not correct
        self.action_space=Discrete(3)
        self.current_queue=0
        self.se=StateEncoder()
        
        self.initiate_state()
        
    def reset(self):
        self.initiate_state()
        return self._return_array()
    
    def initiate_state(self):
        self.time=0
        self.fa=FishAuction(self.departure_rate,self.arrival_rate,self.fire_rate)
        self.fa.init_queues(10)
        self.raw_state=self.fa.get_state(self.cust_id)
        self.state=self.se.encode_state(self.raw_state)
        self.pretty_state=StateFormat(*self.state)
    
    def _return_array(self):
        return np.array(self.state)
        
        
    def step(self,action):        
        self.time+=1
          
        self.process_action(action)
        
        self.raw_state=self.fa.step(fancy=False,cust_ID=0)
        self.state=self.se.encode_state(self.raw_state)
        self.pretty_state=StateFormat(*self.state)
        
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
        #if action==0: print('waiting')
        if action>0:
            
            if self.pretty_state.in_queue:
                #print('swapping queue')
                self.fa.swap_queue(self.cust_id,action)
            else:
                #print("entering queue")
                self.fa.new_arrival(action,self.cust_id)
                self.current_queue=action
            
        
        
    def get_rewards(self,state=None):
        if state is None: state=self.pretty_state
        reward=0
        if state.exit:
            if  state.ewop: 
                reward=self.rw_exit_no_pay
                print("fire!")
            elif state.paid: reward=self.rw_exit_pay
        
        return reward
        
class ObservationSpace(MultiBinary):
        def contains(self, x):
            if isinstance(x, (list,State)):
                x = np.array(x)  # Promote list to array for contains check
            return ((x==0) | (x==1)).all()        

StateFormat=namedtuple('StateFormat',['exit','ewop','paid','in_queue','q_1_pos','q_2_pos'])        
        
class StateEncoder():
    def  __init__(self):
        pass
    
    @staticmethod
    def check_exit(state):
        exit=False
        ewop=False
        paid=False
        
        if state['exits']!=[]:
            for k, li in state['exits'][0].items():
                if 0 in li[0,:]:
                    exit=True
                    if state['exits'][1]=='Paid': paid=True
                    elif state['exits'][1]=='EWOP': ewop=True
                    break

        return np.array((exit,ewop,paid))
    
    @staticmethod
    def get_position(state):
        wi=state['where']   
        wia=np.zeros(3)#####!!!!
        if wi!=(0,0): wia[0]=1
        #remember the first queue is labelled 1
        wia[wi[0]]=wi[1] #returns zeros if given zeros
        return wia
    
    @staticmethod
    def encode_state(state):
        return np.hstack((StateEncoder.check_exit(state),StateEncoder.get_position(state)))