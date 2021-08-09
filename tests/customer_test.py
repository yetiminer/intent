import numpy as np
from intent.customer  import Customer
from copy import deepcopy
from intent.stateencoders import FML_State
import pandas as pd

departure_rate=[2,2]
arrival_rate=[2,2]
fire_rate=0.1
rw_exit_pay=1
rw_exit_no_pay=10
time_limit=102
name='hal'

def test_basic():
    
    cust=Customer(name,departure_rate,arrival_rate,fire_rate,time_limit=time_limit,
                    rw_exit_pay=rw_exit_pay,rw_exit_no_pay=rw_exit_no_pay)
    for i in range(time_limit+2):
        
        prev_state=cust.pretty_state
        action=cust.action_space.sample()
        state,reward,done,info=cust.step(action[0])
        curr_state=cust.pretty_state
        try:
            #check that the thing times out when expected
            t=1
            if cust.time<time_limit: assert not(done) 
            t=2
            if cust.time==time_limit: assert done

            #check  reward for exiting wop
            if curr_state.ewop:
                t=3
                assert reward==rw_exit_no_pay
                t=4
                assert curr_state.exit
            if reward==rw_exit_no_pay:
                t=5
                assert curr_state.ewop*curr_state.exit==1    
            #check reward for exiting
            if curr_state.paid:
                t=6
                assert reward==rw_exit_pay
                t=7
                assert curr_state.exit
            if reward==rw_exit_pay:
                t=8
                assert curr_state.paid*curr_state.exit==1    

            #check no reward for not exiting
            if curr_state.exit*curr_state.ewop*curr_state.paid==1:
                t=9
                assert reward==0
            #check not paid and paid
            t=10
            assert curr_state.ewop*curr_state.paid==0
            
        except AssertionError:
            print(prev_state,reward,state,cust.time,done,t)
            
def test_enter_queue():
    cust=Customer(name,departure_rate,arrival_rate,fire_rate,time_limit=100,rw_exit_pay=1,rw_exit_no_pay=10)
    #enter queue q_num
    for q_num in [1,2]:
    
        state,reward,done,info=cust.step(q_num)

        while True:
            #either we're still in queue 1
            if cust.pretty_state.in_queue==1:
                for qn,q in cust.raw_state['queues'].items():
                    if qn==q_num:
                        assert 0 in q
                    else:
                        assert 0 not in q
                #just wait        
                state,reward,done,info=cust.step(0)
            #or have exited
            else:
                #check exit status
                assert cust.pretty_state.exit==1
                #check paid or not paid
                assert cust.pretty_state.ewop==1 or cust.pretty_state.paid==1
                #join the queue again
                state,reward,done,info=cust.step(q_num)
            
            
            if done: break
                
        assert cust.time>=cust.time_limit
        
def test_cust_array():
    departure_rate=np.array([[2,2],[2,2],[2,2]])
    arrival_rate=departure_rate
    fire_rate=np.array((0.1,0.05,0.05))
    
    name='hal'
    cust=Customer(name,departure_rate,arrival_rate,fire_rate,time_limit=100,rw_exit_pay=1,rw_exit_no_pay=10,fm='array')

    for i in range(time_limit+2):
        
        prev_state=deepcopy(cust.pretty_state)
        actions=cust.action_space.sample(cust.instances)
        state,reward,done,info=cust.step(actions)
        curr_state=cust.pretty_state
        try:
            #check that the thing times out when expected
            t=1
            if cust.time<time_limit: assert not(done) 
            t=2
            if cust.time==time_limit: assert done

            #check  reward for exiting wop
            tf_ewop=(curr_state.ewop==1).flatten()
            t=3
            assert (reward[tf_ewop]==rw_exit_no_pay).all()
            
            
            t=4
            assert curr_state.exit[tf_ewop].all()
            tf_ewop=reward==rw_exit_no_pay #assumes that reward ewop is unique
            t=5
            assert np.logical_and(curr_state.ewop,curr_state.exit).all()    
            #check reward for exiting
            tf_p=np.logical_and(curr_state.exit,~tf_ewop)
            t=6
            assert (reward[tf_p]==rw_exit_pay).all()
            t=7
            assert curr_state.exit[tf_p].all()
            tf_pay_r=reward==rw_exit_pay
            t=8
            assert (tf_pay_r==tf_p).all()   

            #check no reward for not exiting
            tf_iq=curr_state.inqueue==1
            t=9
            assert (reward[tf_iq]==0).all()

            #check exits not in queue
            tf_exit=curr.state.exit==1
            assert (curr_state.inqueue[tf_exit]==0).all()
            
        except AssertionError:
            print(prev_state,reward,state,cust.time,done,t)

def make_state_names(self,q_num,others,prev=False):
    fields=self._fields
    if prev:
        
        state_names=list(fields[:4])+[l+'_'+str(i) for l in fields[4:] for i in range(1,q_num+1)]
        state_names=others+state_names+["_"+l for l in state_names]
    else:
        state_names=others+list(fields[:4])+[l+'_'+str(i) for l in fields[4:] for i in range(1,q_num+1)]
    return state_names

def messy_classifier(inst1):
    inst1['diff']=(inst1['_q_len_1']+inst1['q_arrivals_1']-inst1['q_departures_1']).clip(0,np.inf)-inst1['q_len_1']
    
    swap_from_1_to_2=(inst1['_in_queue']==1)&(inst1['_q_pos_1']>0)&(inst1['Action']==2)&(inst1['in_queue']==1)
    go_to_back_of_1=(inst1['_in_queue']==1)&(inst1['_q_pos_1']>0)&(inst1['Action']==1)&(inst1['in_queue']==1)
    enter_1=(inst1['_in_queue']==0)&(inst1['Action']==1)&(inst1['in_queue']==1)
    enter_1_immediately_exit=(inst1['Action']==1)&(inst1['exit']==1)
    enter_1_EWOP=(inst1['Action']==1)&(inst1['fire']==1)

    swap_from_2_to_1=(inst1['_in_queue']==1)&(inst1['_q_pos_2']>0)&(inst1['Action']==1)&(inst1['in_queue']==1)
    go_to_back_of_2=(inst1['in_queue']==1)&(inst1['_q_pos_2']>0)&(inst1['Action']==2)&(inst1['in_queue']==1)
    enter_2=(inst1['_in_queue']==0)&(inst1['Action']==2)&(inst1['in_queue']==1)
    enter_2_immediately_exit=(inst1['Action']==2)&(inst1['exit']==1)
    enter_2_EWOP=(inst1['Action']==2)&(inst1['fire']==1)

    
    
    inst1.loc[:,'Class']="Unknown"
    inst1['Class'].loc[swap_from_1_to_2]="Swap 1 to 2"
    inst1['Class'].loc[go_to_back_of_1]="go_to_back_of_1"
    inst1['Class'].loc[enter_1_immediately_exit]="enter_1_exit"
    inst1['Class'].loc[enter_1]="enter 1"
    inst1['Class'].loc[enter_1_EWOP]="enter_1_EWOP"

    inst1['Class'].loc[swap_from_2_to_1]="Swap 2 to 1"
    inst1['Class'].loc[go_to_back_of_2]="go_to_back_of_2"
    inst1['Class'].loc[enter_2]="enter_2"
    inst1['Class'].loc[enter_2_immediately_exit]="enter_2_exit"
    inst1['Class'].loc[enter_2_EWOP]="enter_2_EWOP"
    
    return inst1

def check_on_diffs(inst1):
    tf1=inst1['diff']!=0
    assert ((inst1.Class=="Unknown")&tf1).all()==False
    return 

def exit_when_expected_when_already_in_queue(df,q_num):

    for i in range(1,q_num+1):

        action1=df.Action==i
        was_in_q1=df['_q_pos_'+str(i)]>0
        departure_due=(df['_q_len_'+str(i)]+df['q_arrivals_' + str(i)]-df['q_departures_'+str(i)])<0
        not_ewop=df.ewop==0

        #['Action','ewop','exit','_in_queue','in_queue','_q_len_1','q_arrivals_1','q_departures_1','q_len_1','Class','diff']

        (df[action1&was_in_q1&departure_due&not_ewop].exit==1).all()


def test_cust_array_logic():
    time_limit=1000
    departure_rate=np.array([[2,2],[2,2],[2,2]])
    arrival_rate=departure_rate
    fire_rate=np.array((0.1,0.05,0.05))
    
    
    name='hal'
    cust=Customer(name,departure_rate,arrival_rate,fire_rate,time_limit=time_limit,rw_exit_pay=1,rw_exit_no_pay=10,fm='array')
    instances=np.arange(1,cust.instances+1).reshape(cust.instances,1)
    
    history=[]
    state=cust.state
    
    for i in range(time_limit+2):
        
        prev_state=deepcopy(state)
        actions=cust.action_space.sample(cust.instances)
        state,reward,done,info=cust.step(actions)
        
                
        history.append(np.hstack([instances,actions.reshape(cust.instances,1),state,prev_state]))
    
    fields=make_state_names(FML_State,2,['Instance','Action'],prev=True)
    df=pd.DataFrame(np.vstack(history),columns=fields)
    
    men=df.mean()
    
    try:
        assert men.fire>=men.ewop
    except AssertionError:
        print("Fire rate should be as large as EWOP")
        raise AssertionError

    try:
        assert (men>0).all()
    except AssertionError:
        print("Figures should be non-zero on average")
        raise AssertionError
    
    # try:
        # assert (df.min()>=0).all() 
    # except AssertionError:
        # print("Non negative data required")
        # return df
        # #raise AssertionError
        
    df=messy_classifier(df)
    
    check_on_diffs(df)
    
    exit_when_expected_when_already_in_queue(df,cust.q_num)
    
    return df
    
def test_make_customer_array():
    #make customer array takes a single customer instance and makes an array version of it. 
    departure_rate=np.array([[2,2]])
    arrival_rate=departure_rate
    fire_rate=np.array((0.1))

    name='hal'
    cust=Customer(name,departure_rate,arrival_rate,
                  fire_rate,time_limit=100,rw_exit_pay=1,rw_exit_no_pay=10,
                  fm='array')
    
    
    instances=10
    cust_array=Customer.make_customer_array(cust,instances)
    assert cust_array.instances==instances

    for i in range(instances):
        assert (cust_array.departure_rate[i,:]==cust.departure_rate).all() 
        assert (cust_array.arrival_rate[i,:]==cust.arrival_rate).all() 
        assert (cust_array.fire_rate[i,:]==cust.fire_rate).all() 


    assert cust_array.time_limit==cust.time_limit
    assert cust_array.rw_exit_pay==cust.rw_exit_pay
    assert cust_array.rw_exit_no_pay==cust.rw_exit_no_pay    
    assert cust_array.time_limit==cust.time_limit
    assert cust_array.fa_name=='array'
    
            