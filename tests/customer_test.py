import numpy as np
from intent.customer  import Customer

departure_rate=[2,2]
arrival_rate=[2,2]
fire_rate=0.1
rw_exit_pay=1
rw_exit_no_pay=10
time_limit=102


def test_basic():
    name='hal'
    cust=Customer(name,departure_rate,arrival_rate,fire_rate,time_limit=time_limit,
                    rw_exit_pay=rw_exit_pay,rw_exit_no_pay=rw_exit_no_pay)
    for i in range(time_limit+2):
        
        prev_state=cust.pretty_state
        action=cust.action_space.sample()
        state,reward,done,info=cust.step(action)
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