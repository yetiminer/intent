from intent.fishauction import FishAuction
from intent.fishauctionlight import FishMarketLight
from pytest import approx
from collections import namedtuple
import numpy as np
from copy import deepcopy
from intent.fishauctionlight import FML_state

StateFormat=namedtuple('StateFormat',['exit','ewop','paid','in_queue','q_pos','q_len'])  

def test_clone():
    departure_rate=[2,2]
    arrival_rate=[2,2]
    fire_rate=0
    
    
    fa=FishAuction(departure_rate,arrival_rate,fire_rate)
    fa.init_queues(3)
    
    state=fa.step()
    
    fa2=FishAuction.FA_constructor(departure_rate,arrival_rate,fire_rate,state['queues'],state['q_waits'])
    
    assert fa2==fa
    
def test_clone_m():
    lf1=FishMarketLight()
    _=lf1.step()
    lf1.enter_queue(np.array(1))
    
    lf2=lf1.copy()
    
    for a,b in zip(lf1.get_params,lf2.get_params):
            
        try: 
            assert a.all()==b.all()
        except AssertionError:
            print(a,b)
            raise AssertionError
        except AttributeError:
            assert a==b
            
    for a,b in zip(lf1.pretty_state,lf2.pretty_state):
        if isinstance(a,np.ma.core.MaskedArray):
            assert np.ma.allequal(a,b)
        else:
            try: 
                assert a.all()==b.all()
            except AssertionError:

                print(a,b)
                raise AssertionError
            except AttributeError:
                assert a==b

def test_clone_m_arr():
    #test to see cloned fa array form is the same
    params=dict(
        arrival_rate=np.ones((3,2))*2,
        departure_rate=np.ones((3,2))*2,
        fire_rate=np.array([0.1,0.1,0.1]),
        )
    lf1=FishMarketLight(**params)
    _=lf1.step()
    lf1.enter_queue(np.array(1))
    
    lf2=lf1.copy()
    
    for a,b in zip(lf1.get_params,lf2.get_params):
            
        try: 
            assert a.all()==b.all()
        except AssertionError:
            print(a,b)
            raise AssertionError
        except AttributeError:
            assert a==b
            
    for a,b in zip(lf1.pretty_state,lf2.pretty_state):
        if isinstance(a,np.ma.core.MaskedArray):
            assert np.ma.allequal(a,b)
        else:
            try: 
                assert a.all()==b.all()
            except AssertionError:

                print(a,b)
                raise AssertionError
            except AttributeError:
                assert a==b

    
def test_fire_rate():
    departure_rate=[2,2]
    arrival_rate=[2,2]
    fire_rate=0.1

    fa=FishAuction(departure_rate,arrival_rate,fire_rate)
    
    fa.init_queues(3)
    state=fa.get_state(0)
    counter=0
    its=10000

    for i in range(its):
        state=fa.step(fancy=False)
        if state['fire']: 
            counter+=1

    assert counter/its==approx(fire_rate,abs=0.05)
    
    
def test_fire_rate_m():
    d=dict(departure_rate=[2,2],
    arrival_rate=[2,2],
    fire_rate=0.1)

    fa=FishMarketLight(**d)
    
    fa.init_queues(3)
    #state=fa.get_state(0)
    counter=0
    its=10000

    for i in range(its):
        state=fa.step(fancy=True)
        if state.fire: 
            counter+=1

    assert counter/its==approx(d['fire_rate'],abs=0.05)
    
def test_fire_rate_m_arr(instances=10):
    #tests the light fish market for array input
    queues=2
    fire_rate=np.ones(instances)*0.1
    fire_rate[0]=0.05
    params=dict(
    arrival_rate=np.ones((instances,queues))*2,
    departure_rate=np.ones((instances,queues))*2,
    fire_rate=fire_rate)
    

    fa=FishMarketLight(**params)
    
    fa.init_queues(3)
    #state=fa.get_state(0)
    counter=np.zeros(instances)
    its=5000

    for i in range(its):
        state=fa.step(fancy=True)
         
        counter+=state.fire.flatten()
    #return counter/its
    try:
        assert counter/its==approx(fire_rate,abs=0.05) 
    except:
        print("assertion failed")
        print(counter/its-fire_rate)
        raise AssertionError
    
def make_change(random_action,instances,queues):
    #creates array of q entries
    q_row_idx=np.arange(instances)
    queues_entered=random_action-1
    valid_queues=random_action-1>=0
    fancy_index=(q_row_idx[valid_queues],queues_entered[valid_queues])
    queue_pos_guess=np.zeros((instances,queues))
    queue_pos_guess[fancy_index]=1
    return queue_pos_guess

def setup_instances(instances=10,queues=2):
    queues=2
    fire_rate=np.ones(instances)*0.1
    fire_rate[0]=0.05
    params=dict(
    arrival_rate=np.ones((instances,queues))*2,
    departure_rate=np.ones((instances,queues))*2,
    fire_rate=fire_rate)
    
    fa=FishMarketLight(**params)
    return fa

def no_position_new_position(random_action,prev_state,new_state):
    #the case where there is no position, then a new queue is entered
    
    #get the rows where there is no starting position
    a=prev_state.q_pos.mask
    b=np.array([True,True])
    starting_no_position=(a[:,None]==b).all(-1).any(-1)
    
    queues=a.shape[1]
    
    relevant_actions=random_action[starting_no_position]-1
    no_new_adds=starting_no_position.sum()
    new_adds=np.zeros((no_new_adds,queues))
    idx=np.arange(no_new_adds)
    valid_actions=relevant_actions>-1
    new_adds[idx[valid_actions],relevant_actions[valid_actions]]=1

    assert (prev_state.q_len[starting_no_position]+new_adds==new_state.q_len[starting_no_position]).all()
    print("All clear")

def no_action_no_change(random_action,prev_state,new_state):
    no_changes=random_action==0
    assert((prev_state.q_pos[no_changes]==new_state.q_pos[no_changes]).all())

def test_queue_logic(instances=10):
    #tests the light fish market for array input
    instances=10
    queues=2   
    fa=setup_instances(instances,queues)
    
    fa.init_queues(3)
    
    random_action=np.random.randint(0,queues+1,10)
    prev_state=deepcopy(FML_state(*fa.pretty_state))
    #no positions before
    assert (prev_state.q_pos.mask==np.ones((instances,queues))).all()
    
    #check that the initiation works
    assert (fa.queues==np.ones((instances,queues))*3).all()
    
    prev_q_len=fa.queues.copy()    
    fa.enter_queue(random_action)    
    queue_pos_guess=make_change(random_action,instances,queues)

    
    #check that the entered queues work first time around
    assert (queue_pos_guess+prev_q_len==fa.queues).all()
    
    prev_state=deepcopy(FML_state(*fa.pretty_state))
    
    random_action=np.random.randint(0,queues+1,10)
    fa.enter_queue(random_action)
    new_state=FML_state(*fa.pretty_state)
    
    #check case of new positions
    no_position_new_position(random_action,prev_state,new_state)
    #check case of zero actions
    no_action_no_change(random_action,prev_state,new_state)
    
    return random_action, prev_state,new_state