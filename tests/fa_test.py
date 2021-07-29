from intent.fishauction import FishAuction
from intent.fishauctionlight import FishMarketLight
from pytest import approx
from collections import namedtuple
import numpy as np

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
         
        counter+=state.fire
    #return counter/its
    try:
        assert counter/its==approx(fire_rate,abs=0.05) 
    except:
        print("assertion failed")
        print(counter/its-fire_rate)
        raise AssertionError
    